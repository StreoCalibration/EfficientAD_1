import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional


def compute_hard_feature_loss(student_features: torch.Tensor, 
                            teacher_features: torch.Tensor) -> torch.Tensor:
    """
    Compute Hard Feature Loss as described in EfficientAD paper
    
    Args:
        student_features: Student network output [B, C, H, W]
        teacher_features: Teacher network output [B, C, H, W]
        
    Returns:
        Hard feature loss scalar
    """
    # Compute channel-wise squared difference
    diff = torch.pow(student_features - teacher_features, 2)
    
    # Flatten spatial dimensions while keeping batch and channel
    B, C, H, W = diff.shape
    diff_flat = diff.view(B, C, -1)  # [B, C, H*W]
    
    # Find 99.9 percentile threshold across all pixels and channels
    diff_all = diff_flat.view(-1)  # Flatten everything
    k = int(0.999 * diff_all.numel())
    d_hard, _ = torch.kthvalue(diff_all, k)
    
    # Create mask for hard examples (above threshold)
    hard_mask = (diff_flat >= d_hard).float()
    
    # Compute mean of hard examples only
    hard_pixels = diff_flat * hard_mask
    hard_sum = torch.sum(hard_pixels)
    hard_count = torch.sum(hard_mask)
    
    # Avoid division by zero
    if hard_count > 0:
        hard_loss = hard_sum / hard_count
    else:
        hard_loss = torch.mean(diff_flat)
    
    return hard_loss


def compute_pretraining_penalty(student_model: nn.Module, 
                               imagenet_batch: torch.Tensor) -> torch.Tensor:
    """
    Compute pretraining penalty to preserve ImageNet knowledge
    
    Args:
        student_model: Student PDN model
        imagenet_batch: Batch of ImageNet/ImageNette images [B, C, H, W]
        
    Returns:
        Pretraining penalty scalar
    """
    # Forward pass through student model
    features = student_model(imagenet_batch)
    
    # Compute Frobenius norm squared
    # Frobenius norm is sqrt of sum of squared elements
    frobenius_squared = torch.sum(torch.pow(features, 2))
    
    # Normalize by number of elements
    penalty = frobenius_squared / features.numel()
    
    return penalty


def compute_autoencoder_loss(student_features: torch.Tensor,
                           autoencoder_features: torch.Tensor) -> torch.Tensor:
    """
    Compute Student-Autoencoder loss
    
    Args:
        student_features: Student network output [B, C, H, W]
        autoencoder_features: Autoencoder output [B, C, H, W] 
        
    Returns:
        Student-autoencoder loss scalar
    """
    return F.mse_loss(student_features, autoencoder_features)


class EfficientADLoss(nn.Module):
    """Combined loss function for EfficientAD training"""
    
    def __init__(self, 
                 st_weight: float = 1.0,
                 ae_weight: float = 1.0, 
                 penalty_weight: float = 1.0):
        super().__init__()
        self.st_weight = st_weight
        self.ae_weight = ae_weight
        self.penalty_weight = penalty_weight
    
    def forward(self, 
                outputs: Dict[str, torch.Tensor],
                imagenet_batch: torch.Tensor,
                student_model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Compute total loss
        
        Args:
            outputs: Model outputs dict containing student/teacher/autoencoder features
            imagenet_batch: ImageNet batch for penalty computation
            student_model: Student model for penalty computation
            
        Returns:
            Dictionary with individual losses and total loss
        """
        student_features = outputs['student_features']
        teacher_features = outputs['teacher_features']
        autoencoder_features = outputs['autoencoder_features']
        
        # Compute individual losses
        st_loss = compute_hard_feature_loss(student_features, teacher_features)
        ae_loss = compute_autoencoder_loss(student_features, autoencoder_features)
        penalty_loss = compute_pretraining_penalty(student_model, imagenet_batch)
        
        # Weighted combination
        total_loss = (self.st_weight * st_loss + 
                     self.ae_weight * ae_loss + 
                     self.penalty_weight * penalty_loss)
        
        return {
            'total_loss': total_loss,
            'st_loss': st_loss,
            'ae_loss': ae_loss, 
            'penalty_loss': penalty_loss
        }