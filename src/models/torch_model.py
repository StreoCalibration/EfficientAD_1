import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, Dict, Any


class PDN(nn.Module):
    """Patch Descriptor Network - Lightweight feature extractor"""
    
    def __init__(self, out_channels: int = 384, padding: bool = False):
        super().__init__()
        self.out_channels = out_channels
        
        # 4 Conv layers as described in paper
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=1 if padding else 0),
            nn.ReLU(inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1 if padding else 0),
            nn.ReLU(inplace=True)
        )
        
        self.layer3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1 if padding else 0),
            nn.ReLU(inplace=True)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x) 
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class Autoencoder(nn.Module):
    """Simplified Autoencoder that outputs same size as PDN"""
    
    def __init__(self, out_channels: int = 384):
        super().__init__()
        self.out_channels = out_channels
        
        # Simple encoder-decoder that matches PDN architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder to match PDN output
        self.decoder = nn.Sequential(
            nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        output = self.decoder(encoded)
        return output


class EfficientADModel(nn.Module):
    """EfficientAD Model combining Student-Teacher and Autoencoder"""
    
    def __init__(self, model_size: str = 'S', out_channels: int = 384, 
                 teacher_weights: str = None):
        super().__init__()
        self.model_size = model_size
        self.out_channels = out_channels
        
        # Determine output channels based on model size
        if model_size == 'S':
            self.out_channels = 384
        elif model_size == 'M':
            self.out_channels = 768
        else:
            raise ValueError(f"Unsupported model size: {model_size}")
        
        # Initialize networks
        self.teacher = PDN(out_channels=self.out_channels, padding=True)
        self.student = PDN(out_channels=self.out_channels, padding=True)
        self.autoencoder = Autoencoder(out_channels=self.out_channels)
        
        # Freeze teacher network
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Initialize teacher with pretrained weights if provided
        if teacher_weights:
            self._init_teacher_from_pretrained(teacher_weights)
        else:
            self._init_teacher_randomly()
    
    def _init_teacher_randomly(self):
        """Initialize teacher with random weights (for synthetic data testing)"""
        pass  # Default random initialization
    
    def _init_teacher_from_pretrained(self, weights_path: str):
        """Initialize teacher from pretrained model weights"""
        # This would load pretrained weights from WideResNet-101 or EfficientNet
        # For now, use random initialization
        pass
    
    def forward(self, x):
        """Forward pass returning student, teacher and autoencoder outputs"""
        with torch.no_grad():
            teacher_features = self.teacher(x)
        
        student_features = self.student(x)
        autoencoder_features = self.autoencoder(x)
        
        return {
            'student_features': student_features,
            'teacher_features': teacher_features, 
            'autoencoder_features': autoencoder_features
        }
    
    def compute_anomaly_map(self, x, map_size: Tuple[int, int] = None):
        """Compute anomaly heatmap for inference"""
        outputs = self.forward(x)
        
        student_features = outputs['student_features']
        teacher_features = outputs['teacher_features']
        autoencoder_features = outputs['autoencoder_features']
        
        # Compute student-teacher difference map
        st_diff = torch.pow(student_features - teacher_features, 2)
        st_map = torch.mean(st_diff, dim=1, keepdim=True)  # Average across channels
        
        # Compute student-autoencoder difference map  
        sa_diff = torch.pow(student_features - autoencoder_features, 2)
        sa_map = torch.mean(sa_diff, dim=1, keepdim=True)  # Average across channels
        
        # Combine maps (weighted sum)
        anomaly_map = 0.5 * st_map + 0.5 * sa_map
        
        # Resize to input size if specified
        if map_size is not None:
            anomaly_map = F.interpolate(
                anomaly_map, 
                size=map_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        return anomaly_map


def create_efficient_ad_model(model_size: str = 'S', **kwargs) -> EfficientADModel:
    """Factory function to create EfficientAD model"""
    return EfficientADModel(model_size=model_size, **kwargs)