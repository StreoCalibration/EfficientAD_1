import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
import numpy as np
from typing import Dict, Any, Optional, Tuple

from src.models.torch_model import EfficientADModel
from src.utils.loss import EfficientADLoss
from src.data.provider import DatasetProvider


class EfficientADLightning(pl.LightningModule):
    """PyTorch Lightning wrapper for EfficientAD model"""
    
    def __init__(self,
                 model_size: str = 'S',
                 learning_rate: float = 1e-4,
                 st_weight: float = 1.0,
                 ae_weight: float = 1.0,
                 penalty_weight: float = 1.0,
                 dataset_provider: Optional[DatasetProvider] = None,
                 image_size: Tuple[int, int] = (256, 256),
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['dataset_provider'])
        
        # Initialize model
        self.model = EfficientADModel(model_size=model_size)
        
        # Initialize loss
        self.loss_fn = EfficientADLoss(
            st_weight=st_weight,
            ae_weight=ae_weight, 
            penalty_weight=penalty_weight
        )
        
        # Store dataset provider
        self.dataset_provider = dataset_provider
        self.image_size = image_size
        
        # ImageNette data for pretraining penalty
        self.imagenet_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # For validation metrics
        self.validation_outputs = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step with synthetic ImageNette penalty"""
        images, labels = batch
        
        # Forward pass through model
        outputs = self.model(images)
        
        # Generate synthetic ImageNette batch for penalty
        # In practice, you would load actual ImageNette data
        imagenet_batch = self._generate_imagenet_batch(images.shape[0])
        
        # Compute losses
        loss_dict = self.loss_fn(outputs, imagenet_batch, self.model.student)
        
        # Log losses
        self.log('train/total_loss', loss_dict['total_loss'], prog_bar=True)
        self.log('train/st_loss', loss_dict['st_loss'])
        self.log('train/ae_loss', loss_dict['ae_loss'])
        self.log('train/penalty_loss', loss_dict['penalty_loss'])
        
        return loss_dict['total_loss']
    
    def validation_step(self, batch, batch_idx):
        """Validation step with anomaly detection evaluation"""
        if len(batch) == 3:
            # Synthetic data with masks
            images, labels, masks = batch
        else:
            # Real data without masks
            images, labels = batch
            masks = None
        
        # Compute anomaly maps
        with torch.no_grad():
            anomaly_maps = self.model.compute_anomaly_map(images, self.image_size)
        
        # Store outputs for epoch-end metrics
        self.validation_outputs.append({
            'anomaly_maps': anomaly_maps.cpu(),
            'labels': labels.cpu(),
            'masks': masks.cpu() if masks is not None else None
        })
        
        return {'anomaly_maps': anomaly_maps, 'labels': labels}
    
    def on_validation_epoch_end(self):
        """Compute validation metrics at epoch end"""
        if not self.validation_outputs:
            return
        
        # Concatenate all outputs
        all_maps = torch.cat([x['anomaly_maps'] for x in self.validation_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_outputs])
        
        # Compute image-level AUROC
        image_scores = torch.max(all_maps.view(all_maps.size(0), -1), dim=1)[0]
        
        if len(torch.unique(all_labels)) > 1:  # Need both classes for AUROC
            auroc = roc_auc_score(all_labels.numpy(), image_scores.numpy())
            self.log('val/auroc', auroc, prog_bar=True)
        
        # Compute pixel-level metrics if masks available
        all_masks = [x['masks'] for x in self.validation_outputs if x['masks'] is not None]
        if all_masks:
            all_masks = torch.cat(all_masks)
            # Resize anomaly maps to mask size if needed
            if all_maps.shape[-2:] != all_masks.shape[-2:]:
                all_maps = F.interpolate(
                    all_maps, 
                    size=all_masks.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Flatten for pixel-level evaluation
            pixel_scores = all_maps.view(-1).numpy()
            pixel_labels = all_masks.view(-1).numpy()
            
            if len(np.unique(pixel_labels)) > 1:
                pixel_auroc = roc_auc_score(pixel_labels, pixel_scores)
                self.log('val/pixel_auroc', pixel_auroc)
        
        # Clear outputs
        self.validation_outputs.clear()
    
    def configure_optimizers(self):
        """Configure Adam optimizer"""
        optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.hparams.learning_rate
        )
        
        # Optional: Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val/auroc'
        }
    
    def _generate_imagenet_batch(self, batch_size: int) -> torch.Tensor:
        """Generate synthetic ImageNette batch for pretraining penalty"""
        # For now, generate random images
        # In practice, load actual ImageNette data
        device = next(self.model.parameters()).device
        synthetic_batch = torch.randn(
            batch_size, 3, *self.image_size, 
            device=device
        )
        
        # Apply normalization to match ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        synthetic_batch = synthetic_batch * std + mean
        
        return synthetic_batch
    
    def predict_anomaly(self, image: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Predict anomaly for a single image"""
        self.eval()
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)  # Add batch dimension
            
            anomaly_map = self.model.compute_anomaly_map(image, self.image_size)
            anomaly_score = torch.max(anomaly_map).item()
            
            return anomaly_map.squeeze(0), anomaly_score