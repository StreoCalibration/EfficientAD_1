#!/usr/bin/env python3
"""Simple test script for synthetic data generation and model architecture"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.data.synthetic_generator import generate_normal_image, generate_anomalous_image
from src.models.torch_model import EfficientADModel

def test_synthetic_data_generation():
    """Test synthetic data generation"""
    print("Testing synthetic data generation...")
    
    image_size = (256, 256)
    
    # Generate normal image
    normal_img = generate_normal_image(image_size)
    
    # Generate anomalous image
    anomalous_img, mask = generate_anomalous_image(image_size)
    
    # Save test images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(normal_img)
    axes[0].set_title('Normal Image')
    axes[0].axis('off')
    
    axes[1].imshow(anomalous_img)
    axes[1].set_title('Anomalous Image')
    axes[1].axis('off')
    
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('Ground Truth Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_synthetic_generation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("OK Synthetic data generation test completed")
    print("OK Test images saved as 'test_synthetic_generation.png'")

def test_model_architecture():
    """Test model architecture forward pass"""
    print("\nTesting model architecture...")
    
    # Create model
    model = EfficientADModel(model_size='S')
    model.eval()
    
    # Create dummy input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 256, 256)
    
    # Test forward pass
    with torch.no_grad():
        outputs = model(input_tensor)
    
    print(f"OK Model forward pass successful")
    print(f"OK Input shape: {input_tensor.shape}")
    print(f"OK Student features shape: {outputs['student_features'].shape}")
    print(f"OK Teacher features shape: {outputs['teacher_features'].shape}")
    print(f"OK Autoencoder features shape: {outputs['autoencoder_features'].shape}")
    
    # Test anomaly map computation
    with torch.no_grad():
        anomaly_map = model.compute_anomaly_map(input_tensor, map_size=(256, 256))
    
    print(f"OK Anomaly map shape: {anomaly_map.shape}")
    print(f"OK Model architecture test completed")

def test_dataset_provider():
    """Test dataset provider"""
    print("\nTesting dataset provider...")
    
    from src.data.provider import SyntheticDatasetProvider
    
    # Create synthetic dataset provider
    provider = SyntheticDatasetProvider(
        image_size=(256, 256),
        num_samples=10,
        train_batch_size=2,
        eval_batch_size=4
    )
    
    # Test train dataloader
    train_loader = provider.get_train_dataloader()
    train_batch = next(iter(train_loader))
    
    images, labels = train_batch
    print(f"OK Train batch - Images shape: {images.shape}, Labels shape: {labels.shape}")
    
    # Test validation dataloader
    val_loader = provider.get_val_dataloader()
    val_batch = next(iter(val_loader))
    
    if len(val_batch) == 3:
        images, labels, masks = val_batch
        print(f"OK Val batch - Images: {images.shape}, Labels: {labels.shape}, Masks: {masks.shape}")
    else:
        images, labels = val_batch
        print(f"OK Val batch - Images: {images.shape}, Labels: {labels.shape}")
    
    print("OK Dataset provider test completed")

def main():
    print("=== EfficientAD Basic Tests ===\n")
    
    try:
        test_synthetic_data_generation()
        test_model_architecture()
        test_dataset_provider()
        
        print("\n=== All Tests Passed! ===")
        print("Your EfficientAD implementation is ready to use.")
        print("\nNext steps:")
        print("1. Run training: python tools/train.py --config configs/config.yaml")
        print("2. After training, run inference: python tools/inference.py --model_path [checkpoint] --is_synthetic True --output_path results/output.png")
        
    except Exception as e:
        print(f"\nERROR Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()