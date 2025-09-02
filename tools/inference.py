import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torchvision.transforms as transforms

from src.models.lightning_model import EfficientADLightning
from src.data.synthetic_generator import generate_anomalous_image


def load_model(model_path: str) -> EfficientADLightning:
    """Load trained model from checkpoint"""
    model = EfficientADLightning.load_from_checkpoint(model_path)
    model.eval()
    return model


def load_and_preprocess_image(image_path: str, image_size: tuple) -> torch.Tensor:
    """Load and preprocess image"""
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image)


def generate_synthetic_test_image(image_size: tuple) -> tuple:
    """Generate synthetic test image with ground truth"""
    image, mask = generate_anomalous_image(image_size)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image)
    mask_array = np.array(mask) / 255.0
    
    return image_tensor, image, mask_array


def visualize_results(original_image, anomaly_map, output_path: str, 
                     ground_truth_mask=None):
    """Visualize anomaly detection results"""
    fig, axes = plt.subplots(1, 3 if ground_truth_mask is not None else 2, 
                            figsize=(15, 5))
    
    # Original image
    if isinstance(original_image, torch.Tensor):
        # Denormalize if tensor
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_np = original_image.permute(1, 2, 0).numpy()
        original_np = original_np * std + mean
        original_np = np.clip(original_np, 0, 1)
        axes[0].imshow(original_np)
    else:
        axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Anomaly heatmap
    axes[1].imshow(anomaly_map, cmap='jet', alpha=0.7)
    axes[1].set_title('Anomaly Heatmap')
    axes[1].axis('off')
    
    # Ground truth (if available)
    if ground_truth_mask is not None:
        axes[2].imshow(ground_truth_mask, cmap='gray')
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run inference with EfficientAD model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to input image (optional for synthetic)')
    parser.add_argument('--is_synthetic', type=bool, default=False,
                       help='Generate synthetic test image')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save output visualization')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                       help='Image size (height, width)')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    image_size = tuple(args.image_size)
    
    if args.is_synthetic:
        # Generate synthetic test image
        print("Generating synthetic test image...")
        image_tensor, original_pil, ground_truth = generate_synthetic_test_image(image_size)
        image_tensor = image_tensor.to(device)
        
        # Run inference
        print("Running inference...")
        anomaly_map, anomaly_score = model.predict_anomaly(image_tensor)
        
        # Convert to numpy for visualization
        anomaly_map_np = anomaly_map.squeeze().cpu().numpy()
        
        # Visualize results
        visualize_results(original_pil, anomaly_map_np, args.output_path, ground_truth)
        
        print(f"Anomaly score: {anomaly_score:.4f}")
        
    else:
        # Load real image
        if not args.image_path:
            raise ValueError("--image_path is required when --is_synthetic=False")
        
        if not os.path.exists(args.image_path):
            raise ValueError(f"Image not found: {args.image_path}")
        
        print(f"Loading image from: {args.image_path}")
        image_tensor = load_and_preprocess_image(args.image_path, image_size)
        image_tensor = image_tensor.to(device)
        
        # Load original for visualization
        original_pil = Image.open(args.image_path).convert('RGB')
        
        # Run inference
        print("Running inference...")
        anomaly_map, anomaly_score = model.predict_anomaly(image_tensor)
        
        # Convert to numpy for visualization
        anomaly_map_np = anomaly_map.squeeze().cpu().numpy()
        
        # Visualize results
        visualize_results(original_pil, anomaly_map_np, args.output_path)
        
        print(f"Anomaly score: {anomaly_score:.4f}")
    
    print("Inference completed!")


if __name__ == '__main__':
    main()