import argparse
import os
import sys
from pathlib import Path
import glob
from tqdm import tqdm

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


def load_model(model_path: str) -> EfficientADLightning:
    """Load trained model from checkpoint"""
    model = EfficientADLightning.load_from_checkpoint(model_path)
    model.eval()
    return model


def load_and_preprocess_image(image_path: str, image_size: tuple) -> torch.Tensor:
    """Load and preprocess image"""
    try:
        image = Image.open(image_path).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def normalize_anomaly_map(anomaly_map: torch.Tensor) -> np.ndarray:
    """Normalize anomaly map to 0-1 range"""
    # Convert to numpy
    if isinstance(anomaly_map, torch.Tensor):
        anomaly_map_np = anomaly_map.squeeze().cpu().numpy()
    else:
        anomaly_map_np = anomaly_map
    
    # Normalize to 0-1 range
    min_val = anomaly_map_np.min()
    max_val = anomaly_map_np.max()
    
    if max_val > min_val:
        normalized_map = (anomaly_map_np - min_val) / (max_val - min_val)
    else:
        # If all values are the same, set to 0
        normalized_map = np.zeros_like(anomaly_map_np)
    
    return normalized_map


def save_anomaly_map(anomaly_map: np.ndarray, output_path: str, 
                     save_as_image: bool = True, save_as_npy: bool = True):
    """Save normalized anomaly map as image and/or numpy array"""
    
    if save_as_npy:
        # Save as numpy array for exact values
        npy_path = output_path.replace('.png', '.npy')
        np.save(npy_path, anomaly_map)
    
    if save_as_image:
        # Save as image for visualization
        # Create colormap visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(anomaly_map, cmap='jet', vmin=0, vmax=1)
        plt.colorbar(label='Anomaly Score (0-1)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def process_folder(model_path: str, folder_path: str, output_dir: str, 
                   image_size: tuple, batch_size: int = 8,
                   save_as_image: bool = True, save_as_npy: bool = True):
    """Process all images in a folder"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Find all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    image_files.sort()  # Sort for consistent ordering
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images in batches
    results_summary = []
    
    for i in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        valid_files = []
        
        # Load batch of images
        for image_path in batch_files:
            image_tensor = load_and_preprocess_image(image_path, image_size)
            if image_tensor is not None:
                batch_images.append(image_tensor)
                valid_files.append(image_path)
        
        if not batch_images:
            continue
        
        # Stack images into batch tensor
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Run inference
        with torch.no_grad():
            # Process each image individually to get proper anomaly maps
            for j, (image_tensor, image_path) in enumerate(zip(batch_tensor, valid_files)):
                image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
                
                # Get anomaly map and score
                anomaly_map, anomaly_score = model.predict_anomaly(image_tensor)
                
                # Normalize to 0-1 range
                normalized_map = normalize_anomaly_map(anomaly_map)
                
                # Create output filename
                image_name = Path(image_path).stem
                output_path = os.path.join(output_dir, f"{image_name}_anomaly.png")
                
                # Save anomaly map
                save_anomaly_map(normalized_map, output_path, save_as_image, save_as_npy)
                
                # Store results
                results_summary.append({
                    'image_path': image_path,
                    'image_name': image_name,
                    'anomaly_score': anomaly_score,
                    'max_pixel_score': normalized_map.max(),
                    'mean_pixel_score': normalized_map.mean(),
                    'output_path': output_path
                })
    
    # Save results summary
    summary_path = os.path.join(output_dir, 'results_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Image Processing Results Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Input folder: {folder_path}\n")
        f.write(f"Output folder: {output_dir}\n")
        f.write(f"Total images processed: {len(results_summary)}\n")
        f.write(f"Image size: {image_size}\n\n")
        
        f.write("Per-image results:\n")
        f.write("-" * 50 + "\n")
        
        for result in results_summary:
            f.write(f"Image: {result['image_name']}\n")
            f.write(f"  Original path: {result['image_path']}\n")
            f.write(f"  Anomaly score: {result['anomaly_score']:.6f}\n")
            f.write(f"  Max pixel score: {result['max_pixel_score']:.6f}\n")
            f.write(f"  Mean pixel score: {result['mean_pixel_score']:.6f}\n")
            f.write(f"  Output: {result['output_path']}\n\n")
    
    print(f"\nProcessing completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Summary saved to: {summary_path}")
    
    # Print statistics
    scores = [r['anomaly_score'] for r in results_summary]
    max_scores = [r['max_pixel_score'] for r in results_summary]
    
    print(f"\nStatistics:")
    print(f"Anomaly scores - Min: {min(scores):.6f}, Max: {max(scores):.6f}, Mean: {np.mean(scores):.6f}")
    print(f"Max pixel scores - Min: {min(max_scores):.6f}, Max: {max(max_scores):.6f}, Mean: {np.mean(max_scores):.6f}")


def main():
    parser = argparse.ArgumentParser(description='Run folder-based inference with EfficientAD model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--folder_path', type=str, required=True,
                       help='Path to folder containing images to process')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save output anomaly maps')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                       help='Image size (height, width)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for processing (adjust based on GPU memory)')
    parser.add_argument('--save_as_image', type=bool, default=True,
                       help='Save anomaly maps as PNG images')
    parser.add_argument('--save_as_npy', type=bool, default=True,
                       help='Save anomaly maps as numpy arrays')
    
    args = parser.parse_args()
    
    # Validate input paths
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model checkpoint not found: {args.model_path}")
    
    if not os.path.exists(args.folder_path):
        raise ValueError(f"Input folder not found: {args.folder_path}")
    
    if not os.path.isdir(args.folder_path):
        raise ValueError(f"Input path is not a directory: {args.folder_path}")
    
    image_size = tuple(args.image_size)
    
    # Process folder
    process_folder(
        model_path=args.model_path,
        folder_path=args.folder_path,
        output_dir=args.output_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        save_as_image=args.save_as_image,
        save_as_npy=args.save_as_npy
    )


if __name__ == '__main__':
    main()