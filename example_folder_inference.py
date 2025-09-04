#!/usr/bin/env python3
"""
Example script for running folder-based inference with EfficientAD

This script demonstrates how to process an entire folder of images
and get normalized anomaly maps (0-1 range) for each pixel.

Usage examples:
1. Process a batch folder:
   python example_folder_inference.py --model_path "path/to/model.ckpt" --batch_folder "batch_01"

2. Process a custom folder:
   python example_folder_inference.py --model_path "path/to/model.ckpt" --custom_folder "/path/to/images"
"""

import argparse
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tools.folder_inference import process_folder


def main():
    parser = argparse.ArgumentParser(description='Example folder-based inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    
    # Folder selection options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--batch_folder', type=str,
                       help='Batch folder name (e.g., "batch_01") - will look in train directory')
    group.add_argument('--custom_folder', type=str,
                       help='Custom folder path')
    
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: auto-generated)')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                       help='Image size (height, width)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Processing batch size')
    
    args = parser.parse_args()
    
    # Determine input folder
    if args.batch_folder:
        # Use batch folder from train directory
        train_dir = Path("Z:/Data/1_Work/Zeta/20250903_Training/train")
        folder_path = train_dir / args.batch_folder
        folder_name = args.batch_folder
    else:
        # Use custom folder
        folder_path = Path(args.custom_folder)
        folder_name = folder_path.name
    
    # Validate folder exists
    if not folder_path.exists():
        print(f"Error: Folder does not exist: {folder_path}")
        return
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Auto-generate output directory name
        output_dir = f"inference_results_{folder_name}"
    
    print(f"Processing folder: {folder_path}")
    print(f"Output directory: {output_dir}")
    
    # Run inference
    process_folder(
        model_path=args.model_path,
        folder_path=str(folder_path),
        output_dir=output_dir,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        save_as_image=True,
        save_as_npy=True
    )
    
    print(f"\n✅ Inference completed!")
    print(f"Results saved in: {output_dir}")
    print(f"\nOutput files:")
    print(f"  - PNG files: Visualization images with color-mapped anomaly scores")
    print(f"  - NPY files: Raw normalized anomaly maps (0-1 range) for each pixel")
    print(f"  - results_summary.txt: Detailed results for all processed images")


if __name__ == '__main__':
    main()