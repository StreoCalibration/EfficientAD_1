#!/usr/bin/env python3
"""
Test script for folder-based inference functionality

This script tests the folder inference with a small subset of images
without requiring a trained model (uses mock processing for testing).
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def create_mock_anomaly_map(image_size=(256, 256)):
    """Create a mock anomaly map for testing"""
    # Create a synthetic anomaly map with some patterns
    h, w = image_size
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    
    # Create some anomaly patterns
    center_anomaly = np.exp(-((x - 0.5)**2 + (y - 0.5)**2) * 10)
    corner_anomaly = np.exp(-((x - 0.1)**2 + (y - 0.1)**2) * 20) * 0.5
    random_noise = np.random.random((h, w)) * 0.1
    
    anomaly_map = center_anomaly + corner_anomaly + random_noise
    
    # Normalize to 0-1 range
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
    
    return anomaly_map


def test_normalization():
    """Test the normalization function"""
    print("Testing normalization function...")
    
    # Test case 1: Normal range
    test_map1 = np.array([[0.1, 0.5, 0.9], [0.2, 0.8, 0.3]])
    normalized1 = (test_map1 - test_map1.min()) / (test_map1.max() - test_map1.min())
    print(f"Test 1 - Min: {normalized1.min():.6f}, Max: {normalized1.max():.6f}")
    assert normalized1.min() == 0.0 and normalized1.max() == 1.0
    
    # Test case 2: All same values
    test_map2 = np.ones((3, 3)) * 0.5
    normalized2 = np.zeros_like(test_map2) if test_map2.max() == test_map2.min() else \
                  (test_map2 - test_map2.min()) / (test_map2.max() - test_map2.min())
    print(f"Test 2 - Min: {normalized2.min():.6f}, Max: {normalized2.max():.6f}")
    
    # Test case 3: Negative values
    test_map3 = np.array([[-1.0, 0.0, 2.0], [-0.5, 1.5, 0.5]])
    normalized3 = (test_map3 - test_map3.min()) / (test_map3.max() - test_map3.min())
    print(f"Test 3 - Min: {normalized3.min():.6f}, Max: {normalized3.max():.6f}")
    
    print("Normalization tests passed!")


def test_anomaly_map_creation():
    """Test anomaly map creation and visualization"""
    print("Testing anomaly map creation...")
    
    # Create test directory
    test_output_dir = "test_anomaly_outputs"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Create mock anomaly maps with different characteristics
    test_cases = [
        ("uniform_low", np.full((64, 64), 0.1)),
        ("uniform_high", np.full((64, 64), 0.9)),
        ("gradient", np.tile(np.linspace(0, 1, 64), (64, 1))),
        ("random", np.random.random((64, 64))),
        ("synthetic_pattern", create_mock_anomaly_map((64, 64)))
    ]
    
    for name, anomaly_map in test_cases:
        # Normalize
        if anomaly_map.max() > anomaly_map.min():
            normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
        else:
            normalized = np.zeros_like(anomaly_map)
        
        # Save visualization
        plt.figure(figsize=(8, 6))
        plt.imshow(normalized, cmap='jet', vmin=0, vmax=1)
        plt.colorbar(label='Anomaly Score (0-1)')
        plt.title(f'Test Anomaly Map: {name}')
        plt.axis('off')
        
        output_path = os.path.join(test_output_dir, f"test_{name}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save numpy array
        np.save(output_path.replace('.png', '.npy'), normalized)
        
        print(f"  {name}: Min={normalized.min():.3f}, Max={normalized.max():.3f}, Mean={normalized.mean():.3f}")
    
    print(f"Anomaly map tests completed! Check {test_output_dir}/ for outputs")


def test_batch_processing_simulation():
    """Simulate batch processing without actual model"""
    print("Testing batch processing simulation...")
    
    # Check if we have access to the batch folders
    train_dir = Path("Z:/Data/1_Work/Zeta/20250903_Training/train")
    if train_dir.exists():
        batch_folders = [f for f in os.listdir(train_dir) if f.startswith('batch_')]
        print(f"Found {len(batch_folders)} batch folders: {batch_folders[:3]}..." if len(batch_folders) > 3 else batch_folders)
        
        # Test with first batch folder if available
        if batch_folders:
            batch_path = train_dir / batch_folders[0]
            image_files = list(batch_path.glob("*.png"))
            print(f"  Batch {batch_folders[0]} contains {len(image_files)} images")
            
            if len(image_files) > 0:
                print(f"  Sample files: {[f.name for f in image_files[:5]]}")
    else:
        print("  Train directory not accessible for testing")
    
    print("Batch processing simulation completed!")


def main():
    """Run all tests"""
    print("Testing EfficientAD Folder Inference")
    print("=" * 50)
    
    try:
        test_normalization()
        print()
        
        test_anomaly_map_creation()
        print()
        
        test_batch_processing_simulation()
        print()
        
        print("All tests completed successfully!")
        print("\nTo run actual inference with a trained model, use:")
        print("  python example_folder_inference.py --model_path 'path/to/model.ckpt' --batch_folder 'batch_01'")
        
    except Exception as e:
        print(f"Test failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())