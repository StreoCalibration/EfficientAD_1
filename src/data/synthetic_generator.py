import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw
import random
from typing import Tuple, Optional


def generate_normal_image(image_size: Tuple[int, int]) -> Image.Image:
    """Generate a normal image (gray rectangle)"""
    width, height = image_size
    
    # Create base gray image with slight variations
    base_color = random.randint(100, 150)
    image = Image.new('RGB', (width, height), (base_color, base_color, base_color))
    
    # Add subtle texture variations
    pixels = np.array(image)
    noise = np.random.normal(0, 5, pixels.shape).astype(np.int16)
    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(pixels)


def generate_anomalous_image(image_size: Tuple[int, int]) -> Tuple[Image.Image, Image.Image]:
    """Generate an anomalous image with ground truth mask"""
    width, height = image_size
    
    # Start with normal image
    image = generate_normal_image(image_size)
    draw = ImageDraw.Draw(image)
    
    # Create ground truth mask
    mask = Image.new('L', (width, height), 0)
    mask_draw = ImageDraw.Draw(mask)
    
    # Randomly choose anomaly type
    anomaly_type = random.choice(['scratch', 'contamination', 'misplaced_object'])
    
    if anomaly_type == 'scratch':
        # Draw random scratches (lines)
        num_scratches = random.randint(1, 3)
        for _ in range(num_scratches):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            line_width = random.randint(2, 5)
            color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
            
            draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
            mask_draw.line([(x1, y1), (x2, y2)], fill=255, width=line_width + 2)
    
    elif anomaly_type == 'contamination':
        # Draw random spots/blobs
        num_spots = random.randint(1, 5)
        for _ in range(num_spots):
            x, y = random.randint(10, width-10), random.randint(10, height-10)
            radius = random.randint(5, 15)
            color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
            
            bbox = [x-radius, y-radius, x+radius, y+radius]
            draw.ellipse(bbox, fill=color)
            mask_draw.ellipse(bbox, fill=255)
    
    elif anomaly_type == 'misplaced_object':
        # Draw geometric shapes
        num_objects = random.randint(1, 2)
        for _ in range(num_objects):
            x, y = random.randint(20, width-20), random.randint(20, height-20)
            size = random.randint(10, 25)
            color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
            
            shape_type = random.choice(['rectangle', 'circle'])
            if shape_type == 'rectangle':
                bbox = [x-size//2, y-size//2, x+size//2, y+size//2]
                draw.rectangle(bbox, fill=color)
                mask_draw.rectangle(bbox, fill=255)
            else:
                bbox = [x-size//2, y-size//2, x+size//2, y+size//2]
                draw.ellipse(bbox, fill=color)
                mask_draw.ellipse(bbox, fill=255)
    
    return image, mask


class SyntheticTrainDataset(Dataset):
    """Synthetic training dataset (normal images only)"""
    
    def __init__(self, image_size: Tuple[int, int], num_samples: int, transform=None):
        self.image_size = image_size
        self.num_samples = num_samples
        self.transform = transform
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = generate_normal_image(self.image_size)
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # 0 for normal


class SyntheticTestDataset(Dataset):
    """Synthetic test dataset (normal + anomalous images)"""
    
    def __init__(self, image_size: Tuple[int, int], num_samples: int, transform=None):
        self.image_size = image_size
        self.num_samples = num_samples
        self.transform = transform
        
        # Generate half normal, half anomalous
        self.normal_count = num_samples // 2
        self.anomalous_count = num_samples - self.normal_count
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if idx < self.normal_count:
            # Normal image
            image = generate_normal_image(self.image_size)
            mask = Image.new('L', self.image_size, 0)  # All zeros mask
            label = 0
        else:
            # Anomalous image
            image, mask = generate_anomalous_image(self.image_size)
            label = 1
        
        if self.transform:
            image = self.transform(image)
            # Also transform mask to tensor
            mask = torch.from_numpy(np.array(mask)).float() / 255.0
        
        return image, label, mask