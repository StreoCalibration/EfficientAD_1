import os
from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

class DatasetProvider(ABC):
    """Abstract base class for dataset providers."""

    def __init__(self, train_batch_size: int, eval_batch_size: int):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

    @abstractmethod
    def get_train_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def get_val_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def get_test_dataloader(self) -> DataLoader:
        pass

class SyntheticDatasetProvider(DatasetProvider):
    """Provides synthetic datasets for training and evaluation."""

    def __init__(self, image_size: Tuple[int, int], num_samples: int,
                 train_batch_size: int, eval_batch_size: int):
        super().__init__(train_batch_size, eval_batch_size)
        self.image_size = image_size
        self.num_samples = num_samples
        self._create_datasets()

    def _create_datasets(self):
        # Create dummy data
        images = torch.randn(self.num_samples, 3, *self.image_size)
        labels = torch.zeros(self.num_samples, dtype=torch.long)
        masks = torch.zeros(self.num_samples, 1, *self.image_size)

        # Split data
        train_size = int(0.7 * self.num_samples)
        val_size = int(0.15 * self.num_samples)
        test_size = self.num_samples - train_size - val_size
        
        dataset = TensorDataset(images, labels, masks)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def get_val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def get_test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)


class RealDatasetProvider(DatasetProvider):
    """Provides real datasets from a folder path."""

    def __init__(self, data_path: str, category: str, image_size: Tuple[int, int],
                 train_batch_size: int, eval_batch_size: int):
        super().__init__(train_batch_size, eval_batch_size)
        self.data_path = data_path
        self.category = category
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        self._create_datasets()

    def _create_datasets(self):
        train_path = os.path.join(self.data_path, self.category, 'train')
        test_path = os.path.join(self.data_path, self.category, 'test')
        
        self.train_dataset = ImageFolder(train_path, transform=self.transform)
        
        test_dataset = MVTecADDataset(test_path, transform=self.transform)
        
        # Split test dataset for validation
        val_size = int(0.5 * len(test_dataset))
        test_size = len(test_dataset) - val_size
        self.val_dataset, self.test_dataset = random_split(
            test_dataset, [val_size, test_size]
        )

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def get_val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def get_test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)

class MVTecADDataset(Dataset):
    """Custom dataset for MVTec AD, which has a specific structure."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # good images are label 0
        good_path = os.path.join(root_dir, 'good')
        if os.path.exists(good_path):
            good_images = sorted(os.listdir(good_path))
            for img in good_images:
                self.image_paths.append(os.path.join(good_path, img))
                self.labels.append(0)

        # defected images are label 1
        defect_types = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d != 'good']
        for defect in defect_types:
            defect_images_path = os.path.join(root_dir, defect)
            defect_images = sorted(os.listdir(defect_images_path))
            for img in defect_images:
                self.image_paths.append(os.path.join(defect_images_path, img))
                self.labels.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
