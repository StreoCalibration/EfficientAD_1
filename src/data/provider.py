import os
from typing import Optional, Tuple, List
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import pytorch_lightning as pl
from glob import glob


class PadAndResizeToSize:
    """
    사용자 정의 변환: 이미지를 목표 크기로 패딩합니다.
    이미지가 목표 크기보다 크면, 비율을 유지하면서 축소됩니다.
    그 후, 목표 크기에 맞게 패딩이 추가됩니다.
    """
    def __init__(self, size: Tuple[int, int], fill=0, padding_mode='constant'):
        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img: Image.Image) -> Image.Image:
        # PIL의 thumbnail 함수는 이미지 비율을 유지하면서 주어진 크기 내에 맞게 이미지를 축소합니다.
        # 이 함수는 이미지를 직접 수정합니다.
        img.thumbnail(self.size, Image.Resampling.LANCZOS)
        
        w, h = img.size
        target_w, target_h = self.size
        
        # 패딩 계산
        pad_left = (target_w - w) // 2
        pad_top = (target_h - h) // 2
        
        # 패딩 튜플: (왼쪽, 위, 오른쪽, 아래)
        padding = (pad_left, pad_top, target_w - w - pad_left, target_h - h - pad_top)
        
        return TF.pad(img, padding, self.fill, self.padding_mode)


class RealDataset(Dataset):
    """실제 이미지 로딩을 위한 데이터셋, 다중 채널 학습 지원"""

    def __init__(self, data_path: str, category: str, image_size: Tuple[int, int],
                 split: str = 'train', is_validation: bool = False):
        self.data_path = data_path
        self.category = category
        self.image_size = image_size
        self.split = split
        self.is_validation = is_validation

        self.image_files = self._get_image_files()

        # 훈련용 변환: 1채널 흑백 이미지용
        self.train_transform = transforms.Compose([
            PadAndResizeToSize(self.image_size),  # Resize 대신 패딩 변환 사용
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # 테스트용 변환: 3채널 RGB 이미지용
        self.test_transform = transforms.Compose([
            PadAndResizeToSize(self.image_size),  # Resize 대신 패딩 변환 사용
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_image_files(self) -> List[str]:
        if self.split == 'train':
            base_path = os.path.join(self.data_path, self.category, 'train', 'good', '*.png')
            image_files = sorted(glob(base_path))
            
            # 9개 단위로 이미지를 처리하므로, 파일 수가 9의 배수가 되도록 조정
            num_groups = len(image_files) // 9
            if num_groups == 0:
                print(f"경고: '{base_path}' 경로에 9개 그룹을 만들기에 충분한 이미지가 없습니다. 발견된 이미지 수: {len(image_files)}")
                return []

            image_files = image_files[:num_groups * 9]

            # 훈련/검증 그룹 분할 (80% 훈련, 20% 검증)
            num_train_groups = int(num_groups * 0.8)
            
            if self.is_validation:
                # 마지막 20%를 검증용으로 사용
                return image_files[num_train_groups * 9:]
            else:
                # 처음 80%를 훈련용으로 사용
                return image_files[:num_train_groups * 9]
        else: # test
            image_files = sorted(glob(os.path.join(self.data_path, self.category, 'test', '*', '*.png')))
        return image_files

    def __len__(self):
        if self.split == 'train':
            # 9개의 이미지가 하나의 샘플
            return len(self.image_files) // 9
        else:
            return len(self.image_files)

    def __getitem__(self, idx):
        if self.split == 'train':
            start_idx = idx * 9
            image_paths = self.image_files[start_idx : start_idx + 9]
            
            # 각 이미지를 흑백으로 변환하고 변환 적용
            images = [self.train_transform(Image.open(p).convert('L')) for p in image_paths]
            
            # 9개의 1채널 이미지를 9채널 텐서로 결합 (dim=0)
            stacked_image = torch.cat(images, dim=0)
            # 훈련/검증 데이터는 모두 정상(good)이므로 라벨 0을 반환
            return stacked_image, 0
        else:
            image_path = self.image_files[idx]
            image = Image.open(image_path).convert('RGB')
            image = self.test_transform(image)

            label_str = os.path.basename(os.path.dirname(image_path))
            label = 0 if label_str == 'good' else 1
            return image, label


class RealDatasetProvider(pl.LightningDataModule):
    def __init__(self, data_path: str, category: str, image_size: Tuple[int, int],
                 train_batch_size: int, eval_batch_size: int):
        super().__init__()
        self.data_path = data_path
        self.category = category
        self.image_size = image_size
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = RealDataset(self.data_path, self.category, self.image_size, split='train')
            self.val_dataset = RealDataset(self.data_path, self.category, self.image_size, split='train', is_validation=True)
        if stage == 'test' or stage is None:
            self.test_dataset = RealDataset(self.data_path, self.category, self.image_size, split='test')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)


class SyntheticDatasetProvider(pl.LightningDataModule):
    """합성 데이터셋을 제공합니다."""

    def __init__(self, image_size: Tuple[int, int], num_samples: int,
                 train_batch_size: int, eval_batch_size: int):
        super().__init__()
        self.image_size = image_size
        self.num_samples = num_samples
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        # 더미 데이터 생성
        images = torch.randn(self.num_samples, 3, *self.image_size)
        
        train_size = int(0.7 * self.num_samples)
        val_test_size = self.num_samples - train_size
        
        # 합성 데이터의 경우, 훈련 로더는 이미지만 반환
        train_images = images[:train_size]
        
        # 검증 및 테스트 로더는 이미지와 라벨 반환
        val_test_images = images[train_size:]
        labels = torch.zeros(len(val_test_images), dtype=torch.long)
        
        self.train_dataset = TensorDataset(train_images)
        
        val_test_dataset = TensorDataset(val_test_images, labels)

        # 검증/테스트 분할
        val_size = val_test_size // 2
        test_size = val_test_size - val_size
        
        self.val_dataset, self.test_dataset = random_split(
            val_test_dataset, [val_size, test_size]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)
