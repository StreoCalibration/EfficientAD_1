import pytorch_lightning as pl
import torch
from typing import Optional, Tuple
from sklearn.metrics import roc_auc_score

from src.models.torch_model import EfficientAD

class EfficientADLightning(pl.LightningModule):
    """EfficientAD 모델을 위한 PyTorch Lightning 래퍼"""
    def __init__(self,
                 model_size: str = 's',
                 learning_rate: float = 1e-4,
                 st_weight: float = 1.0,
                 ae_weight: float = 1.0,
                 penalty_weight: float = 1.0, # 현재 구현에서는 사용되지 않음
                 dataset_provider: Optional[pl.LightningDataModule] = None,
                 image_size: Tuple[int, int] = (256, 256),
                 in_channels: int = 3):
        super().__init__()
        # 하이퍼파라미터를 저장합니다. dataset_provider는 저장하지 않습니다.
        self.save_hyperparameters(ignore=['dataset_provider'])
        
        # EfficientAD 모델 초기화
        self.model = EfficientAD(model_size, in_channels=in_channels)
        
        # 손실 함수 정의
        self.loss_st = torch.nn.MSELoss()
        self.loss_ae = torch.nn.MSELoss()
        
        self.dataset_provider = dataset_provider
        self.image_size = image_size

        # 검증 및 테스트 결과를 저장하기 위한 리스트
        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # 배치에서 이미지만 추출
        images = batch[0] if isinstance(batch, list) and len(batch) > 0 else batch

        # 모델 forward pass
        teacher_output, student_output, ae_output = self.model(images)
        
        # 손실 계산
        st_loss = self.loss_st(student_output, teacher_output)
        ae_loss = self.loss_ae(ae_output, images)
        
        total_loss = self.hparams.st_weight * st_loss + self.hparams.ae_weight * ae_loss
        
        # 손실 로깅
        self.log('train/total_loss', total_loss, prog_bar=True)
        self.log('train/st_loss', st_loss)
        self.log('train/ae_loss', ae_loss)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        
        teacher_output, student_output, ae_output = self.model(images)
        
        st_loss = self.loss_st(student_output, teacher_output)
        ae_loss = self.loss_ae(ae_output, images)
        
        total_loss = self.hparams.st_weight * st_loss + self.hparams.ae_weight * ae_loss
        self.log('val/total_loss', total_loss, prog_bar=True)

        # 이상치 점수 계산
        anomaly_map = torch.mean(torch.pow(student_output - teacher_output, 2), dim=1, keepdim=True)
        score = torch.max(anomaly_map.view(anomaly_map.size(0), -1), dim=1)[0]
        
        self.validation_outputs.append({'labels': labels.cpu(), 'scores': score.cpu()})
        return {'val_loss': total_loss}

    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return
            
        labels = torch.cat([x['labels'] for x in self.validation_outputs])
        scores = torch.cat([x['scores'] for x in self.validation_outputs])
        
        # AUROC 계산 (라벨이 두 종류 이상일 때)
        if len(torch.unique(labels)) > 1:
            auroc = roc_auc_score(labels.numpy(), scores.numpy())
            self.log('val/auroc', auroc, prog_bar=True)
        
        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        images, labels = batch
        teacher_output, student_output, _ = self.model(images)
        anomaly_map = torch.mean(torch.pow(student_output - teacher_output, 2), dim=1, keepdim=True)
        score = torch.max(anomaly_map.view(anomaly_map.size(0), -1), dim=1)[0]
        self.test_outputs.append({'labels': labels.cpu(), 'scores': score.cpu()})

    def on_test_epoch_end(self):
        if not self.test_outputs:
            return
            
        labels = torch.cat([x['labels'] for x in self.test_outputs])
        scores = torch.cat([x['scores'] for x in self.test_outputs])
        
        if len(torch.unique(labels)) > 1:
            auroc = roc_auc_score(labels.numpy(), scores.numpy())
            self.log('test/auroc', auroc, prog_bar=True)
        
        self.test_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        return optimizer