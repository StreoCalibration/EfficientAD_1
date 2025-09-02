# EfficientAD Implementation

EfficientAD 논문의 완벽한 구현체로, 산업용 이상 탐지를 위한 경량화된 딥러닝 모델입니다.

## 주요 특징

- **논문 충실성**: Hard Feature Loss와 Pretraining Penalty 등 핵심 아이디어를 정확히 구현
- **데이터 유연성**: 합성 데이터와 실제 데이터(MVTec AD) 모두 지원
- **배포 용이성**: PyTorch Lightning 기반으로 확장성과 재현성 확보
- **실용성**: AOI 장비 등 실제 산업 환경 적용을 고려한 설계

## 설치 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 프로젝트 구조

```
EfficientAD_1/
├── configs/           # 설정 파일
├── data/             # 데이터셋 저장소
├── src/              # 소스 코드
│   ├── data/         # 데이터 처리
│   ├── models/       # 모델 구현
│   └── utils/        # 유틸리티
├── tools/            # 훈련/추론 스크립트
├── deployment/       # 배포 관련
└── results/          # 결과 저장
```

## 사용 방법

### 1. 합성 데이터로 빠른 프로토타이핑

빠른 테스트와 모델 검증을 위해 합성 데이터를 사용합니다.

```bash
# 합성 데이터로 훈련
python tools/train.py --config configs/config.yaml

# 합성 데이터로 추론 테스트
python tools/inference.py \
  --model_path results/EfficientAD/synthetic_synthetic/version_0/checkpoints/last.ckpt \
  --is_synthetic True \
  --output_path results/synthetic_test_output.png
```

**설정 파일 (configs/config.yaml) 주요 항목:**
```yaml
model:
  model_size: S  # S (Small) 또는 M (Medium)
data:
  source: synthetic
  image_size: [256, 256]
  num_samples: 1000
trainer:
  accelerator: cpu  # 또는 cuda
  max_epochs: 10
```

### 2. 실제 데이터(MVTec AD)로 본격 훈련

#### 데이터 준비
```bash
# MVTec AD 데이터셋 다운로드 및 압축 해제
mkdir -p data/mvtec_ad
# MVTec AD 데이터를 data/mvtec_ad/ 디렉토리에 배치
```

#### 훈련 실행
```bash
# 실제 데이터로 훈련 (bottle 카테고리 예시)
python tools/train.py --config configs/config_real.yaml

# 특정 체크포인트에서 재개
python tools/train.py --config configs/config_real.yaml --resume path/to/checkpoint.ckpt
```

#### 추론 실행
```bash
# 실제 이미지로 추론
python tools/inference.py \
  --model_path results/EfficientAD/bottle_real/version_0/checkpoints/best.ckpt \
  --image_path data/mvtec_ad/bottle/test/broken_large/000.png \
  --output_path results/bottle_inference_output.png

# 이미지 크기 지정
python tools/inference.py \
  --model_path path/to/model.ckpt \
  --image_path path/to/image.jpg \
  --output_path results/output.png \
  --image_size 512 512
```

### 3. 설정 파일 커스터마이징

#### 데이터 설정
```yaml
data:
  source: real  # synthetic 또는 real
  path: ./data/mvtec_ad  # 실제 데이터 경로
  category: bottle  # MVTec AD 카테고리 (bottle, cable, capsule 등)
  image_size: [256, 256]
  train_batch_size: 1  # 논문에서 권장하는 배치 크기
  eval_batch_size: 32
```

#### 모델 설정
```yaml
model:
  name: efficient_ad
  model_size: M  # S: 384 채널, M: 768 채널

# 손실 함수 가중치
st_weight: 1.0      # Student-Teacher 손실
ae_weight: 1.0      # Student-Autoencoder 손실
penalty_weight: 1.0 # 사전훈련 페널티
```

#### 훈련 설정
```yaml
trainer:
  accelerator: cuda  # cpu 또는 cuda
  devices: 1
  max_epochs: 100

learning_rate: 1e-4
```

## 지원하는 데이터셋

### 1. 합성 데이터
- 자동 생성되는 정상/비정상 이미지
- 스크래치, 오염, 잘못된 객체 등 다양한 이상 패턴
- 빠른 프로토타이핑과 알고리즘 검증에 적합

### 2. MVTec AD 데이터셋
- 산업용 이상 탐지 표준 데이터셋
- 15개 카테고리 지원
- 픽셀 단위 ground truth 마스크 제공

**지원 카테고리:**
- Textures: carpet, grid, leather, tile, wood
- Objects: bottle, cable, capsule, hazelnut, metal_nut, pill, screw, toothbrush, transistor, zipper

## 성능 메트릭

훈련 중 자동으로 계산되는 주요 메트릭:

- **Image-level AUROC**: 이미지 전체의 정상/비정상 분류 성능
- **Pixel-level AUROC**: 픽셀 단위 이상 영역 탐지 성능 (마스크가 있는 경우)
- **Loss Components**: 
  - `st_loss`: Student-Teacher 손실
  - `ae_loss`: Student-Autoencoder 손실  
  - `penalty_loss`: 사전훈련 페널티 손실

## 결과 분석

### 1. TensorBoard 모니터링
```bash
tensorboard --logdir results/EfficientAD
```

### 2. 체크포인트 관리
- 최고 성능 모델: `best.ckpt`
- 최종 에포크 모델: `last.ckpt`
- 상위 3개 모델 자동 저장

### 3. 시각화 결과
추론 결과는 다음 형태로 저장됩니다:
- 원본 이미지
- 이상 탐지 히트맵
- Ground Truth (합성 데이터의 경우)

## 고급 사용법

### 1. 커스텀 데이터셋 사용
```python
from src.data.provider import DatasetProvider

class CustomDatasetProvider(DatasetProvider):
    def get_train_dataloader(self):
        # 커스텀 데이터로더 구현
        pass
```

### 2. 모델 아키텍처 수정
```python
from src.models.torch_model import EfficientADModel

# 커스텀 채널 수 사용
model = EfficientADModel(model_size='S', out_channels=512)
```

### 3. 손실 함수 가중치 조정
실험을 통해 최적의 가중치를 찾을 수 있습니다:
```yaml
st_weight: 2.0      # Student-Teacher 손실 강화
ae_weight: 1.0      
penalty_weight: 0.5 # 페널티 약화
```

## 문제 해결

### 1. 메모리 부족
```yaml
data:
  train_batch_size: 1  # 배치 크기 줄이기
  eval_batch_size: 16
```

### 2. 훈련 속도 개선
```yaml
trainer:
  accelerator: cuda
  devices: 1
  precision: 16  # Mixed precision training
```

### 3. 오버피팅 방지
```yaml
trainer:
  max_epochs: 50  # 에포크 수 줄이기
  
# Early stopping이 자동으로 적용됨 (patience=10)
```

## 배포 및 최적화

### 1. ONNX 변환 (예정)
```bash
python deployment/export_onnx.py --checkpoint path/to/model.ckpt
```

### 2. OpenVINO 변환 (예정)
```bash
python deployment/export_openvino.py --onnx_path model.onnx
```

## 참고 자료

- [EfficientAD 논문](https://arxiv.org/abs/2303.14535)
- [MVTec AD 데이터셋](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [PyTorch Lightning 문서](https://pytorch-lightning.readthedocs.io/)

## 라이선스

This implementation is for research and educational purposes.