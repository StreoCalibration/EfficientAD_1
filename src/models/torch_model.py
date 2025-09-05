import torch
import torch.nn as nn

# --- 주의 ---
# Teacher 모델은 인터넷을 통해 facebookresearch/deit 에서 사전 훈련된 모델을 다운로드합니다.
# 실행 환경에 인터넷 연결이 필요할 수 있습니다.

class Teacher(nn.Module):
    """사전 훈련된 모델을 사용하여 특징을 추출하는 Teacher 네트워크"""
    def __init__(self, model_name='deit_small_patch16_224', in_channels=3):
        super(Teacher, self).__init__()
        
        # 입력 채널이 3이 아닌 경우, 3으로 변환하는 어댑터 레이어
        if in_channels != 3:
            self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1)
        else:
            self.input_adapter = None

        # 지정된 DeiT 모델 로드
        self.model = torch.hub.load('facebookresearch/deit:main',
                                    model_name, pretrained=True)
        
        # Teacher 모델은 학습되지 않도록 설정
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.patch_size = self.model.patch_embed.patch_size
        self.num_patches = self.model.patch_embed.num_patches

    def forward(self, x):
        if self.input_adapter:
            x = self.input_adapter(x)

        with torch.no_grad():
            # DeiT 모델의 forward 로직 (CLS 토큰 제외)
            x = self.model.patch_embed(x)
            cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
            pos_embed = self.model.pos_embed
            x = torch.cat((cls_token, x), dim=1) + pos_embed
            x = self.model.pos_drop(x)
            for blk in self.model.blocks:
                x = blk(x)
            x = self.model.norm(x)
            x = x[:, 1:, :]  # CLS 토큰 제거
            
            # 특징 맵을 2D 형태로 재구성
            batch_size, _, channels = x.shape
            height = width = int(self.num_patches ** 0.5)
            x = x.reshape(batch_size, height, width, channels)
            x = x.permute(0, 3, 1, 2).contiguous()
        return x

class Student(nn.Module):
    """Teacher 네트워크의 출력을 모방하도록 학습하는 Student 네트워크"""
    def __init__(self, in_channels=384, out_channels=384):
        super(Student, self).__init__()
        # Teacher의 특징맵과 동일한 크기의 입력을 받아 처리하는 간단한 CNN
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.model(x)

class Autoencoder(nn.Module):
    """입력 이미지를 재구성하도록 학습하는 오토인코더"""
    def __init__(self, in_channels=3, out_channels=3):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class EfficientAD(nn.Module):
    """Teacher, Student, Autoencoder를 결합한 EfficientAD 모델"""
    def __init__(self, model_size='s', in_channels=3):
        super().__init__()
        self.model_size = model_size.lower() # 모델 사이즈를 소문자로 변환
        
        if self.model_size == 's':
            teacher_out_channels = 384
            teacher_model_name = 'deit_small_patch16_224'
        elif self.model_size == 'm':
            teacher_out_channels = 768
            teacher_model_name = 'deit_base_patch16_224' # 'm' 사이즈일 때 사용할 모델
        else:
            raise ValueError(f"알 수 없는 모델 사이즈: {model_size}")

        self.teacher = Teacher(model_name=teacher_model_name, in_channels=in_channels)
        self.student = Student(in_channels=teacher_out_channels, out_channels=teacher_out_channels)
        self.autoencoder = Autoencoder(in_channels=in_channels, out_channels=in_channels)

    def forward(self, x):
        teacher_output = self.teacher(x)
        student_output = self.student(teacher_output)
        ae_output = self.autoencoder(x)
        return teacher_output, student_output, ae_output