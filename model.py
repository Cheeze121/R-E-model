import torch
import torch.nn as nn
import torchvision.models as models

class LuminolResNet18(nn.Module):
    def __init__(self, hidden_dim=256, dropout_rate=0.5):
        super(LuminolResNet18, self).__init__()
        
        # 1. ImageNet에 사전 학습된 ResNet-18 모델 불러오기 (특징 추출기 역할)
        # pretrained=True 대신 최신 torchvision 버전에 맞게 weights='DEFAULT' 권장
        # 호환성을 위해 구버전/신버전 모두 대응 가능한 방식 사용
        try:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except AttributeError:
            self.backbone = models.resnet18(pretrained=True)
            
        # 2. ResNet의 기본 구조 파악
        # ResNet18은 Conv1 -> Layer1 -> Layer2 -> Layer3 -> Layer4를 거쳐
        # 512차원의 특징 맵(Feature Map)을 출력합니다. (PPT 11페이지 내용과 일치)
        
        # 기본 Backbone에서 마지막 Fully Connected Layer(fc) 부분을 제외하고 추출
        self.features = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1, # 64 채널
            self.backbone.layer2, # 128 채널
            self.backbone.layer3, # 256 채널
            self.backbone.layer4, # 512 채널 (과제 3 다. 모델설명 PPT 반영)
        )
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. PPT 구조에 맞춘 Custom Regression Head 추가
        # 512가지 특징 -> 종합 -> 최종 경과 시간(1개 값) 출력
        self.regression_head = nn.Sequential(
            nn.Linear(512, hidden_dim), # FC Layer 1
            nn.ReLU(),                  # ReLU
            nn.Dropout(p=dropout_rate), # Dropout
            nn.Linear(hidden_dim, 1)    # FC Layer 2 (Output: 시간)
        )

    def forward(self, x):
        # 특징 추출
        x = self.features(x)
        
        # Global Average Pooling (B, 512, H, W) -> (B, 512, 1, 1)
        x = self.avgpool(x)
        
        # Flatten (B, 512, 1, 1) -> (B, 512)
        x = torch.flatten(x, 1)
        
        # 회귀 헤드를 통과하여 시간 예측
        x = self.regression_head(x)
        
        # x 차원이 (Batch_size, 1) 이므로 (Batch_size) 형태로 반환하려면 squeeze() 가능하지만,
        # Loss 계산 시 unsqueeze()된 타겟과 맞추는 것이 편하므로 그냥 반환합니다.
        return x
