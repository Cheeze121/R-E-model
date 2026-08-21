import os
import torch
import torchvision.transforms as transforms
from dataset import crop_luminescent_part
from model import LuminolResNet18

def predict_time(image_path, model_weight_path="best_model_fold_1.pth"):
    """
    단일 이미지 파일을 입력받아 학습된 모델을 통해 경과 시간을 예측합니다.
    """
    # 1. 디바이스 설정 (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. 모델 구조 초기화 (train.py에서 사용한 것과 동일한 구조여야 함)
    model = LuminolResNet18(hidden_dim=256, dropout_rate=0.5)
    
    # 학습된 가중치 파일(.pth)이 있는지 확인
    if not os.path.exists(model_weight_path):
        print(f"[오류] 학습된 가중치 파일 '{model_weight_path}'을 찾을 수 없습니다.")
        print("먼저 train.py를 성공적으로 끝까지 실행하여 모델을 학습시켜야 합니다.")
        return
        
    # 가중치 덮어쓰기 (weights_only=True는 보안 권장사항)
    model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True))
    model = model.to(device)
    
    # 검증/추론 모드로 전환 (Dropout 비활성화, BatchNorm 고정 등)
    model.eval() 
    
    # 3. 이미지 전처리 수행
    print(f"\n🔍 [{image_path}] 이미지 분석 중...")
    try:
        # dataset.py에 만들어둔 '발광 영역 자동 크롭' 기능 그대로 활용
        cropped_img = crop_luminescent_part(image_path)
    except Exception as e:
        print(f"이미지 전처리 중 오류 발생: {e}")
        return
        
    # PyTorch 입력 형태에 맞게 224x224 리사이징 및 텐서 변환
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(cropped_img)
    
    # 모델은 Batch 단위로 처리하므로 맨 앞에 차원을 하나 추가해줍니다.
    # 형태 변화: (C, H, W) -> (1, C, H, W)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # 4. 인공지능 예측 수행
    with torch.no_grad(): # 역전파(학습)를 하지 않으므로 메모리 절약
        output = model(input_batch)
        # 텐서에서 숫자 값만 뽑아냅니다.
        predicted_time = output.item() 
        
    print(f"✅ 분석 완료! 인공지능이 예측한 혈흔 경과 시간은 약 **{predicted_time:.2f}** 입니다.")

if __name__ == "__main__":
    # 테스트에 사용할 이미지 파일 경로
    # 이곳에 테스트할 사진의 이름을 적어주세요.
    TEST_IMAGE_PATH = "test_img.jpg"
    
    # 사용할 학습 완료된 모델 가중치 이름 (train.py 실행 후 생성됨)
    SAVED_MODEL_WEIGHT = "best_model_fold_1.pth"
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"[알림] '{TEST_IMAGE_PATH}' 파일이 같은 폴더에 없습니다.")
        print("사진을 준비한 뒤 폴더에 넣고, 코드를 실행해주세요.")
    else:
        predict_time(image_path=TEST_IMAGE_PATH, model_weight_path=SAVED_MODEL_WEIGHT)
