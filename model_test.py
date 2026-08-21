import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from dataset import crop_luminescent_part
from model import LuminolResNet18

def predict_and_explain(image_path, model_weight_path="best_model_fold_1.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LuminolResNet18(hidden_dim=256, dropout_rate=0.5)
    
    if not os.path.exists(model_weight_path):
        print(f"[오류] 학습된 가중치 파일 '{model_weight_path}'을 찾을 수 없습니다.")
        return
        
    model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval() 
    
    print(f"\n🔍 [{image_path}] 이미지 분석 중...")
    try:
        cropped_img = crop_luminescent_part(image_path)
    except Exception as e:
        print(f"이미지 전처리 중 오류 발생: {e}")
        return
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(cropped_img)
    input_batch = input_tensor.unsqueeze(0).to(device)
    input_batch.requires_grad = True # Grad-CAM을 위해 기울기 활성화

    # --- Grad-CAM 구현 (시각화) ---
    # 마지막 합성곱 층(layer4)의 피처맵과 그래디언트를 가져오기 위한 Hook 설정
    feature_maps = []
    gradients = []
    
    def forward_hook(module, input, output):
        feature_maps.append(output)
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
        
    # ResNet18의 마지막 특징 추출 층(layer4)에 훅 걸기
    target_layer = model.features[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)
    
    # 순전파(예측)
    output = model(input_batch)
    predicted_time = output.item()
    
    # 역전파(기울기 계산)
    model.zero_grad()
    output.backward()
    
    # Grad-CAM 히트맵 생성
    weights = torch.mean(gradients[0], dim=(2, 3))[0, :] # Global Average Pooling over gradients
    cam = torch.zeros(feature_maps[0].shape[2:], dtype=torch.float32).to(device)
    
    for i, w in enumerate(weights):
        cam += w * feature_maps[0][0, i, :, :]
        
    cam = torch.relu(cam) # ReLU를 통과시켜 양수(영향을 준 부분)만 남김
    cam = cam.cpu().detach().numpy()
    
    # 정규화 (0 ~ 1)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)
        
    # 히트맵을 원본 이미지 크기(224x224)로 확대
    cam_img = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_img), cv2.COLORMAP_JET)
    
    # 크롭된 원본 이미지 (역정규화 적용)
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    inv_tensor = inv_normalize(input_tensor)
    img_display = inv_tensor.permute(1, 2, 0).cpu().numpy()
    img_display = np.clip(img_display, 0, 1)
    img_display_uint8 = np.uint8(255 * img_display)
    
    # 원본 이미지와 히트맵 합성
    heatmap_overlay = cv2.addWeighted(img_display_uint8, 0.6, heatmap, 0.4, 0)
    
    # 결과 이미지 저장 (보고서용)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_display)
    plt.title("Cropped Input")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    
    plt.suptitle(f"Predicted Time: {predicted_time:.2f}")
    plt.tight_layout()
    plt.savefig("result_heatmap.png")
    
    print(f"✅ 분석 완료! 인공지능이 예측한 혈흔 경과 시간은 약 **{predicted_time:.2f}** 입니다.")
    print("📁 AI가 판단의 근거로 삼은 부위가 'result_heatmap.png'로 저장되었습니다.")

if __name__ == "__main__":
    TEST_IMAGE_PATH = "test_img.jpg"
    SAVED_MODEL_WEIGHT = "best_model_fold_1.pth"
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"[알림] '{TEST_IMAGE_PATH}' 파일이 같은 폴더에 없습니다.")
    else:
        # pip install matplotlib
        predict_and_explain(image_path=TEST_IMAGE_PATH, model_weight_path=SAVED_MODEL_WEIGHT)
