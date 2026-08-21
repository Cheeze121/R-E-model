import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def crop_luminescent_part(image_path):
    """
    암실에서 촬영된 루미놀 반응 이미지에서 발광 영역만 자동으로 찾아 크롭합니다.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 발광 영역을 찾기 위한 이진화 (Otsu 알고리즘 사용)
    # 암실 촬영이므로 밝은 부분(발광 부분)이 뚜렷하게 구분됨
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 윤곽선이 발견되지 않은 경우 원본 반환
    if not contours:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    
    # 가장 넓은 영역의 윤곽선 찾기 (가장 큰 발광체)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 해당 윤곽선을 포함하는 최소 크기의 사각형(Bounding Box) 구하기
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # 여유 공간(Padding)을 조금 주고 자를 수도 있지만, 일단 타이트하게 크롭
    cropped_img = img[y:y+h, x:x+w]
    
    # OpenCV의 BGR을 RGB로 변환 후 PIL Image 객체로 반환
    cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped_rgb)

class BloodstainDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        csv_file: 이미지 경로와 정답 시간(Time)이 포함된 CSV 파일 경로
        """
        self.data_frame = pd.read_csv(csv_file)
        
        # PPT 과제 3 요구사항에 맞춰 224x224 리사이징 및 정규화 적용
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet 기본 평균
                                     std=[0.229, 0.224, 0.225])  # ImageNet 기본 표준편차
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # CSV 파일의 컬럼명은 'image_path'와 'time'으로 가정
        img_path = self.data_frame.iloc[idx]['image_path']
        
        # 발광 영역 자동 크롭
        image = crop_luminescent_part(img_path)
        
        # 정답 라벨(시간) 가져오기
        time_label = float(self.data_frame.iloc[idx]['time'])
        
        # 텐서 및 리사이즈 변환 적용
        if self.transform:
            image = self.transform(image)
            
        # 회귀(Regression) 모델이므로 float32 텐서로 반환
        return image, torch.tensor(time_label, dtype=torch.float32)
