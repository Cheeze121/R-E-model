import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def crop_luminescent_part(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_img = img[y:y+h, x:x+w]
    
    cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped_rgb)

class BloodstainDataset(Dataset):
    def __init__(self, csv_file, is_train=True, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        
        if transform is None:
            if is_train:
                # 학습용: 데이터 증강(Augmentation) 추가 - 뒤집기, 회전 등
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5), # 좌우 반전
                    transforms.RandomVerticalFlip(p=0.5),   # 상하 반전
                    transforms.RandomRotation(degrees=180), # 임의의 각도로 회전
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                # 검증/테스트용: 증강 없이 원본만 평가
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data_frame.iloc[idx]['image_path']
        image = crop_luminescent_part(img_path)
        time_label = float(self.data_frame.iloc[idx]['time'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(time_label, dtype=torch.float32)
