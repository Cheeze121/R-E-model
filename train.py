import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np

from dataset import BloodstainDataset
from model import LuminolResNet18

def train_kfold(csv_file, num_epochs=30, batch_size=16, k_folds=5, learning_rate=1e-4, hidden_dim=256, dropout=0.5):
    # 디바이스 설정 (GPU 사용 가능하면 GPU, 아니면 CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 기기: {device}")

    # 전체 데이터셋 로드
    dataset = BloodstainDataset(csv_file=csv_file)
    print(f"총 데이터 개수: {len(dataset)}개")

    # K-Fold 교차 검증 설정 (계획서 요구사항)
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # 평가 지표인 MAE(Mean Absolute Error)를 저장할 리스트
    fold_results = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"\n--- Fold {fold + 1}/{k_folds} 시작 ---")
        
        # 각 Fold에 맞게 데이터 분할
        train_sub = Subset(dataset, train_ids)
        val_sub = Subset(dataset, val_ids)
        
        # DataLoader 설정
        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)
        
        # 모델 초기화 (각 Fold마다 모델을 새로 학습)
        model = LuminolResNet18(hidden_dim=hidden_dim, dropout_rate=dropout)
        model = model.to(device)
        
        # 손실 함수(MAE) 및 옵티마이저 설정
        criterion = nn.L1Loss() # MAE
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        best_val_mae = float('inf')
        
        for epoch in range(num_epochs):
            # --- 학습(Train) 단계 ---
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).unsqueeze(1) # 차원 맞추기: (B, 1)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                
            train_loss = train_loss / len(train_sub)
            
            # --- 검증(Validation) 단계 ---
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device).unsqueeze(1)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
                    
            val_loss = val_loss / len(val_sub)
            
            # 10 에폭마다 출력
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] - Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f}")
            
            # 가장 성능이 좋은 모델 가중치 임시 저장
            if val_loss < best_val_mae:
                best_val_mae = val_loss
                torch.save(model.state_dict(), f"best_model_fold_{fold+1}.pth")
                
        print(f"Fold {fold + 1}의 최고 성능 (Best Val MAE): {best_val_mae:.4f}")
        fold_results.append(best_val_mae)
        
    # K-Fold 전체 결과 출력
    print(f"\n===== K-Fold 교차 검증 최종 결과 =====")
    for i, res in enumerate(fold_results):
        print(f"Fold {i+1}: {res:.4f}")
    print(f"평균 MAE: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")

if __name__ == '__main__':
    # 학습할 CSV 파일 이름 (직접 만든 파일 이름으로 변경하세요)
    CSV_FILE_PATH = "data.csv"
    
    # 샘플 CSV 파일이 없는 경우를 위한 안내 (데이터셋 생성 후 실행할 수 있도록)
    if not os.path.exists(CSV_FILE_PATH):
        print(f"[알림] '{CSV_FILE_PATH}' 파일이 없습니다.")
        print("실제 학습을 진행하려면 'image_path'와 'time' 컬럼이 있는 CSV 파일을 준비해주세요.")
    else:
        # 학습 루프 실행
        train_kfold(
            csv_file=CSV_FILE_PATH,
            num_epochs=30,      # 테스트 시 에폭을 줄여서 빠르게 확인 가능
            batch_size=8,       # GPU VRAM 상황에 맞춰 조절
            k_folds=5           # 가장 보편적인 5-Fold
        )
