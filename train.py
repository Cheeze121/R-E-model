import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

from dataset import BloodstainDataset
from model import LuminolResNet18

def train_kfold(csv_file, num_epochs=30, batch_size=16, k_folds=5, learning_rate=1e-4, hidden_dim=256, dropout=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 기기: {device}")

    # 데이터 증강을 위해 학습용/검증용 데이터셋을 분리해서 로드
    train_dataset_full = BloodstainDataset(csv_file=csv_file, is_train=True)
    val_dataset_full = BloodstainDataset(csv_file=csv_file, is_train=False)
    
    print(f"총 데이터 개수: {len(train_dataset_full)}개")

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    
    # 5-Fold 전체의 에폭별 평균 Loss를 기록할 딕셔너리
    history = {'train_loss': np.zeros(num_epochs), 'val_loss': np.zeros(num_epochs)}
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset_full)):
        print(f"\n--- Fold {fold + 1}/{k_folds} 시작 ---")
        
        # 증강이 켜진 학습 셋에서 train_ids만 추출
        train_sub = Subset(train_dataset_full, train_ids)
        # 증강이 꺼진 검증 셋에서 val_ids만 추출
        val_sub = Subset(val_dataset_full, val_ids)
        
        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)
        
        model = LuminolResNet18(hidden_dim=hidden_dim, dropout_rate=dropout).to(device)
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        best_val_mae = float('inf')
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            train_loss = train_loss / len(train_sub)
            
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
            
            # 히스토리 기록 (나중에 그래프를 그리기 위함)
            history['train_loss'][epoch] += train_loss / k_folds
            history['val_loss'][epoch] += val_loss / k_folds
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] - Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f}")
            
            if val_loss < best_val_mae:
                best_val_mae = val_loss
                torch.save(model.state_dict(), f"best_model_fold_{fold+1}.pth")
                
        print(f"Fold {fold + 1}의 최고 성능 (Best Val MAE): {best_val_mae:.4f}")
        fold_results.append(best_val_mae)
        
    print(f"\n===== K-Fold 교차 검증 최종 결과 =====")
    for i, res in enumerate(fold_results):
        print(f"Fold {i+1}: {res:.4f}")
    print(f"평균 MAE: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    
    # 보고서 작성용 Loss 그래프 자동 저장
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), history['train_loss'], label='Train MAE')
    plt.plot(range(1, num_epochs + 1), history['val_loss'], label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error (Time)')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_graph.png')
    print("\n[알림] 학습 오차 그래프가 'loss_graph.png'로 저장되었습니다. (보고서 첨부용)")

if __name__ == '__main__':
    CSV_FILE_PATH = "bloodstain_data.csv"
    if not os.path.exists(CSV_FILE_PATH):
        print(f"[알림] '{CSV_FILE_PATH}' 파일이 없습니다. (실제 데이터 획득 후 실행하세요)")
    else:
        train_kfold(
            csv_file=CSV_FILE_PATH,
            num_epochs=30,      
            batch_size=8,       
            k_folds=5           
        )
