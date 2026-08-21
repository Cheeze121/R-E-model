# 혈흔 경과 시간 예측 (ResNet-18)

이 폴더에는 혈흔의 커피링 현상과 루미놀 반응을 이용해 훼손된 혈흔의 경과 시간을 예측하는 PyTorch 딥러닝 코드 모델이 담겨있습니다.

## 📂 파일 구성
1. **`dataset.py`**: 
   - `crop_luminescent_part()`: OpenCV의 Otsu 알고리즘을 사용하여 촬영된 이미지 내에서 스스로 가장 밝은 발광체를 찾아 크롭(Crop)합니다.
   - `BloodstainDataset`: 이미지를 224x224로 리사이징하고 텐서로 변환하는 과정을 포함합니다.
2. **`model.py`**:
   - `LuminolResNet18`: 사전 학습된 ResNet-18을 Backbone으로 사용하며, 마지막을 분류(Classification)가 아닌 시간 회귀(Regression)를 위한 구조(Global Average Pooling ➔ FC1 ➔ ReLU ➔ Dropout ➔ FC2 ➔ Output(1))로 변경한 모델입니다.
3. **`train.py`**:
   - 5-Fold 교차 검증을 통해 모델의 범용적인 성능을 테스트합니다.
   - 손실 함수는 계획서에 기재된 MAE(Mean Absolute Error, L1 Loss)를 사용합니다.

## 🚀 사용 방법
### 1. CSV 데이터셋 준비
학습을 위해서는 이미지 파일이 있는 경로와 정답 시간 라벨이 적힌 CSV 파일이 필요합니다. 엑셀에서 다음과 같은 형태로 작성 후 `bloodstain_data.csv` 이름으로 저장하세요. (파일 이름은 `train.py` 맨 밑에서 변경 가능합니다)

| image_path | time |
|---|---|
| ./images/blood_10min_01.jpg | 10 |
| ./images/blood_20min_01.jpg | 20 |
| ./images/blood_30min_01.jpg | 30 |

### 2. 필요한 라이브러리 설치
터미널을 열고 다음 명령어를 통해 라이브러리를 설치합니다.
```bash
pip install -r requirements.txt
```

### 3. 학습 시작하기
터미널이나 명령 프롬프트에서 `train.py`를 실행합니다.
```bash
python train.py
```

## 🛠️ 추후 수정이 필요한 부분
- 이미지를 촬영하는 조건(암실, 플래시 등)에 따라 `dataset.py` 내의 크롭 감도(threshold) 알고리즘 수정이 필요할 수도 있습니다.
- GPU 메모리 용량에 따라 `train.py` 맨 밑에 있는 `batch_size`를 4, 8, 16, 32 등으로 적절히 변경해서 실행하세요.
