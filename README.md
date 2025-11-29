# kmjAiWorld

MNIST 손글씨 숫자 인식을 위한 신경망 프로젝트입니다. PyTorch를 사용하여 구현된 신경망과 FastAPI 기반 REST API 서버를 제공합니다.

## 📋 프로젝트 개요

이 프로젝트는 MNIST 데이터셋을 사용하여 손글씨 숫자(0-9)를 인식하는 신경망을 학습하고, 학습된 모델을 FastAPI를 통해 REST API로 제공합니다.

### 주요 기능

- **신경망 구현**: PyTorch 기반 신경망
- **학습 알고리즘**: 미니배치 확률적 경사 하강법(Mini-Batch Gradient Descent)
- **REST API**: FastAPI를 통한 예측 서비스
- **CORS 지원**: 프론트엔드 연동을 위한 CORS 설정

## 🏗️ 프로젝트 구조

```
kmjAiWorld/
├── src/
│   ├── mnist_loader.py      # MNIST 데이터 로더
│   └── network.py            # 신경망 모델 구현
├── train/
│   └── train_network.py      # 모델 학습 스크립트
├── trained_data/
│   └── mnist_net.pth         # 학습된 모델 파일
├── data/                     # MNIST 데이터셋
├── main.py                   # FastAPI 서버
├── download_dataset.py       # 데이터셋 다운로드 스크립트
└── requirements.txt          # 의존성 패키지
```

## 🚀 시작하기

### 필수 요구사항

- Python 3.8 이상
- pip

### 설치

1. 저장소 클론
```bash
git clone https://github.com/kmj5004/kmjAiWorldApi.git
cd kmjAiWorld
```

2. 가상환경 생성 및 활성화
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

3. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

### 사용 방법

#### 1. 모델 학습
```bash
cd train
python train_network.py
```

학습 파라미터:
- 입력층: 784 뉴런 (28x28 픽셀)
- 은닉층: 30 뉴런
- 출력층: 10 뉴런 (0-9 숫자)
- 학습률(eta): 3.0
- 미니배치 크기: 10
- 에포크: 30
- L2 정규화: 0.01

#### 3. API 서버 실행
```bash
cd ..
uvicorn main:app --reload
```

서버는 `http://localhost:8000`에서 실행됩니다.

## 📡 API 사용법

### 엔드포인트

#### GET `/`
서버 상태 확인
```bash
curl http://localhost:8000/
```

#### POST `/network/predict`
손글씨 숫자 예측

**요청 형식:**
```json
{
  "image": [0.0, 0.1, 0.2, ..., 0.5]  // 784개의 float 값 (28x28 픽셀)
}
```

**응답 형식:**
```json
{
  "result": 5  // 예측된 숫자 (0-9)
}
```

**예제:**
```bash
curl -X POST "http://localhost:8000/network/predict" \
  -H "Content-Type: application/json" \
  -d '{"image": [0.0, 0.1, ...]}'
```

## 테스트 방법
Github의 kmjAiWorldWeb을 클론 받아 실행시키면 이미지 업로드, 28x28 사이즈로 압축, 픽셀값 계산, 예측값 까지 수월하게 테스트 가능합니다

## 🧠 신경망 아키텍처

- **입력층**: 784개 뉴런 (28x28 픽셀의 그레이스케일 이미지)
- **은닉층**: 30개 뉴런 (Sigmoid 활성화 함수)
- **출력층**: 10개 뉴런 (0-9 숫자 분류)

### 학습 알고리즘
- **최적화**: SGD (Stochastic Gradient Descent)
- **손실 함수**: Cross Entropy Loss
- **정규화**: L2 정규화 (weight_decay=0.01)

## 🛠️ 기술 스택

- **딥러닝**: PyTorch
- **웹 프레임워크**: FastAPI
- **데이터 처리**: NumPy, SciPy
- **시각화**: Matplotlib
- **데이터셋**: MNIST (torchvision)

## 📝 라이선스

이 프로젝트는 개인 학습 목적으로 작성되었습니다.

## 👤 작성자

kmjminjae
