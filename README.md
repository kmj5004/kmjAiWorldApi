# kmjAiWorld

AI를 활용한 머신러닝 프로젝트입니다. MNIST 손글씨 숫자 인식 신경망과 틱택토(Tic-Tac-Toe) AI 게임을 제공합니다.

## 📋 프로젝트 개요

이 프로젝트는 두 가지 주요 AI 기능을 제공합니다:
1. MNIST 데이터셋을 사용한 손글씨 숫자(0-9) 인식
2. 신경망 기반 틱택토 AI 게임

### 주요 기능

- **MNIST 신경망**: PyTorch 기반 손글씨 숫자 인식
- **틱택토 AI**: 신경망 기반 틱택토 게임 AI
- **학습 알고리즘**: 미니배치 확률적 경사 하강법(Mini-Batch Gradient Descent)
- **REST API**: FastAPI를 통한 예측 서비스
- **웹 게임 인터페이스**: 틱택토 게임 HTML/JavaScript UI
- **CORS 지원**: 프론트엔드 연동을 위한 CORS 설정

## 🏗️ 프로젝트 구조

```
kmjAiWorld/
├── src/
│   ├── mnist/                # MNIST 도메인
│   │   ├── config_utils.py   # 설정 관리
│   │   ├── mnist_loader.py   # 데이터 로더
│   │   ├── mnist_router.py   # API 라우터
│   │   └── network.py        # 신경망 모델
│   └── tictactoe/            # 틱택토 도메인
│       ├── tictactoe_ai.py   # AI 모델
│       └── tictactoe_router.py # API 라우터
├── train/
│   └── train_network.py      # 모델 학습 스크립트
├── trained_data/
│   └── mnist_net.pth         # 학습된 MNIST 모델
├── models/
│   └── tictactoe_model.pth   # 학습된 틱택토 모델
├── static/
│   └── tictactoe.html        # 틱택토 게임 웹 UI
├── data/                     # MNIST 데이터셋
├── main.py                   # FastAPI 서버 (라우터 통합)
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

### 메인 엔드포인트

#### GET `/`
서버 상태 및 사용 가능한 엔드포인트 확인
```bash
curl http://localhost:8000/
```

### MNIST API

모든 MNIST 엔드포인트는 `/mnist` 경로로 시작합니다.

#### POST `/mnist/predict`
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
curl -X POST "http://localhost:8000/mnist/predict" \
  -H "Content-Type: application/json" \
  -d '{"image": [0.0, 0.1, ...]}'
```

#### 기타 MNIST 엔드포인트
- `POST /mnist/train` - 모델 학습
- `GET /mnist/test` - 모델 평가
- `GET /mnist/best-config` - 최적 설정 조회
- `POST /mnist/optimize` - 하이퍼파라미터 최적화
- `WS /mnist/ws/training` - 학습 진행 상황 (WebSocket)

### 틱택토 API

#### GET `/tictactoe/new-game`
새로운 게임 시작
```bash
curl http://localhost:8000/tictactoe/new-game
```

#### POST `/tictactoe/move`
AI의 다음 수 얻기

**요청 형식:**
```json
{
  "board": [1, 0, 0, 0, -1, 0, 0, 0, 0]  // 9개의 정수 (0=빈칸, 1=X, -1=O)
}
```

**응답 형식:**
```json
{
  "move": 2,
  "board": [1, 0, -1, 0, -1, 0, 0, 0, 0],
  "winner": null,
  "message": "AI plays position 2"
}
```

#### 웹 게임 플레이
브라우저에서 `http://localhost:8000/static/tictactoe.html`로 접속하여 AI와 게임을 플레이할 수 있습니다.

## 테스트 방법

### MNIST
Github의 kmjAiWorldWeb을 클론 받아 실행시키면 이미지 업로드, 28x28 사이즈로 압축, 픽셀값 계산, 예측값 까지 수월하게 테스트 가능합니다

### 틱택토
1. 서버 실행 후 브라우저에서 `http://localhost:8000/static/tictactoe.html` 접속
2. 웹 UI를 통해 AI와 대결
3. 또는 API 엔드포인트를 직접 호출하여 테스트

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
- **데이터 처리**: NumPy, SciPy, Pandas
- **머신러닝**: scikit-learn
- **시각화**: Matplotlib
- **데이터셋**: MNIST (torchvision), 틱택토 게임 데이터

## 📝 라이선스

이 프로젝트는 개인 학습 목적으로 작성되었습니다.

## 👤 작성자

kmjminjae
