# kmjAiWorld

AI/머신러닝 기반 멀티 도메인 프로젝트입니다. MNIST 손글씨 인식, 틱택토 AI, 오목(Gomoku) AI를 FastAPI REST API와 웹 UI로 제공합니다.

## 프로젝트 개요

이 프로젝트는 세 가지 AI 도메인을 제공합니다:

| 도메인 | 설명 | 핵심 기술 |
|--------|------|-----------|
| **MNIST** | 손글씨 숫자(0-9) 인식 | PyTorch MLP, Adam, Dropout |
| **틱택토** | 신경망 기반 게임 AI | 4층 Feedforward Network |
| **오목** | 15x15 보드 게임 AI | ResNet + MCTS (AlphaGo Zero 스타일) |

### 주요 기능

- **딥러닝 모델 서빙**: FastAPI를 통한 실시간 추론 API
- **WebSocket 실시간 모니터링**: 학습 진행률 실시간 브로드캐스트
- **Self-Play 강화학습**: AlphaGo Zero 방식의 자기 대국 학습 파이프라인
- **하이퍼파라미터 자동 최적화**: Multiprocessing 기반 병렬 Grid Search
- **웹 게임 UI**: 브라우저에서 바로 플레이 가능한 틱택토/오목 인터페이스
- **GPU 자동 감지**: CUDA 가용 시 자동 GPU 학습 전환

## 프로젝트 구조

```
kmjAiWorld/
├── src/
│   ├── mnist/                    # MNIST 도메인
│   │   ├── config_utils.py       # 하이퍼파라미터 설정 관리
│   │   ├── mnist_loader.py       # 데이터 로더
│   │   ├── mnist_router.py       # API 라우터
│   │   └── network.py            # MLP 신경망 (784→256→128→10)
│   ├── tictactoe/                # 틱택토 도메인
│   │   ├── tictactoe_ai.py       # AI 모델 (9→128→64→32→9)
│   │   └── tictactoe_router.py   # API 라우터
│   └── gomoku/                   # 오목 도메인
│       ├── game.py               # 게임 로직 (15x15 보드, 승리 판정)
│       ├── network.py            # ResNet + Policy/Value Head
│       ├── mcts.py               # Monte Carlo Tree Search
│       ├── self_play.py          # Self-Play 학습 데이터 생성
│       ├── parallel_train.py     # 병렬 학습 파이프라인
│       └── gomoku_router.py      # API 라우터
├── train/
│   ├── train_network.py          # MNIST 학습
│   ├── train_gomoku_simple.py    # 오목 단일 학습
│   ├── train_gomoku_parallel.py  # 오목 병렬 학습
│   ├── train_gomoku.py           # 오목 학습 스크립트
│   └── find_eta_lamda.py         # 하이퍼파라미터 최적화 (Grid Search)
├── test/
│   └── test_network.py           # MNIST 모델 평가
├── static/
│   ├── tictactoe.html            # 틱택토 웹 게임 UI
│   └── gomoku.html               # 오목 웹 게임 UI
├── trained_data/
│   └── mnist_net.pth             # 학습된 MNIST 모델
├── models/
│   ├── tictactoe_model.pth       # 학습된 틱택토 모델
│   └── gomoku/gomoku_net.pth     # 학습된 오목 모델
├── data/                         # MNIST 데이터셋 (자동 다운로드)
├── main.py                       # FastAPI 진입점 (라우터 통합)
├── best_config.json              # MNIST 최적 하이퍼파라미터
└── requirements.txt              # 의존성 패키지
```

## 시작하기

### 필수 요구사항

- Python 3.8 이상
- pip

### 설치

```bash
git clone https://github.com/kmj5004/kmjAiWorldApi.git
cd kmjAiWorldApi

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### API 서버 실행

```bash
uvicorn main:app --reload
```

서버: `http://localhost:8000` / Swagger 문서: `http://localhost:8000/docs`

### 모델 학습

```bash
# MNIST 학습
python train/train_network.py

# 오목 학습 (Self-Play)
python train/train_gomoku_parallel.py   # 병렬 학습
python train/train_gomoku_simple.py     # 단일 프로세스 학습

# MNIST 하이퍼파라미터 최적화
python train/find_eta_lamda.py

# MNIST 모델 평가
python test/test_network.py
```

## 신경망 아키텍처

### MNIST — MLP (다층 퍼셉트론)

```
Input(784) → Linear(256) → ReLU → Dropout(0.4)
           → Linear(128) → ReLU → Dropout(0.4)
           → Linear(10)
```

- **Optimizer**: Adam (lr: 0.01~0.06, weight_decay: 0.001)
- **Loss**: Cross-Entropy Loss
- **정규화**: Dropout(0.4) + L2 Regularization

### 틱택토 — Feedforward Network

```
Input(9) → Linear(128) → ReLU → Dropout(0.3)
         → Linear(64)  → ReLU → Dropout(0.3)
         → Linear(32)  → ReLU
         → Linear(9)   → Softmax
```

- **Optimizer**: Adam (lr: 0.001)
- **Loss**: Cross-Entropy Loss
- **학습 방식**: 지도 학습 (게임 데이터셋), Early Stopping 적용

### 오목 — ResNet + Dual Head (AlphaGo Zero 스타일)

```
Input(3×15×15) → Conv2d(128) → BatchNorm → ReLU
               → ResidualBlock × 5
               ├─ Policy Head → Conv(2) → BN → FC(225) → Log-Softmax  (착수 확률)
               └─ Value Head  → Conv(1) → BN → FC(256) → FC(1) → Tanh (승률 평가)
```

- **Residual Block**: Conv → BN → ReLU → Conv → BN + Skip Connection → ReLU
- **Optimizer**: Adam (lr: 0.001)
- **Loss**: Policy Loss (Negative Log-Likelihood) + Value Loss (MSE)
- **학습 방식**: Self-Play 강화학습 + Experience Replay (버퍼 10,000)

### MCTS (Monte Carlo Tree Search)

- **UCB 스코어**: `value + c_puct × prior × √(parent_visits) / (1 + visits)` (c_puct: 1.4)
- **시뮬레이션 횟수**: Easy(100) / Normal(400) / Hard(800)
- **Temperature**: 초반(1.0, 탐색) → 후반(0.1, 활용)
- **프로세스**: Selection → Expansion → Evaluation(신경망) → Backup

## API 엔드포인트

### MNIST (`/mnist`)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/mnist/predict` | 손글씨 숫자 예측 (784개 float 배열) |
| POST | `/mnist/train` | 모델 학습 시작 (BackgroundTask) |
| POST | `/mnist/train/stop` | 학습 중단 |
| GET | `/mnist/train/status` | 학습 진행 상태 |
| GET | `/mnist/test` | 테스트셋 정확도 평가 |
| POST | `/mnist/optimize` | 하이퍼파라미터 자동 최적화 |
| GET | `/mnist/optimize/status` | 최적화 진행 상태 |
| POST | `/mnist/reload` | 저장된 모델 재로드 |
| GET | `/mnist/best-config` | 최적 하이퍼파라미터 조회 |
| WS | `/mnist/ws/training` | 학습 진행률 실시간 스트리밍 |

### 틱택토 (`/tictactoe`)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/tictactoe/new-game` | 새 게임 시작 |
| POST | `/tictactoe/move` | AI 수 계산 (보드: 9개 정수, 0/1/-1) |
| POST | `/tictactoe/check-winner` | 승자 확인 |
| GET | `/tictactoe/train-status` | 모델 학습 상태 |
| GET | `/tictactoe/board-representation` | 보드 표현 정보 |

### 오목 (`/gomoku`)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/gomoku/new-game` | 새 게임 시작 (세션 ID 발급) |
| POST | `/gomoku/move` | AI 수 계산 (MCTS + 신경망, 난이도 선택) |
| POST | `/gomoku/human-move` | 사람 수 입력 |
| GET | `/gomoku/model-status` | 모델 로드 상태 |
| GET | `/gomoku/info` | 게임 규칙 및 AI 정보 |

### 웹 게임

- 틱택토: `http://localhost:8000/static/tictactoe.html`
- 오목: `http://localhost:8000/static/gomoku.html`

## 기술 스택

| 분류 | 기술 |
|------|------|
| **딥러닝** | PyTorch (MLP, CNN, ResNet) |
| **웹 프레임워크** | FastAPI, Uvicorn, WebSocket |
| **데이터 처리** | NumPy, Pandas, scikit-learn |
| **시각화** | Matplotlib |
| **병렬 처리** | multiprocessing, ProcessPoolExecutor, asyncio |
| **데이터셋** | MNIST (torchvision), 틱택토 CSV, 오목 Self-Play 생성 |

## 테스트 방법

### MNIST
[kmjAiWorldWeb](https://github.com/kmj5004/kmjAiWorldWeb)을 클론하여 실행하면 이미지 업로드, 28x28 리사이즈, 픽셀값 변환, 예측까지 테스트할 수 있습니다.

### 틱택토 / 오목
서버 실행 후 브라우저에서 각 게임의 웹 UI에 접속하여 AI와 대결할 수 있습니다.

## 라이선스

이 프로젝트는 개인 학습 목적으로 작성되었습니다.

## 작성자

kmjminjae
