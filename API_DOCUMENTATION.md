# API 문서 - 신경망 학습 시스템

프론트엔드 개발자를 위한 API 엔드포인트 가이드입니다.

## 기본 정보

- **Base URL**: `http://localhost:8000`
- **WebSocket URL**: `ws://localhost:8000/ws/training`
- **Content-Type**: `application/json`

---

## 📡 WebSocket

### `/ws/training`

실시간 학습 진행상황과 최적화 결과를 전송받는 WebSocket 연결입니다.

**연결 방법:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/training');

ws.onopen = () => {
    console.log('WebSocket 연결됨');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('받은 데이터:', data);
};
```

**수신 데이터 타입:**

#### 1. 학습 진행 데이터
```javascript
{
    "epoch": 1,                    // 현재 에포크
    "total_epochs": 30,            // 전체 에포크 수
    "loss": 0.1234,                // 손실값
    "accuracy": 9500,              // 정답 개수
    "accuracy_percentage": 95.0,   // 정확도 (%)
    "total_test": 10000            // 전체 테스트 수
}
```

#### 2. 학습 완료/중지/오류
```javascript
{
    "status": "completed",         // "completed" | "stopped" | "error"
    "message": "학습 완료 및 모델 저장됨 (자동 재로드 완료)"
}
```

#### 3. 최적화 진행 데이터
```javascript
{
    "optimization": true,
    "iteration": 1,                // 현재 반복 횟수
    "total_iterations": 5,         // 전체 반복 횟수
    "combination": 2,              // 현재 조합 번호
    "total_combinations": 18,      // 전체 조합 수
    "optimization_result": {
        "iteration": 1,
        "eta": 0.01,
        "lambda": 0.0001,
        "accuracy": 9550,
        "percentage": 95.5,
        "total": 10000,
        "is_best": false
    },
    "current_best_eta": 0.01,
    "current_best_lambda": 0.0001,
    "current_best_percentage": 95.5
}
```

#### 4. 최적화 완료
```javascript
{
    "optimization": true,
    "status": "completed",
    "best_eta": 0.01,
    "best_lambda": 0.0001,
    "best_percentage": 95.5,
    "all_results": [               // 모든 조합의 결과 배열
        {
            "iteration": 1,
            "eta": 0.009,
            "lambda": 0.00001,
            "accuracy": 9500,
            "percentage": 95.0,
            "total": 10000,
            "is_best": false
        },
        // ... 더 많은 결과
    ],
    "message": "최적값 탐색 완료: eta=0.010000, lambda=0.000100, accuracy=95.50%"
}
```

---

## 🔧 REST API 엔드포인트

### 1. 기본 정보

#### `GET /`
서버 상태 확인

**Request:** 없음

**Response:**
```javascript
{
    "message": "Hello World"
}
```

---

### 2. 설정 관리

#### `GET /network/best-config`
best_config.json에 저장된 최적의 하이퍼파라미터 조회

**Request:** 없음

**Response:**
```javascript
{
    "eta": 0.0008127853044787373,
    "l2_lambda": 4.222795013060559e-06,
    "epochs": 30,
    "mini_batch_size": 32
}
```

**Error Response:**
```javascript
{
    "error": "설정 파일을 불러올 수 없습니다: ..."
}
```

---

### 3. 모델 예측

#### `POST /network/predict`
이미지 데이터로 숫자 예측 (학습 중에는 사용 불가)

**Request:**
```javascript
{
    "image": [0.0, 0.1, 0.2, ..., 0.9]  // 784개의 float 배열 (28x28 이미지)
}
```

**Response:**
```javascript
{
    "result": 5  // 예측된 숫자 (0-9)
}
```

**Error Response:**
```javascript
{
    "error": "모델이 현재 학습 중입니다. 학습이 완료된 후 다시 시도하세요."
}
```

---

### 4. 모델 테스트

#### `GET /network/test`
테스트 데이터셋으로 모델 평가

**Request:** 없음

**Response:**
```javascript
{
    "accuracy": 9550,           // 정답 개수
    "total": 10000,             // 전체 테스트 개수
    "percentage": 95.5,         // 정확도 (%)
    "message": "정확도: 95.50%"
}
```

**Error Response:**
```javascript
{
    "error": "모델이 현재 학습 중입니다."
}
```

---

### 5. 모델 학습

#### `POST /network/train`
모델 학습 시작 (백그라운드에서 실행)

**Request:**
```javascript
{
    "eta": 0.01,              // 학습률 (null이면 best_config에서 로드)
    "l2_lambda": 0.0001,      // 람다 값 (null이면 best_config에서 로드)
    "save_model": true        // 학습 후 모델 저장 여부
}
```

**Response:**
```javascript
{
    "message": "학습이 백그라운드에서 시작되었습니다.",
    "eta": 0.01,
    "l2_lambda": 0.0001,
    "save_model": true,
    "info": "None 값은 best_config.json에서 로드됩니다."
}
```

**Error Response:**
```javascript
{
    "error": "이미 학습이 진행 중입니다.",
    "status": {
        "is_training": true,
        "message": "학습이 진행 중입니다..."
    }
}
```

**학습 진행상황은 WebSocket으로 실시간 수신됩니다.**

---

#### `GET /network/train/status`
현재 학습 상태 확인

**Request:** 없음

**Response:**
```javascript
{
    "is_training": false,
    "message": "학습 완료 및 모델 저장됨 (자동 재로드 완료)"
}
```

---

#### `POST /network/train/stop`
진행 중인 학습 중지

**Request:** 없음

**Response:**
```javascript
{
    "message": "학습 중지 요청이 전송되었습니다."
}
```

**Error Response:**
```javascript
{
    "error": "현재 진행 중인 학습이 없습니다."
}
```

---

### 6. 모델 재로드

#### `POST /network/reload`
저장된 모델을 메모리에 다시 로드

**Request:** 없음

**Response:**
```javascript
{
    "message": "모델이 성공적으로 재로드되었습니다."
}
```

**Error Response:**
```javascript
{
    "error": "모델이 현재 학습 중입니다."
}
// 또는
{
    "error": "모델 로드 실패: ..."
}
```

---

### 7. 하이퍼파라미터 최적화

#### `POST /network/optimize`
최적의 학습률과 람다 값 탐색 (매우 오래 걸림, 백그라운드 실행)

**Request:**
```javascript
{
    "initial_etas": [0.009, 0.01, 0.02, 0.03, 0.05, 0.06],    // 학습률 후보군
    "initial_lambdas": [0.00001, 0.0001, 0.001],               // 람다 후보군
    "iterations": 5                                            // 반복 횟수 (1-10)
}
```

**Response:**
```javascript
{
    "message": "최적값 탐색이 백그라운드에서 시작되었습니다.",
    "warning": "이 작업은 상당한 시간과 자원을 소모할 수 있습니다.",
    "initial_etas": [0.009, 0.01, 0.02, 0.03, 0.05, 0.06],
    "initial_lambdas": [0.00001, 0.0001, 0.001],
    "iterations": 5
}
```

**Error Response:**
```javascript
{
    "error": "이미 최적화가 진행 중입니다.",
    "status": {
        "is_optimizing": true,
        "message": "최적값 탐색이 진행 중입니다..."
    }
}
// 또는
{
    "error": "모델이 현재 학습 중입니다. 학습이 완료된 후 시도하세요."
}
```

**최적화 진행상황과 결과는 WebSocket으로 실시간 수신됩니다.**

---

#### `GET /network/optimize/status`
최적화 진행 상태 확인

**Request:** 없음

**Response:**
```javascript
{
    "is_optimizing": false,
    "message": "최적값 탐색 완료: eta=0.010000, lambda=0.000100, accuracy=95.50%"
}
```

---

## 🎯 사용 시나리오

### 시나리오 1: 모델 학습

```javascript
// 1. WebSocket 연결
const ws = new WebSocket('ws://localhost:8000/ws/training');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.epoch) {
        // 학습 진행 중
        console.log(`Epoch ${data.epoch}/${data.total_epochs}`);
        console.log(`Loss: ${data.loss}, Accuracy: ${data.accuracy_percentage}%`);
        // 그래프 업데이트
    }
    
    if (data.status === 'completed') {
        console.log('학습 완료!');
    }
};

// 2. 학습 시작
fetch('http://localhost:8000/network/train', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        eta: null,           // best_config 값 사용
        l2_lambda: null,     // best_config 값 사용
        save_model: true
    })
});

// 3. 필요시 중지
// fetch('http://localhost:8000/network/train/stop', { method: 'POST' });
```

---

### 시나리오 2: 모델 테스트

```javascript
// 1. 모델 테스트
const response = await fetch('http://localhost:8000/network/test');
const result = await response.json();

console.log(`정확도: ${result.percentage}%`);
console.log(`정답: ${result.accuracy}/${result.total}`);
```

---

### 시나리오 3: 최적값 찾기

```javascript
// 1. WebSocket 연결 (시나리오 1과 동일)
const ws = new WebSocket('ws://localhost:8000/ws/training');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.optimization && data.optimization_result) {
        // 각 조합의 결과를 실시간으로 받음
        const result = data.optimization_result;
        console.log(`Iteration ${result.iteration}: eta=${result.eta}, lambda=${result.lambda} => ${result.percentage}%`);
        // 실시간 그래프 업데이트
    }
    
    if (data.optimization && data.status === 'completed') {
        // 최적화 완료
        console.log(`최적값: eta=${data.best_eta}, lambda=${data.best_lambda}`);
        console.log(`전체 결과:`, data.all_results);
        // 최종 그래프 그리기
    }
};

// 2. 최적화 시작
fetch('http://localhost:8000/network/optimize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        initial_etas: [0.009, 0.01, 0.02, 0.03],
        initial_lambdas: [0.00001, 0.0001, 0.001],
        iterations: 3
    })
});

// 3. 상태 확인 (선택사항)
const status = await fetch('http://localhost:8000/network/optimize/status');
const statusData = await status.json();
console.log(statusData.message);
```

---

### 시나리오 4: 이미지 예측

```javascript
// 28x28 이미지를 784개의 float 배열로 변환
const imageData = [0.0, 0.1, 0.2, /* ... 784개 ... */];

const response = await fetch('http://localhost:8000/network/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageData })
});

const result = await response.json();
console.log(`예측 결과: ${result.result}`); // 0-9
```

---

## ⚠️ 주의사항

1. **동시 실행 제한**
   - 학습과 최적화는 동시에 실행할 수 없습니다
   - 학습/최적화 중에는 예측 API 사용 불가

2. **WebSocket 연결**
   - 학습/최적화 시작 전에 WebSocket을 먼저 연결하세요
   - 연결이 끊어지면 자동 재연결 로직 구현 권장

3. **학습 중 모델 사용**
   - 학습이 완료되면 자동으로 모델이 재로드됩니다
   - 서버 재시작 불필요

4. **최적화 시간**
   - iterations=5, etas=6개, lambdas=3개일 경우 매우 오래 걸립니다
   - 테스트 시에는 작은 값으로 시작하세요

5. **데이터 형식**
   - 이미지 데이터는 0.0 ~ 1.0 사이의 정규화된 값이어야 합니다
   - 배열 길이는 정확히 784개여야 합니다

---

## 📊 전역 변수 (프론트엔드)

최적화 결과는 다음 전역 변수에 저장됩니다:

### `window.optimizationLiveData`
실시간 진행 데이터 (최적화 진행 중)

```javascript
{
    current_iteration: 1,
    total_iterations: 5,
    results_by_iteration: {
        1: [ /* 결과 배열 */ ],
        2: [ /* 결과 배열 */ ]
    },
    current_best: {
        eta: 0.01,
        lambda: 0.0001,
        percentage: 95.5
    }
}
```

### `window.optimizationData`
최종 완료 데이터 (최적화 완료 후)

```javascript
{
    results: [ /* 모든 조합의 결과 배열 */ ],
    best_eta: 0.01,
    best_lambda: 0.0001,
    best_percentage: 95.5
}
```

---

## 🔗 관련 문서

- [최적화 데이터 구조 상세 가이드](./OPTIMIZATION_DATA.md)
- 그래프 구현 예시 및 데이터 활용 팁은 위 문서 참조

---

## 💡 팁

1. **best_config 활용**: 학습 시 `eta`와 `l2_lambda`를 `null`로 보내면 자동으로 최적값 사용
2. **로그 확인**: 브라우저 콘솔에서 `console.log(window.optimizationData)` 로 데이터 확인 가능
3. **에러 핸들링**: 모든 API는 에러 시 `error` 필드를 반환하므로 체크 필수
4. **상태 관리**: 학습/최적화 시작 전에 `/status` 엔드포인트로 현재 상태 확인 권장
