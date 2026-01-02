# 최적화 데이터 구조 설명

최적값 탐색 기능이 실행되면 WebSocket을 통해 실시간으로 데이터가 전송됩니다.
이 데이터를 사용하여 그래프를 그릴 수 있습니다.

## 실시간 데이터 (`window.optimizationLiveData`)

최적화가 진행되는 동안 각 조합의 결과가 실시간으로 업데이트됩니다.

```javascript
window.optimizationLiveData = {
    current_iteration: 1,              // 현재 반복 횟수
    total_iterations: 5,                // 전체 반복 횟수
    results_by_iteration: {             // iteration별 결과
        1: [                            // 1차 반복 결과들
            {
                iteration: 1,
                eta: 0.009,
                lambda: 0.00001,
                accuracy: 9500,
                percentage: 95.0,
                total: 10000,
                is_best: false
            },
            // ... 더 많은 조합들
        ],
        2: [...],                       // 2차 반복 결과들
        // ...
    },
    current_best: {                     // 현재까지의 최고 결과
        eta: 0.01,
        lambda: 0.0001,
        percentage: 95.5
    }
}
```

## 최종 완료 데이터 (`window.optimizationData`)

모든 최적화가 완료되면 전체 결과와 최적값이 제공됩니다.

```javascript
window.optimizationData = {
    results: [                          // 모든 조합의 결과 배열
        {
            iteration: 1,               // 몇 번째 반복에서 테스트했는지
            eta: 0.009,                 // 테스트한 학습률
            lambda: 0.00001,            // 테스트한 람다 값
            accuracy: 9500,             // 정답 개수
            percentage: 95.0,           // 정확도 (%)
            total: 10000,               // 전체 테스트 개수
            is_best: false              // 최적값 여부
        },
        {
            iteration: 1,
            eta: 0.01,
            lambda: 0.00001,
            accuracy: 9550,
            percentage: 95.5,
            total: 10000,
            is_best: true               // 이 조합이 최적!
        },
        // ... 더 많은 결과들
    ],
    best_eta: 0.01,                     // 최적 학습률
    best_lambda: 0.00001,               // 최적 람다 값
    best_percentage: 95.5               // 최고 정확도
}
```

## 그래프 구현 예시

### 1. Iteration별 결과 비교 (산점도)

각 iteration에서 테스트한 조합들을 다른 색상으로 표시:

```javascript
// 최적화 완료 후 호출
function drawOptimizationGraph() {
    const data = window.optimizationData;
    if (!data) return;
    
    // iteration별로 색상 구분
    const colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF'];
    
    const datasets = [];
    
    // iteration별로 데이터셋 생성
    for (let i = 1; i <= 5; i++) {
        const iterationResults = data.results.filter(r => r.iteration === i);
        
        datasets.push({
            label: `반복 ${i}`,
            data: iterationResults.map(r => ({
                x: r.eta,
                y: r.lambda,
                percentage: r.percentage
            })),
            backgroundColor: colors[i - 1],
            borderColor: colors[i - 1],
            pointRadius: 6,
            pointHoverRadius: 8
        });
    }
    
    // 최적값 표시
    datasets.push({
        label: '최적값',
        data: [{
            x: data.best_eta,
            y: data.best_lambda,
            percentage: data.best_percentage
        }],
        backgroundColor: '#FF0000',
        borderColor: '#FF0000',
        pointRadius: 12,
        pointStyle: 'star'
    });
    
    // Chart.js로 그리기
    // const ctx = document.getElementById('optimizationChart').getContext('2d');
    // new Chart(ctx, { ... });
}
```

### 2. 실시간 업데이트 그래프

진행 중인 데이터를 실시간으로 표시:

```javascript
// WebSocket 메시지 수신 시 호출
function updateOptimizationGraph() {
    const liveData = window.optimizationLiveData;
    if (!liveData) return;
    
    // 현재까지의 모든 결과 수집
    const allResults = [];
    for (let iteration in liveData.results_by_iteration) {
        allResults.push(...liveData.results_by_iteration[iteration]);
    }
    
    // iteration별로 색상 구분하여 실시간 업데이트
    // ...
}
```

### 3. 정확도 히트맵

ETA와 Lambda의 조합에 따른 정확도를 색상으로 표시:

```javascript
function drawHeatmap() {
    const data = window.optimizationData;
    if (!data) return;
    
    // ETA와 Lambda를 축으로 하는 히트맵
    const heatmapData = data.results.map(r => ({
        x: r.eta,
        y: r.lambda,
        v: r.percentage  // value (정확도)
    }));
    
    // 정확도가 높을수록 진한 색으로 표시
    // ...
}
```

## 데이터 활용 팁

1. **색상 구분**: 각 iteration을 다른 색상으로 표시하여 탐색 범위가 좁혀지는 과정을 시각화
2. **크기 구분**: 정확도에 따라 점의 크기를 다르게 하여 좋은 조합 강조
3. **애니메이션**: 실시간 데이터를 사용하여 탐색 과정을 애니메이션으로 표현
4. **3D 그래프**: ETA, Lambda, Accuracy를 3차원 그래프로 표현

## WebSocket 이벤트 처리

```javascript
// 최적화 진행 중
if (data.optimization && data.optimization_result) {
    // 새로운 결과가 도착할 때마다 그래프 업데이트
    updateOptimizationGraph();
}

// 최적화 완료
if (data.optimization && data.status === 'completed') {
    // 최종 그래프 그리기
    drawOptimizationGraph();
}
```

## 콘솔에서 데이터 확인

브라우저 콘솔에서 언제든지 데이터를 확인할 수 있습니다:

```javascript
// 실시간 데이터 확인
console.log(window.optimizationLiveData);

// 최종 결과 확인
console.log(window.optimizationData);

// 특정 iteration 결과만 확인
console.log(window.optimizationLiveData.results_by_iteration[1]);
```
