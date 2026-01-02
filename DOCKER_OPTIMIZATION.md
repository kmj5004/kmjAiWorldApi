# Docker 이미지 크기 최적화 가이드

현재 이미지 크기: **1.54GB** → 목표: **600MB 이하**

## 🎯 최적화 방법 요약

| 방법 | 예상 크기 | 설명 |
|------|----------|------|
| 기본 (현재) | 1.54GB | GPU 지원 PyTorch |
| 최적화 1 | ~800MB | CPU-only PyTorch |
| 최적화 2 | ~600MB | Multi-stage build + 버전 고정 |
| 최적화 3 | ~500MB | 추가 캐시 정리 |

---

## 🚀 빠른 적용 방법

### 방법 1: 개선된 Dockerfile 사용 (권장)

현재 `Dockerfile`이 이미 최적화되어 있습니다.

```bash
# 빌드
docker build -t kmj_legend_ai_world:optimized .

# 크기 확인
docker images kmj_legend_ai_world
```

**주요 개선사항:**
- ✅ Multi-stage build 적용
- ✅ CPU-only PyTorch 사용
- ✅ 불필요한 파일 제외 (.dockerignore)
- ✅ 비root 사용자 (보안 강화)

---

### 방법 2: 최고 수준 최적화 (최소 크기)

더 작은 이미지가 필요한 경우:

```bash
# Dockerfile.optimized 사용
docker build -f Dockerfile.optimized -t kmj_legend_ai_world:ultra-optimized .

# 크기 확인
docker images kmj_legend_ai_world:ultra-optimized
```

**추가 최적화:**
- 패키지 버전 고정으로 캐시 효율 증대
- Python 캐시 완전 제거
- 헬스체크 최적화

---

## 📊 크기 절감 분석

### 주요 용량 차지 항목

```
PyTorch (GPU):        ~1.2GB
PyTorch (CPU):        ~700MB
FastAPI + 의존성:     ~50MB
애플리케이션 코드:     ~5MB
학습된 모델:          ~3MB
베이스 이미지:        ~130MB
```

### 최적화 효과

1. **GPU → CPU PyTorch**: -500MB
2. **Multi-stage build**: -100MB
3. **불필요한 파일 제외**: -66MB (data + 기타)
4. **캐시 정리**: -50MB

**총 절감: ~700MB (1.54GB → ~840MB)**

---

## 🔧 적용된 최적화 기법

### 1. CPU-only PyTorch 사용

GPU가 필요없다면 CPU 버전 사용:

```dockerfile
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
```

**절감 효과: -500MB**

### 2. Multi-stage Build

빌드 도구를 최종 이미지에서 제외:

```dockerfile
# 빌드 단계
FROM python:3.11-slim as builder
RUN apt-get install gcc g++
RUN pip install ...

# 최종 단계 (빌드 도구 없음)
FROM python:3.11-slim
COPY --from=builder /usr/local/lib/python3.11/site-packages ...
```

**절감 효과: -100MB**

### 3. .dockerignore 최적화

불필요한 파일 제외:

```
data/          # 63MB
train/         # 16KB
*.md           # 문서
test/          # 테스트 파일
```

**절감 효과: -66MB**

### 4. 캐시 제거

```dockerfile
RUN pip install --no-cache-dir ...
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN find . -type f -name '*.pyc' -delete
```

**절감 효과: -50MB**

---

## 🛠️ 추가 최적화 옵션

### 옵션 1: Alpine Linux 사용 (고급)

더 작은 베이스 이미지를 원하면:

```dockerfile
FROM python:3.11-alpine

# 하지만 빌드 시간이 길고 호환성 문제 가능
RUN apk add --no-cache gcc musl-dev linux-headers
```

**장점:** -50MB
**단점:** 빌드 시간 증가, 호환성 문제 가능

### 옵션 2: 모델을 별도 볼륨으로

학습된 모델을 이미지에 포함하지 않고 마운트:

```bash
docker run -v ./trained_data:/app/trained_data kmj_legend_ai_world
```

**절감 효과: -3MB** (미미하지만 모델이 크면 유용)

### 옵션 3: 특정 패키지만 설치

사용하지 않는 패키지 제거:

```txt
# requirements.txt
fastapi
uvicorn[standard]  # uvicorn만 필요
torch              # scipy, matplotlib 제거 가능?
```

---

## 📝 빌드 및 배포

### 최적화된 이미지 빌드

```bash
# 방법 1: 기본 최적화 (권장)
docker build -t kmj_legend_ai_world:v2 .

# 방법 2: 최고 최적화
docker build -f Dockerfile.optimized -t kmj_legend_ai_world:v2-ultra .

# 빌드 캐시 사용 안함 (클린 빌드)
docker build --no-cache -t kmj_legend_ai_world:v2 .
```

### 크기 확인

```bash
# 이미지 목록 및 크기
docker images

# 특정 이미지 상세 정보
docker inspect kmj_legend_ai_world:v2 | grep Size

# 레이어별 크기 확인
docker history kmj_legend_ai_world:v2
```

### 실행

```bash
# 기본 실행
docker run -p 8000:80 kmj_legend_ai_world:v2

# 볼륨 마운트 (모델 외부화)
docker run -p 8000:80 \
  -v ./trained_data:/app/trained_data \
  kmj_legend_ai_world:v2
```

---

## ✅ 검증

### 1. 크기 확인

```bash
docker images | grep kmj_legend_ai_world
```

**목표:** 1.54GB → 600-800MB

### 2. 동작 확인

```bash
# 컨테이너 실행
docker run -d -p 8000:80 --name test kmj_legend_ai_world:v2

# API 테스트
curl http://localhost:8000/

# 로그 확인
docker logs test

# 정리
docker stop test && docker rm test
```

### 3. 레이어 분석

```bash
# 어떤 레이어가 큰지 확인
docker history kmj_legend_ai_world:v2 --no-trunc

# 또는 dive 도구 사용 (설치 필요)
# brew install dive
dive kmj_legend_ai_world:v2
```

---

## 🎯 권장 설정

프로덕션 환경에 따른 권장 설정:

### 개발 환경
- 파일: `Dockerfile` (기본)
- 예상 크기: ~800MB
- GPU 지원 필요시 원본 유지

### 프로덕션 (CPU만 필요)
- 파일: `Dockerfile` (현재 최적화됨)
- 예상 크기: ~800MB
- CPU-only PyTorch 사용

### 엣지/IoT 환경
- 파일: `Dockerfile.optimized`
- 예상 크기: ~600MB
- 모든 최적화 적용

---

## 🚨 주의사항

1. **PyTorch CPU vs GPU**
   - GPU가 필요하면 원본 PyTorch 사용 필요 (크기 증가)
   - CPU만으로 충분하면 CPU 버전 권장

2. **의존성 버전**
   - 버전을 고정하면 재현 가능하지만 보안 패치 누락 가능
   - 주기적으로 업데이트 필요

3. **빌드 시간**
   - 최적화된 이미지는 빌드 시간이 더 오래 걸릴 수 있음
   - CI/CD에서 캐시 활용 권장

4. **호환성**
   - Alpine 사용 시 일부 Python 패키지 호환성 문제 가능
   - slim 이미지 권장

---

## 📈 개선 결과 예상

| 구분 | 이전 | 이후 | 절감 |
|------|------|------|------|
| 이미지 크기 | 1.54GB | ~800MB | 48% |
| 빌드 시간 | 5분 | 7분 | +40% |
| 메모리 사용 | 1.5GB | 1.2GB | 20% |
| 배포 시간 | 3분 | 1.5분 | 50% |

---

## 🔗 추가 참고자료

- [Docker 공식 최적화 가이드](https://docs.docker.com/develop/dev-best-practices/)
- [PyTorch CPU-only 설치](https://pytorch.org/get-started/locally/)
- [Multi-stage builds](https://docs.docker.com/build/building/multi-stage/)

---

## 💡 다음 단계

1. ✅ `.dockerignore` 업데이트 완료
2. ✅ `Dockerfile` 최적화 완료
3. ✅ `Dockerfile.optimized` 생성 완료
4. ⏳ 빌드 및 테스트
5. ⏳ ECR에 푸시

```bash
# 테스트 빌드
docker build -t kmj_legend_ai_world:v2-test .

# 크기 확인
docker images kmj_legend_ai_world:v2-test
```
