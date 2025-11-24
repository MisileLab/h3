# YOLO Integration Guide

이 문서는 arcx 프로젝트에 통합된 YOLO11 기반 아이템 가치 자동 평가 시스템에 대한 가이드입니다.

## 개요

YOLO11 모델을 사용하여 extraction 화면에서 아이템을 자동으로 감지하고, 레어리티에 따라 가치를 계산하는 시스템입니다.

### 주요 기능

- **자동 아이템 감지**: YOLO11로 extraction 화면에서 아이템 자동 인식
- **레어리티 분류**: Epic, Rare, Uncommon, Common 등 자동 분류
- **가치 자동 계산**: 아이템 타입과 레어리티에 따른 가치 자동 산출
- **세트 보너스**: 아이템 조합에 따른 추가 가치 적용
- **게임 페이즈 조정**: 시즌 초/중/후반에 따른 가치 멀티플라이어
- **RL 훈련 통합**: 학습 데이터에 자동으로 가치 정보 포함

## 아키텍처

```
Extraction 화면
    ↓
YOLO11 감지
    ↓
아이템 분류 (타입 + 레어리티)
    ↓
가치 계산 + 보너스
    ↓
DataLogger → Parquet
    ↓
RL 훈련
```

## 설치

### 1. 의존성 설치

```bash
cd backend
uv pip install -e .
```

필수 패키지:
- `ultralytics >= 8.3.0` (YOLO11)
- `opencv-python >= 4.8.0`
- `safetensors >= 0.4.0`

### 2. 모델 준비

YOLO11 모델을 학습하거나 사전 학습된 모델을 사용:

```bash
# 학습된 모델 배치
cp your_trained_model.safetensors models/item_detector_yolo11.safetensors
```

## 사용 방법

### 1. 기본 사용 (EvaluationPipeline)

```python
from arcx.training import EvaluationPipeline
import cv2
import torch

# 파이프라인 생성 (자동 평가 활성화)
pipeline = EvaluationPipeline(
    auto_valuate=True,
    game_phase="mid_wipe"
)

# 런 시작
pipeline.start_run(run_id="raid_001", map_id="forest")

# 의사결정 로깅
for i in range(10):
    z_seq = torch.randn(32, 512)  # 인코더 출력
    pipeline.log_decision(
        decision_idx=i,
        t_sec=i * 5.0,
        action="stay" if i < 9 else "extract",
        z_seq=z_seq
    )

# Extraction 화면 스크린샷 로드
screenshot = cv2.imread("extraction_screen.png")

# 런 종료 (자동 평가)
pipeline.end_run_with_screenshot(
    screenshot=screenshot,
    total_time_sec=50.0,
    success=True
)

# 데이터 저장
pipeline.save()
```

### 2. 수동 평가

```python
from arcx.valuation import ItemValuator
import cv2

# Valuator 생성
valuator = ItemValuator(
    model_path="models/item_detector_yolo11.safetensors",
    confidence=0.5,
    device="cuda",
    game_phase="mid_wipe"
)

# 스크린샷 로드
screenshot = cv2.imread("extraction_screen.png")

# 평가
result = valuator.valuate_screenshot(screenshot)

print(f"총 가치: {result.total_value}")
print(f"아이템 수: {result.num_items}")
print(f"평균 confidence: {result.avg_confidence}")
print(f"타입별 가치: {result.value_breakdown}")
print(f"레어리티 카운트: {result.rarity_counts}")
```

### 3. DataLogger 직접 사용

```python
from arcx.data.logger import DataLogger
from arcx.valuation import ItemValuator
import cv2

logger = DataLogger()
valuator = ItemValuator()

# 런 시작
logger.start_run("run_001", "map_01")

# 의사결정 로깅
# ... log_decision() 호출들 ...

# 런 종료 (YOLO 평가 포함)
screenshot = cv2.imread("extraction.png")
logger.end_run(
    screenshot=screenshot,
    valuator=valuator,
    total_time_sec=120.0,
    success=True
)

logger.save()
```

## YOLO 모델 학습

### 방법 1: 비디오에서 프레임 추출 (권장)

게임플레이 비디오를 녹화한 후 프레임을 추출하는 방식입니다.

#### 1-1. 비디오 녹화

게임플레이 비디오를 녹화합니다 (OBS, GeForce Experience 등 사용).

#### 1-2. 프레임 추출

```bash
# 단일 비디오에서 프레임 추출 (1 FPS)
just yolo-extract-video gameplay.mp4 frames fps=1

# 여러 비디오 일괄 처리
just yolo-extract-videos videos/ extracted_frames fps=1

# 특정 구간만 추출
just yolo-extract-video gameplay.mp4 frames fps=1 start=60 end=180

# 고급 옵션
cd backend
uv run python scripts/yolo/extract_video_frames.py \
    --video gameplay.mp4 \
    --output frames \
    --fps 1 \
    --start 60 \
    --max-frames 500 \
    --resize 1920x1080
```

옵션:
- `--fps`: 추출 FPS (예: 1 = 초당 1프레임)
- `--interval`: 초 단위 간격 (예: 5 = 5초마다 1프레임)
- `--start`: 시작 시간 (초)
- `--end`: 종료 시간 (초)
- `--max-frames`: 최대 프레임 수
- `--resize`: 리사이즈 (WIDTHxHEIGHT)

#### 1-3. Extraction 화면 필터링 (선택사항)

추출된 프레임 중 extraction 화면만 골라냅니다:

```bash
# 대화형 필터링 (권장)
just yolo-filter-frames frames/ filtered_frames/
```

대화형 모드 단축키:
- `SPACE`: 선택하고 저장
- `S`: 스킵
- `Q`: 종료
- `A`: 다음 10개 자동 선택

```bash
# 템플릿 기반 자동 필터링
just yolo-filter-auto frames/ filtered_frames/ template.png

# 고급 옵션
cd backend
uv run python scripts/yolo/filter_extraction_frames.py \
    --input frames \
    --output filtered \
    --mode auto \
    --template extraction_template.png \
    --threshold 0.8
```

#### 1-4. 완전 자동 비디오 워크플로우

```bash
# 단일 비디오: 추출 -> 필터링 -> Annotation -> 학습
just yolo-video-workflow gameplay.mp4

# 여러 비디오: 추출만 수행
just yolo-videos-workflow videos/ extracted fps=1
# 이후 수동으로 필터링 및 annotation
```

### 방법 2: 스크린샷 직접 수집

Extraction 화면 스크린샷을 `screenshots/` 디렉토리에 직접 저장합니다.

### 2. Annotation

```bash
just yolo-annotate screenshots/
```

- 마우스로 바운딩 박스 그리기
- 아이템 타입과 레어리티 지정
- `annotations.json`에 저장

### 3. 데이터셋 준비

```bash
just yolo-prepare annotations.json screenshots/
```

YOLO 형식으로 변환:
- `data/yolo_items/images/train/`
- `data/yolo_items/labels/train/`
- `data/yolo_items/images/val/`
- `data/yolo_items/labels/val/`

### 4. 학습

```bash
# 기본 학습 (100 epochs)
just yolo-train

# 커스텀 설정
just yolo-train data_dir=data/yolo_items model=yolo11s.pt epochs=150 batch=32

# GPU 지정
just yolo-train-device data/yolo_items 0 yolo11n.pt 100
```

학습 중 자동으로 safetensors로 변환됩니다:
- `runs/yolo_train/item_detector/weights/best.pt`
- `runs/yolo_train/item_detector/weights/best.safetensors`

### 5. 모델 배포

```bash
cp runs/yolo_train/item_detector/weights/best.safetensors \
   models/item_detector_yolo11.safetensors
```

## 아이템 가치 설정

`backend/arcx/valuation/config.py`에서 아이템 가치 설정:

```python
ITEM_VALUE_MAP = {
    "weapon": {
        "legendary": 5000.0,
        "epic": 2500.0,
        "rare": 1000.0,
        "uncommon": 400.0,
        "common": 100.0,
    },
    "armor": {
        "legendary": 3000.0,
        "epic": 1500.0,
        # ...
    },
    # ...
}

# 게임 페이즈별 멀티플라이어
PHASE_MULTIPLIERS = {
    "early_wipe": 1.5,
    "mid_wipe": 1.0,
    "late_wipe": 0.7,
}

# 세트 보너스
SET_BONUSES = {
    "weapon_armor_combo": 1.1,
    "full_loadout": 1.2,
}
```

## API 사용

### POST /valuate

스크린샷을 평가합니다.

```bash
curl -X POST http://localhost:8765/valuate \
  -H "Content-Type: application/json" \
  -d '{
    "screenshot_base64": "<base64_encoded_image>",
    "game_phase": "mid_wipe"
  }'
```

Response:
```json
{
  "total_value": 4500.0,
  "num_items": 5,
  "avg_confidence": 0.87,
  "items": [
    {
      "item_type": "weapon",
      "rarity": "epic",
      "confidence": 0.92,
      "estimated_value": 2500.0,
      "bbox": [100, 200, 300, 400]
    },
    // ...
  ],
  "value_breakdown": {
    "weapon": 2500.0,
    "armor": 1500.0,
    "material": 500.0
  },
  "rarity_counts": {
    "epic": 2,
    "rare": 2,
    "uncommon": 1
  },
  "phase_multiplier": 1.0
}
```

### POST /run/end

런 종료 시 자동 평가 결과 포함:

```bash
curl -X POST http://localhost:8765/run/end \
  -H "Content-Type: application/json" \
  -d '{
    "run_id": "raid_001",
    "auto_valuation": {
      "total_value": 4500.0,
      "num_items": 5,
      "avg_confidence": 0.87,
      // ...
    },
    "total_time_sec": 180.0,
    "success": true,
    "action_taken": "extract"
  }'
```

## 데이터 스키마

Parquet 로그에 저장되는 필드:

```python
{
    "run_id": str,
    "decision_idx": int,
    "t_sec": float,
    "map_id": str,
    "action": str,
    "final_loot_value": float,  # YOLO 계산값
    "total_time_sec": float,
    "success": bool,
    "z_seq": List[float],

    # YOLO 메타데이터
    "num_items_detected": int,
    "avg_detection_confidence": float,
    "value_breakdown_json": str,  # JSON
    "rarity_counts_json": str,    # JSON
}
```

## Justfile 명령어

### YOLO 학습
```bash
just yolo-annotate <images_dir>          # Annotation
just yolo-prepare <annotations> <images>  # 데이터셋 준비
just yolo-train                           # 학습
just yolo-workflow <images_dir>           # 전체 워크플로우
```

### 테스트
```bash
just test-evaluation          # EvaluationPipeline 테스트
just test-extraction-detector # ExtractionDetector 테스트
just eval-examples            # 예제 실행
just eval-screenshot <path>   # 스크린샷 평가
```

### 유틸리티
```bash
just yolo-validate <model> <data_dir>  # 모델 검증
just yolo-test-image <model> <image>   # 이미지 테스트
just yolo-convert-safetensors <pt>     # .pt → .safetensors 변환
just clean-yolo                        # 학습 데이터 정리
```

## 고급 사용

### 커스텀 Extraction 감지

`ExtractionDetector`를 커스터마이즈하여 게임별 UI 감지:

```python
from arcx.capture import ExtractionDetector
import cv2

class CustomExtractionDetector(ExtractionDetector):
    def is_extraction_screen(self, frame):
        # 커스텀 감지 로직
        # 예: 템플릿 매칭, OCR, 색상 분석 등

        # "Extraction Success" 텍스트 감지
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ... OCR 로직 ...

        return detected
```

### 동적 가치 조정

런타임에 게임 페이즈 변경:

```python
pipeline = EvaluationPipeline(game_phase="early_wipe")

# 시즌 중반으로 변경
pipeline.valuator.set_game_phase("mid_wipe")
```

### 배치 평가

여러 스크린샷 일괄 처리:

```python
from pathlib import Path
import cv2

valuator = ItemValuator()
screenshots_dir = Path("screenshots")

results = []
for img_path in screenshots_dir.glob("*.png"):
    img = cv2.imread(str(img_path))
    result = valuator.valuate_screenshot(img)
    results.append({
        "filename": img_path.name,
        "value": result.total_value,
        "items": result.num_items
    })

# 결과 분석
import pandas as pd
df = pd.DataFrame(results)
print(df.describe())
```

## 문제 해결

### YOLO 모델이 로드되지 않음

```python
# 수동으로 모델 경로 지정
valuator = ItemValuator(
    model_path="path/to/your/model.safetensors",
    confidence=0.5
)
```

### 낮은 detection confidence

- Confidence 임계값 낮추기: `confidence=0.3`
- 더 많은 학습 데이터 수집
- 모델 재학습 (더 많은 epochs)
- 더 큰 모델 사용 (`yolo11s.pt`, `yolo11m.pt`)

### 잘못된 가치 계산

1. `config.py`에서 아이템 가치 재조정
2. 게임 페이즈 확인
3. YOLO 감지 결과 확인 (`result.items`)

## 예제

전체 예제는 `backend/examples/yolo_evaluation_example.py` 참조:

```bash
just eval-examples
```

## 참고

- YOLO11 문서: https://docs.ultralytics.com/models/yolo11/
- Safetensors: https://huggingface.co/docs/safetensors/
- 프로젝트 README: `README.md`
