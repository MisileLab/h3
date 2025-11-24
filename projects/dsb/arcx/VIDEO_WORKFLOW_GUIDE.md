# 비디오 기반 YOLO 학습 워크플로우 가이드

이 가이드는 게임플레이 비디오에서 프레임을 추출하고 YOLO 모델을 학습하는 전체 과정을 설명합니다.

## 개요

```
게임 녹화 영상
    ↓
프레임 추출 (1 FPS)
    ↓
Extraction 화면 필터링 (대화형)
    ↓
아이템 Annotation
    ↓
YOLO11 학습
    ↓
모델 배포
```

## 준비물

1. **게임플레이 비디오**
   - 포맷: MP4, AVI, MOV, MKV 등
   - 해상도: 1080p 이상 권장
   - 길이: 10-30분 권장
   - 내용: Extraction 화면이 포함된 게임플레이

2. **저장 공간**
   - 비디오당 약 1-2GB
   - 추출된 프레임: 1분당 약 10-20MB (1 FPS)

## 상세 워크플로우

### 1. 게임플레이 녹화

#### OBS Studio 설정 (권장)

```
해상도: 1920x1080
프레임레이트: 30 또는 60 FPS
인코더: x264 또는 NVENC (NVIDIA)
품질: High Quality, Medium File Size
포맷: MP4
```

**녹화 팁:**
- Extraction 화면이 보이는 순간을 여러 번 녹화
- 다양한 아이템 조합 포함
- 다양한 레어리티 아이템 포함
- UI가 명확하게 보이도록

### 2. 프레임 추출

#### 2-1. 단일 비디오 처리

```bash
# 기본 추출 (1 FPS)
just yolo-extract-video gameplay.mp4 frames fps=1

# 특정 구간만 추출 (60초~180초)
just yolo-extract-video gameplay.mp4 frames fps=1 start=60 end=180

# 더 많은 프레임 (2 FPS)
just yolo-extract-video gameplay.mp4 frames fps=2
```

#### 2-2. 여러 비디오 일괄 처리

```bash
# videos/ 디렉토리의 모든 비디오 처리
just yolo-extract-videos videos/ extracted_frames fps=1
```

디렉토리 구조:
```
videos/
├── gameplay1.mp4
├── gameplay2.mp4
└── gameplay3.mp4

extracted_frames/
├── gameplay1/
│   ├── gameplay1_000000.png
│   ├── gameplay1_000001.png
│   └── ...
├── gameplay2/
│   └── ...
└── gameplay3/
    └── ...
```

#### 2-3. 고급 옵션

```bash
cd backend
uv run python scripts/yolo/extract_video_frames.py \
    --video gameplay.mp4 \
    --output frames \
    --fps 1 \
    --start 60 \
    --end 300 \
    --max-frames 500 \
    --resize 1920x1080 \
    --quality 95
```

**옵션 설명:**
- `--fps`: 초당 추출 프레임 수 (기본: 원본 FPS)
- `--interval`: 프레임 추출 간격 (초) - fps 대신 사용 가능
- `--start`: 시작 시간 (초)
- `--end`: 종료 시간 (초)
- `--max-frames`: 최대 프레임 수
- `--prefix`: 파일명 prefix (기본: "frame")
- `--quality`: 이미지 품질 0-100 (기본: 95)
- `--resize`: 리사이즈 (WIDTHxHEIGHT)

**권장 설정:**
- 일반적인 경우: `--fps 1` (1초당 1프레임)
- 빠른 게임플레이: `--fps 2` (1초당 2프레임)
- 긴 비디오: `--interval 5` (5초마다 1프레임)

### 3. Extraction 화면 필터링

추출된 수백~수천 개의 프레임 중에서 실제로 필요한 extraction 화면만 골라냅니다.

#### 3-1. 대화형 필터링 (권장)

```bash
just yolo-filter-frames frames/ filtered_frames/
```

**단축키:**
- `SPACE`: 현재 프레임 선택하고 저장
- `S`: 현재 프레임 스킵
- `Q`: 종료
- `A`: 다음 10개 프레임 자동 선택 (빠른 선택용)

**팁:**
- Extraction 화면이 보이면 `SPACE` 또는 `A`
- 일반 게임플레이 화면은 `S`로 스킵
- 유사한 화면이 연속으로 나오면 `A`로 10개 선택

#### 3-2. 템플릿 기반 자동 필터링

먼저 대표적인 extraction 화면 1장을 `template.png`로 저장한 후:

```bash
just yolo-filter-auto frames/ filtered_frames/ template.png
```

이 방법은 유사한 UI 구조를 가진 화면을 자동으로 찾습니다.

**임계값 조정:**
```bash
cd backend
uv run python scripts/yolo/filter_extraction_frames.py \
    --input frames \
    --output filtered \
    --mode auto \
    --template template.png \
    --threshold 0.8  # 0.0~1.0, 높을수록 엄격
```

### 4. 아이템 Annotation

필터링된 프레임에 아이템을 표시하고 라벨링합니다.

```bash
just yolo-annotate filtered_frames/
```

#### 4-1. Annotation 방법

1. **마우스로 바운딩 박스 그리기**
   - 왼쪽 클릭 & 드래그로 사각형 그리기
   - 아이템 전체를 포함하도록

2. **아이템 타입 입력**
   ```
   Item types: weapon, armor, material, consumable, mod, currency
   Item type: weapon
   ```

3. **레어리티 입력**
   ```
   Rarities: legendary, epic, rare, uncommon, common
   Rarity: epic
   ```

4. **다음 아이템 표시**
   - 같은 화면에 여러 아이템이 있으면 반복
   - 모든 아이템 표시 완료 후 `s` 키로 저장

#### 4-2. Annotation 단축키

- `u`: 마지막 박스 취소
- `s`: 저장하고 다음 이미지
- `n`: 저장하지 않고 다음 이미지 (스킵)
- `q`: 종료

#### 4-3. Annotation 팁

**정확한 바운딩 박스:**
- 아이템 전체를 포함 (너무 타이트하지 않게)
- 아이템 아이콘 + 이름 텍스트 포함
- 레어리티 색상 테두리 포함

**일관성:**
- 같은 타입의 아이템은 항상 같은 타입으로 라벨링
- 레어리티는 게임 내 표시 기준으로 정확히

**품질 vs 속도:**
- 최소 500-1000개 아이템 annotation 권장
- 정확도가 중요: 애매한 것은 스킵

### 5. 데이터셋 준비

Annotation이 완료되면 YOLO 형식으로 변환합니다.

```bash
just yolo-prepare annotations.json filtered_frames/
```

이 명령은:
1. Annotation을 YOLO 형식으로 변환
2. 학습/검증/테스트 세트로 분할 (70/20/10)
3. `data/yolo_items/` 디렉토리에 저장

**생성되는 구조:**
```
data/yolo_items/
├── dataset.yaml         # YOLO 설정 파일
├── images/
│   ├── train/          # 학습 이미지
│   ├── val/            # 검증 이미지
│   └── test/           # 테스트 이미지
└── labels/
    ├── train/          # 학습 라벨
    ├── val/            # 검증 라벨
    └── test/           # 테스트 라벨
```

### 6. YOLO11 학습

```bash
# 기본 학습 (100 epochs, batch 16)
just yolo-train

# 커스텀 설정
just yolo-train data_dir=data/yolo_items model=yolo11s.pt epochs=150 batch=32

# GPU 지정
just yolo-train-device data/yolo_items 0 yolo11n.pt 100
```

**모델 선택:**
- `yolo11n.pt`: Nano - 빠르지만 낮은 정확도 (시작용)
- `yolo11s.pt`: Small - 균형잡힌 선택 (권장)
- `yolo11m.pt`: Medium - 더 높은 정확도
- `yolo11l.pt`: Large - 최고 정확도, 느림
- `yolo11x.pt`: Extra Large - 프로덕션용

**학습 설정 권장:**
```bash
# 첫 학습 (빠른 테스트)
just yolo-train data_dir=data/yolo_items model=yolo11n.pt epochs=50 batch=16

# 본격 학습
just yolo-train data_dir=data/yolo_items model=yolo11s.pt epochs=150 batch=32

# 최종 모델
just yolo-train data_dir=data/yolo_items model=yolo11m.pt epochs=200 batch=16
```

**학습 시간 예상:**
- RTX 3060: yolo11s, 100 epochs ≈ 2-3시간
- RTX 4090: yolo11s, 100 epochs ≈ 30-45분

### 7. 모델 검증

학습이 완료되면 모델을 검증합니다.

```bash
just yolo-validate runs/yolo_train/item_detector/weights/best.safetensors
```

**좋은 결과:**
- mAP50: > 0.7
- mAP50-95: > 0.5
- Precision: > 0.8
- Recall: > 0.7

**개선이 필요한 경우:**
- 더 많은 데이터 수집
- 더 오래 학습 (epochs 증가)
- 더 큰 모델 사용
- Augmentation 조정

### 8. 모델 배포

```bash
# safetensors 모델을 프로덕션 위치로 복사
cp runs/yolo_train/item_detector/weights/best.safetensors \
   models/item_detector_yolo11.safetensors

# 또는 자동 변환 명령 사용
just yolo-convert-best
```

### 9. 테스트

```bash
# 단일 스크린샷 평가
just eval-screenshot test_image.png

# 예제 실행
just eval-examples
```

## 완전 자동 워크플로우

모든 단계를 한 번에 실행 (필터링은 대화형):

```bash
just yolo-video-workflow gameplay.mp4
```

이 명령은:
1. 프레임 추출 (1 FPS)
2. 대화형 필터링 (사용자 선택)
3. Annotation 도구 실행 (사용자 라벨링)
4. 데이터셋 준비
5. YOLO 학습

## 문제 해결

### 프레임이 너무 많음

```bash
# 추출 간격 늘리기
just yolo-extract-video gameplay.mp4 frames fps=0.5  # 2초마다

# 또는 최대 프레임 수 제한
cd backend
uv run python scripts/yolo/extract_video_frames.py \
    --video gameplay.mp4 \
    --output frames \
    --max-frames 300
```

### 필터링이 너무 느림

1. 템플릿 자동 필터링 사용
2. 또는 프레임 추출 시 간격 늘리기

### Annotation이 오래 걸림

1. 여러 세션으로 나누기 (이어서 작업 가능)
2. 중요한 화면만 정확히 annotation
3. 최소 500-1000개 아이템이면 충분

### 학습이 너무 느림

1. 작은 모델 사용 (`yolo11n.pt`)
2. Batch size 줄이기
3. Epochs 줄이기 (50-100)
4. GPU 업그레이드 고려

## 모범 사례

### 데이터 수집

- ✅ 다양한 맵에서 수집
- ✅ 다양한 아이템 조합
- ✅ 모든 레어리티 균등하게
- ✅ 최소 500-1000개 아이템
- ❌ 동일한 화면 반복 수집

### Annotation

- ✅ 일관된 기준 유지
- ✅ 정확한 바운딩 박스
- ✅ 애매한 경우 스킵
- ❌ 너무 타이트한 박스
- ❌ 부분적으로 가려진 아이템

### 학습

- ✅ 검증 손실 모니터링
- ✅ Overfitting 주의
- ✅ 여러 모델 실험
- ❌ 너무 긴 학습 (diminishing returns)

## 다음 단계

1. 더 많은 비디오 수집 및 처리
2. 모델 fine-tuning
3. 프로덕션 배포
4. 실전 테스트 및 피드백

자세한 내용은 `YOLO_INTEGRATION.md`를 참조하세요.
