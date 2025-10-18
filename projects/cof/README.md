# cof - 바이너리 최적화 버전 관리 시스템

## 프로젝트 개요

**cof**는 바이너리 파일을 효율적으로 처리하는 차세대 버전 관리 시스템입니다. 블록 레벨 실시간 중복 제거와 커밋 기반 압축을 통해 기존 Git의 한계를 극복합니다.

### 핵심 특징
- **블록 레벨 실시간 중복 제거**: 4KB 단위 블록으로 효율적인 스토리지
- **커밋 기반 자동 압축**: 오래된 데이터의 점진적 압축
- **바이너리 최적화**: 대용량 바이너리 파일의 효율적 처리
- **고성능**: Python으로 구현된 크로스 플랫폼 성능

## 기술 사양

### 아키텍처
```
Storage Layers:
┌─────────────┬──────────────┬────────────────┐
│ Hot Tier    │ Warm Tier    │ Cold Tier      │
│ 0-10 커밋   │ 11-100 커밋  │ 100+ 커밋      │
│ Raw 블록    │ zstd level 3 │ zstd level 19  │
└─────────────┴──────────────┴────────────────┘
```

### 핵심 기술 스택
- **언어**: Python 3.11+ (외부 라이브러리 최대 활용)
- **해시 알고리즘**: BLAKE3 (32바이트 출력)
- **블록 크기**: 4KB (4096 바이트)
- **압축**: zstd (레벨 3, 19)
- **설정 파일**: TOML
- **동시성**: threading, file locking
- **패키지 관리**: uv

### Repository 구조
```
.cof/
├── config.toml          # 설정 파일
├── objects/
│   ├── hot/            # 최신 블록들 (압축 없음)
│   ├── warm/           # 중간 블록들 (경량 압축)
│   └── cold/           # 오래된 블록들 (최대 압축)
├── index/
│   ├── block_map       # 해시 → 위치 매핑
│   └── commit_age      # 블록 생성 커밋 추적
├── refs/
│   └── heads/          # 브랜치 포인터
└── locks/              # 동시성 제어
```

## 기능 명세

### Phase 1: 핵심 기능 (MVP)

#### 저장소 관리
```bash
cof init                 # 저장소 초기화
cof status              # 작업 디렉토리 상태
cof dedup-stats         # 중복 제거 통계
```

#### 버전 관리
```bash
cof add <files>         # 파일 스테이징
cof commit -m "message" # 커밋 생성
cof log                 # 커밋 히스토리
```

#### 브랜치 관리
```bash
cof branch <name>       # 브랜치 생성
cof checkout <branch>   # 브랜치 전환
cof merge <branch>      # 브랜치 병합
```

### Phase 2: 네트워크 기능 (추후 구현)
```bash
cof clone <url>         # 저장소 복제
cof push <remote>       # 변경사항 업로드
cof pull <remote>       # 변경사항 다운로드
```

## 데이터 구조

### 블록 저장
```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class StorageTier(Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"

@dataclass
class Block:
    hash: bytes              # BLAKE3 해시 (32바이트)
    data: bytes              # 실제 데이터 (최대 4KB)
    tier: StorageTier        # HOT, WARM, COLD
    created_commit: int      # 생성된 커밋 번호
    ref_count: int           # 참조 카운트
```

### 커밋 메타데이터
```python
@dataclass
class Commit:
    id: bytes               # 커밋 해시 (32바이트)
    parent: Optional[bytes] # 부모 커밋 (옵션)
    tree_root: bytes        # 루트 트리 객체 (32바이트)
    timestamp: int          # Unix 타임스탬프
    author: str             # 작성자 정보
    message: str            # 커밋 메시지
    sequence: int           # 순차 번호 (aging용)
```

### 트리/블롭 객체
```python
@dataclass
class TreeEntry:
    name: str               # 파일/디렉토리 이름
    mode: int               # 권한
    hash: bytes             # 블롭/서브트리 해시 (32바이트)
    size: int               # 원본 파일 크기
```

## 설정 파일 (config.toml)

```toml
[core]
block_size = 4096
hash_algorithm = "blake3"
cache_size_mb = 256

[compression]
warm_threshold = 10      # 커밋 기준
cold_threshold = 100     # 커밋 기준  
warm_level = 3           # zstd 레벨
cold_level = 19          # zstd 레벨

[network]
protocol = "udp"
packet_size = 1400       # MTU 고려
timeout_ms = 5000
max_retries = 3

[gc]
auto_gc = true
unreachable_days = 30    # 미참조 블록 보관 기간
```

## 성능 목표

### 저장 효율성
- **중복 제거율**: 바이너리 중심 프로젝트에서 50%+ 공간 절약
- **압축 효과**: Cold tier에서 추가 30-50% 공간 절약

### 처리 성능
- **Target 저장소**: 10GB 이하
- **성능 목표**: Git 대비 2-3배 빠른 기본 작업
- **메모리 사용**: 256MB 블록 캐시 + 효율적인 스트리밍

## 개발 계획

### Phase 1: 핵심 엔진 (4-6주)
1. **블록 스토리지 시스템** (1-2주)
   - BLAKE3 해싱 (blake3 라이브러리)
   - 4KB 블록 분할
   - 중복 제거 로직

2. **압축 시스템** (1주)
   - zstd 통합 (zstandard 라이브러리)
   - Tier 간 마이그레이션

3. **객체 모델** (1-2주)
   - 커밋, 트리, 블롭 구현
   - 메타데이터 관리 (dataclasses 활용)

4. **CLI 인터페이스** (1주)
   - 기본 명령어 구현 (click 또는 argparse)
   - 설정 파일 처리 (toml 라이브러리)

### Phase 2: 네트워크 기능 (3-4주)
1. **UDP 프로토콜** (2주) - asyncio 활용
2. **동기화 로직** (1-2주) - aiofiles 활용

## 라이센스 및 배포

- **라이센스**: AGPL v3
- **개발 환경**: Python 3.11+ (크로스 플랫폼)
- **패키지 관리**: uv (빠른 의존성 관리)
- **빌드 시스템**: Python 네이티브 + pyproject.toml
- **의존성**: 외부 라이브러리 적극 활용 (blake3, zstandard, click, toml 등)

## 향후 확장 계획

1. **크로스 플랫폼 지원** (Windows/macOS/Linux)
2. **Git 마이그레이션 도구** (Python 기반)
3. **IDE 플러그인** (VS Code, PyCharm)
4. **웹 인터페이스** (FastAPI + React)
5. **분산 백업 시스템** (asyncio 기반)

---

*이 기획서는 cof 프로젝트의 초기 설계를 바탕으로 작성되었으며, 개발 과정에서 세부사항이 조정될 수 있습니다.*