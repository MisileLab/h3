# ArcX: EV Overlay System for Extraction Shooters

Real-time Expected Value (EV) overlay system for games like ARC Raiders, helping players decide whether to continue farming or extract based on ML-powered risk analysis.

## Architecture Overview

```
Game Screen → dxcam Capture → Ring Buffer → Frame Encoder → Latent Sequence
                                                                ↓
                                                         Temporal Q-Net
                                                                ↓
                                               Q(stay), Q(extract) Calculation
                                                                ↓
                                           FastAPI ←→ JSON ←→ Native Overlay
                                             ↓
                                     Parquet Logging (Training Data)

Extraction Screen → YOLO11 Item Detection → Value Calculation → Auto Loot Value
                         ↓                          ↓
                   Item Classification      Set Bonuses Applied
                   (Type + Rarity)                  ↓
                                              Data Logger Integration
```

## Features

### Core EV System
- **Real-time EV Calculation**: ML model computes expected values for "stay" vs "extract" actions
- **Multi-GPU Support**: Auto-detects CUDA, ROCm, DirectML, or falls back to CPU
- **Distributional Q-Learning**: Handles uncertainty via quantile regression
- **Risk Profiles**: Adjustable risk tolerance (safe/neutral/aggressive)
- **Efficient Logging**: Stores features (not raw video) in Parquet format
- **Safe Checkpoints**: Uses `safetensors` for model weights (fast, pickle-free)
- **Native Overlay**: Frameless always-on-top window (no Overwolf dependency)

### YOLO11 Item Valuation (NEW)
- **Automatic Item Detection**: YOLO11-based extraction screen analysis
- **Rarity Classification**: Automatic detection of item rarity (Epic, Rare, etc.)
- **Smart Valuation**: Context-aware value calculation with set bonuses
- **Game Phase Adjustment**: Dynamic value multipliers based on season phase
- **RL Integration**: Seamless integration with training pipeline
- **Safetensors Support**: Fast and secure model loading
- **Screenshot Capture**: Automatic extraction screen detection

## Project Structure

```
arcx/
├── backend/              # Python backend system
│   ├── arcx/
│   │   ├── api/         # FastAPI server
│   │   ├── ml/          # ML models (encoder, Q-net)
│   │   ├── valuation/   # YOLO11 item valuation (NEW)
│   │   ├── capture/     # Screen capture & extraction detection
│   │   ├── data/        # Data logging/loading (Polars)
│   │   ├── device/      # Device backend detection
│   │   ├── overlay/     # Native overlay (Tk-based client)
│   │   ├── training/    # Training & evaluation pipeline
│   │   └── config.py    # Configuration
│   ├── scripts/
│   │   └── yolo/        # YOLO training & annotation tools
│   ├── examples/        # Usage examples
│   ├── pyproject.toml
│   ├── serve.py         # Run API server
│   └── train.py         # Train models
├── models/                 # Trained model checkpoints (.safetensors)
├── data/                   # Parquet logs & training data
├── YOLO_INTEGRATION.md     # YOLO integration guide
└── VIDEO_WORKFLOW_GUIDE.md # Video-based training workflow
```

## Installation

### Prerequisites

Install modern package manager (10-100x faster!):
```bash
# Install uv (Python - 10-100x faster than pip)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or use `just` commands:
```bash
just install              # Install all dependencies
just update               # Update all dependencies
just upgrade              # Update + rebuild everything
```

See [UV_GUIDE.md](UV_GUIDE.md) for detailed installation and usage.

### Backend (Python)

```bash
cd backend
uv pip install -e .

# Or use sync (faster)
uv sync

# For AMD ROCm support (Windows RX 7000 series):
just install-rocm

# For DirectML support (AMD/Intel GPU fallback):
just install-directml

# See AMD_ROCM_GUIDE.md for detailed GPU compatibility
```

### Native Overlay (Python)

Run the frameless overlay directly from Python:

```bash
cd backend
python -m arcx.overlay.native_overlay
```

> Requires the API server to be running (see Usage below).

## Usage

### 1. Start API Server

```bash
cd backend
python serve.py
```

Server will start on `http://127.0.0.1:8765`

### 2. Launch Native Overlay

```bash
cd backend
python -m arcx.overlay.native_overlay
```

The overlay is a frameless, always-on-top window. Drag the header to move it; use the Start button to begin capture.

### 3. Start Game

The overlay will appear in-game, showing:
- EV values for stay/extract
- Recommendation message with color coding
- Risk profile slider in settings

## API Endpoints

### `GET /ev`

Get current EV recommendation.

**Response:**
```json
{
  "ev_stay": 120.5,
  "ev_extract": 85.3,
  "delta_ev": 35.2,
  "recommendation": "stay",
  "confidence": 0.82,
  "message": "지금은 파밍 계속 권장 (EV 높음)",
  "color": "green",
  "timestamp": 1234567890.0,
  "risk_profile": "neutral"
}
```

### `POST /config`

Update risk profile.

**Request:**
```json
{
  "risk_profile": "aggressive"
}
```

### `POST /feedback`

Submit user feedback.

**Request:**
```json
{
  "run_id": "abc123",
  "decision_idx": 5,
  "timestamp": 1234567890.0,
  "rating": "good"
}
```

### `POST /run/start`

Start a new run.

**Request:**
```json
{
  "map_id": "forest_zone_01"
}
```

**Response:**
```json
{
  "run_id": "uuid-here"
}
```

### `POST /run/end`

End current run with YOLO auto-valuation.

**Request:**
```json
{
  "run_id": "uuid-here",
  "auto_valuation": {
    "total_value": 4500.0,
    "num_items": 5,
    "avg_confidence": 0.87,
    "items": [...],
    "value_breakdown": {...},
    "rarity_counts": {...},
    "phase_multiplier": 1.0
  },
  "total_time_sec": 1800.0,
  "success": true,
  "action_taken": "extract"
}
```

### `POST /valuate` (NEW)

Valuate extraction screenshot using YOLO11.

**Request:**
```json
{
  "screenshot_base64": "<base64_encoded_image>",
  "game_phase": "mid_wipe"
}
```

**Response:**
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
    }
  ],
  "value_breakdown": {
    "weapon": 2500.0,
    "armor": 1500.0
  },
  "rarity_counts": {
    "epic": 2,
    "rare": 3
  },
  "phase_multiplier": 1.0
}
```

## Training

### EV Model Training

#### 1. Collect Data

Play games with the system running. Data is automatically logged to `data/*.parquet`.

#### 2. Train Model

```bash
cd backend
python train.py --epochs 50 --batch-size 32
```

Models are saved as `.safetensors` files in `models/` directory.

### YOLO Item Detector Training (NEW)

#### Option 1: From Video (Recommended)

```bash
# Full automated workflow
just yolo-video-workflow gameplay.mp4

# Or step by step:
# 1. Extract frames (1 FPS)
just yolo-extract-video gameplay.mp4 frames fps=1

# 2. Filter extraction screens (interactive)
just yolo-filter-frames frames/ filtered/

# 3. Annotate
just yolo-annotate filtered/

# 4. Prepare & train
just yolo-prepare annotations.json filtered/
just yolo-train
```

#### Option 2: From Screenshots

```bash
# Full workflow
just yolo-workflow screenshots/

# Or step by step:
# 1. Annotate
just yolo-annotate screenshots/

# 2. Prepare dataset
just yolo-prepare annotations.json screenshots/

# 3. Train
just yolo-train

# Custom settings
just yolo-train data_dir=data/yolo_items model=yolo11s.pt epochs=150 batch=32
```

#### 5. Deploy Model

```bash
cp runs/yolo_train/item_detector/weights/best.safetensors \
   models/item_detector_yolo11.safetensors
```

See `YOLO_INTEGRATION.md` for detailed guide.

## Device Backend Detection

ArcX automatically selects the best available backend:

1. **NVIDIA CUDA**: Full PyTorch CUDA support
2. **AMD ROCm**: AMD GPU on Windows with ROCm PyTorch
3. **DirectML**: AMD/Intel GPU via torch-directml
4. **CPU**: Fallback

Check detected backend in logs:
```
INFO - Using backend: cuda, device: cuda:0
```

## ML Model Architecture

### Frame Encoder
- **Backbone**: ResNet-34 (pretrained on ImageNet)
- **Input**: [3, 225, 400] downscaled frame
- **Output**: z ∈ R^512 latent vector

### Temporal Q-Net
- **Input**: [32, 512] latent sequence (32 frames)
- **Encoder**: BiGRU + Attention pooling
- **Output**: [2, 16] distributional Q-values
  - 2 actions: stay/extract
  - 16 quantiles for uncertainty

### Inference
```python
EV_stay = quantile_Q(s, stay)[q_idx]
EV_extract = quantile_Q(s, extract)[q_idx]

# q_idx based on risk profile:
# - safe: 0.2 (conservative)
# - neutral: 0.5 (median)
# - aggressive: 0.8 (optimistic)
```

## Data Schema (Parquet)

Each decision point is logged with:

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | str | Unique run identifier |
| `decision_idx` | i32 | Decision index in run |
| `t_sec` | f32 | Time since raid start |
| `map_id` | str | Map identifier |
| `action` | str | Action taken (stay/extract) |
| `final_loot_value` | f32 | Final loot value |
| `total_time_sec` | f32 | Total raid duration |
| `success` | bool | Extraction success |
| `z_seq` | List[f32] | Flattened latent sequence (L*D) |

## Development Roadmap

- [x] Phase 1: Backend scaffolding (FastAPI, config, device auto-detect)
- [x] Phase 2: ML model implementation (ResNet encoder + BiGRU/Transformer Q-net)
- [x] Phase 3: Capture & real-time inference (dxcam capture, threaded engine)
- [x] Phase 4: Data pipeline & training (Parquet logging, quantile Huber trainer)
- [x] Phase 5: Native overlay (frameless Tk UI, risk controls, run lifecycle)
- [x] Phase 6: Safetensors deployment (checkpointing/loading without ONNX)
- [x] Phase 7: Multi-game ready (config-driven capture/model settings)

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.
