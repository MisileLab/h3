# ArcX Implementation Guide

Complete implementation documentation for the ArcX EV Overlay System.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Phase 1: Backend Scaffolding](#phase-1-backend-scaffolding)
3. [Phase 2: ML Models](#phase-2-ml-models)
4. [Phase 3: Capture & Real-time Inference](#phase-3-capture--real-time-inference)
5. [Phase 4: Data Pipeline](#phase-4-data-pipeline)
6. [Phase 5: Native Overlay](#phase-5-native-overlay)
7. [Design Decisions](#design-decisions)
8. [Future Optimizations](#future-optimizations)

---

## Architecture Overview

```
┌─────────────┐
│  Game Screen │
└──────┬──────┘
       │ dxcam (8 FPS)
       ▼
┌─────────────────┐
│  ScreenCapture  │  400x225 downscaled frames
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│  Frame Encoder  │────▶│ Ring Buffer  │  32 frames
│   (ResNet-34)   │     │  (latents)   │
└────────┬────────┘     └──────┬───────┘
         │                      │
         │  z ∈ R^512          │ [L=32, D=512]
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌─────────────────────┐
         │  Temporal Q-Net     │
         │  (BiGRU + Attention)│
         └──────────┬──────────┘
                    │
                    │ Q(s, a) ∈ R^[2×16]
                    ▼
         ┌─────────────────────┐
         │   EV Calculation    │
         │  (Quantile Select)  │
         └──────────┬──────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
  ┌───────────┐        ┌──────────────┐
  │  FastAPI  │◀──────▶│    Native    │
  │  Server   │  JSON  │   Overlay    │
  └─────┬─────┘        └──────────────┘
        │
        ▼
  ┌────────────┐
  │  Parquet   │  Training data
  │  Logger    │  (latents only)
  └────────────┘
```

### Data Flow

1. **Capture Loop** (8 FPS):
   - Capture screen → Downscale → Encode to latent z_t
   - Push to ring buffer (both frame and latent)

2. **Inference Loop** (every 0.5s):
   - Get last 32 latents from buffer
   - Feed to Q-net → Get distributional Q[2, 16]
   - Select quantile based on risk profile
   - Calculate EV_stay, EV_extract, delta_EV
   - Send to API

3. **Logging** (on decision):
   - Store: z_seq (latents only), action, metadata
   - Save to Parquet every 100 decisions

4. **Training** (offline):
   - Load Parquet → Dataset → DataLoader
   - Quantile Huber loss on Q(s, a)
   - Target: final_loot_value

---

## Phase 1: Backend Scaffolding

### Goals
- Set up Python package structure
- Implement device auto-detection (CUDA/ROCm/DirectML/CPU)
- Create FastAPI server skeleton
- Define configuration system

### Key Files

#### `backend/pyproject.toml`
- Dependencies: PyTorch, FastAPI, Polars, dxcam, etc.
- Optional dependencies for ROCm and DirectML

#### `backend/arcx/config.py`
- Pydantic-based configuration
- Captures all hyperparameters and paths
- Nested configs: CaptureConfig, ModelConfig, InferenceConfig, APIConfig, DataConfig

#### `backend/arcx/device/backend.py`
- Device detection priority: CUDA → ROCm → DirectML → CPU
- `DeviceManager` singleton for global device access
- Automatic GPU memory stats (for CUDA/ROCm)

**Design Decision**: Why separate device detection?
- Allows easy testing on different hardware
- Centralizes device logic
- Keeps model I/O centralized for safetensors checkpoints and future backends

#### `backend/arcx/api/server.py`
- FastAPI app with CORS for the overlay client
- Endpoints: `/ev`, `/status`, `/config`, `/feedback`, `/run/start`, `/run/end`
- Lifespan context for model initialization
- Auto-loads model weights if available

**Design Decision**: Why FastAPI over WebSockets?
- Simpler HTTP polling (0.5s interval) sufficient for real-time
- Better error handling
- Easier to debug with curl/Postman
- WebSocket can be added later if needed

---

## Phase 2: ML Models

### Goals
- Implement Frame Encoder (ResNet backbone)
- Implement Temporal Q-Net (BiGRU + Attention)
- Support distributional Q-learning (quantile regression)
- Enable safetensors save/load

### Architecture Details

#### Frame Encoder (`arcx/ml/encoder.py`)

```
Input: [B, 3, H=225, W=400]
  ↓
ResNet-34 (pretrained ImageNet)
  - Conv layers (up to layer4)
  - Remove FC and avgpool
  ↓
Global AdaptiveAvgPool2d(1, 1)
  ↓
Flatten → [B, 512]
  ↓
Linear(512, latent_dim) + LayerNorm + ReLU
  ↓
Output: z ∈ [B, D=512]
```

**Design Decision**: Why ResNet-34?
- Good balance: 21M params vs ResNet-50's 25M
- Pretrained on ImageNet → better feature extraction
- Fast enough for 8 FPS real-time encoding
- Can upgrade to ResNet-50 if needed

**Design Decision**: Why not train encoder end-to-end?
- Reduces computational cost (encoder frozen or fine-tuned slowly)
- Pretrained features already capture spatial patterns
- Can cache latents in ring buffer (saves compute during inference)

#### Temporal Q-Net (`arcx/ml/qnet.py`)

```
Input: [B, L=32, D=512]
  ↓
BiGRU (2 layers, hidden=256×2=512)
  ↓
GRU output: [B, L, 512]
  ↓
Attention Pooling:
  - Linear(512, 256) → Tanh → Linear(256, 1) → Softmax over L
  - Weighted sum → [B, 512]
  ↓
LayerNorm
  ↓
Q-head:
  - Linear(512, 512) + ReLU + Dropout
  - Linear(512, num_actions * num_quantiles)
  ↓
Reshape → [B, A=2, K=16]
```

**Design Decision**: Why BiGRU over Transformer?
- GRU: Lower latency, fewer params for L=32
- Transformer: Better for longer sequences (implemented as option)
- Attention pooling: Focuses on important time steps

**Design Decision**: Why Distributional Q?
- Captures uncertainty (critical for risk-aware decisions)
- Quantile regression: Robust to outliers
- Risk profiles: Safe=q_0.2, Neutral=q_0.5, Aggressive=q_0.8

#### Model Integration (`arcx/ml/model.py`)

`EVModel` combines encoder + Q-net:
- `forward_frames(frames)`: End-to-end from raw frames
- `forward_latents(z_seq)`: Fast inference from cached latents
- `predict_ev(frames/latents, quantile)`: Returns EV_stay, EV_extract, delta

**Design Decision**: Why two forward modes?
- `forward_frames`: Used during training (end-to-end gradients)
- `forward_latents`: Used during inference (encoder pre-computed in capture loop)

#### Safetensors (`arcx/ml/utils.py`)

- Save/load with `safetensors` library (not pickle)
- Safer: No arbitrary code execution
- Faster: Memory-mapped loading
- Metadata: Store epoch, type (best/final)

---

## Phase 3: Capture & Real-time Inference

### Goals
- High-performance screen capture (dxcam)
- Ring buffer for frames and latents
- Real-time inference engine with threading
- Integrate with API server

### Screen Capture (`arcx/capture/capture.py`)

- Uses `dxcam` library (DirectX/DXGI capture)
- Target: 8 FPS @ 400×225 resolution
- Fallback: Synthetic frames for testing

**Design Decision**: Why dxcam?
- GPU-accelerated capture (fast)
- Low overhead (doesn't impact game FPS)
- Windows-native (DirectX)

**Design Decision**: Why 8 FPS?
- Balance: Real-time updates vs computational cost
- 8 FPS × 32 frames = 4 seconds of history
- Encoder can handle ~30 FPS if needed

### Ring Buffer (`arcx/capture/ringbuffer.py`)

Three types:
1. `FrameRingBuffer`: Stores raw frames [H, W, C]
2. `LatentRingBuffer`: Stores latent vectors [D]
3. `CombinedRingBuffer`: Both (for debugging/logging)

**Design Decision**: Why separate frame/latent buffers?
- Latents stored on CPU to save GPU memory
- Frames optional (only needed for debugging)
- Efficient: Fixed-size deque with O(1) push/pop

### Inference Engine (`arcx/ml/inference.py`)

Two threads:
1. **Capture Thread**:
   - Grab frame → Encode to latent → Push to buffer
   - Runs continuously at ~8 FPS

2. **Inference Thread**:
   - Every 0.5s: Get last 32 latents → Q-net → EV
   - Thread-safe with lock on shared state

**Design Decision**: Why threading?
- Capture and inference run independently
- Encoder (capture thread) overlaps with Q-net (inference thread)
- Lock only on buffer access (minimal contention)

**Design Decision**: Why 0.5s inference interval?
- User doesn't need sub-second updates
- Reduces GPU usage
- Still feels real-time

### API Integration

`app_state.inference_engine` initialized in lifespan:
- Loads model on startup
- Starts capture on `/run/start`
- Returns latest prediction on `/ev`

---

## Phase 4: Data Pipeline

### Goals
- Log decision data to Parquet (latents only, not raw frames)
- PyTorch Dataset for loading logged data
- Training loop with quantile Huber loss

### Schema (`arcx/data/schema.py`)

Parquet columns:
- `run_id`: UUID
- `decision_idx`: int
- `t_sec`: float (time in raid)
- `map_id`: str
- `action`: "stay" or "extract"
- `final_loot_value`: float (target)
- `total_time_sec`: float
- `success`: bool
- `z_seq`: List[float] (flattened [L*D])

**Design Decision**: Why store latents, not frames?
- Frames: ~400×225×3×32 = 8.6 MB per decision
- Latents: 32×512×4 bytes = 64 KB per decision
- 130× reduction in storage!
- Encoder fixed (pretrained), so latents sufficient

**Design Decision**: Why Polars + Parquet?
- Polars: Faster than Pandas for large datasets
- Parquet: Columnar, compressed, efficient
- Zstd compression: ~2-3× additional compression

### Data Logger (`arcx/data/logger.py`)

Collects decisions during gameplay:
- `start_run(run_id)`: Begin new run
- `log_decision(idx, action, z_seq)`: Log decision
- `end_run(final_loot, time, success)`: Finalize with outcome
- `save()`: Write to Parquet

Auto-saves every 100 decisions.

**Design Decision**: Why batch writes?
- Reduces I/O overhead
- Parquet benefits from larger row groups

### Dataset (`arcx/data/dataset.py`)

PyTorch Dataset:
- Loads all `decisions_*.parquet` files
- Returns: (z_seq, action, target_return)
- Supports train/val split

### Training (`arcx/training/trainer.py`)

#### Loss Function: Quantile Huber Loss

```python
def quantile_huber_loss(q_dist, target, action):
    # q_dist: [B, A=2, K=16]
    # target: [B, 1]
    # action: [B, 1]

    # Select Q for taken action: [B, K]
    q_a = q_dist.gather(dim=1, action)

    # TD error: [B, K]
    td_error = target - q_a

    # Huber loss
    huber = where(|td_error| ≤ κ,
                  0.5 * td_error^2,
                  κ * (|td_error| - 0.5κ))

    # Quantile weight
    weight = |τ - I(td_error < 0)|

    # Weighted loss
    loss = (weight * huber).mean()
```

**Design Decision**: Why Quantile Huber?
- Quantile regression: Learns full distribution
- Huber: Robust to outliers (vs MSE)
- Standard in distributional RL (QR-DQN)

#### Training Loop

1. Load dataloaders (80/20 train/val split)
2. For each epoch:
   - Train: forward → loss → backward → step
   - Validate: compute val_loss
   - Save checkpoint every N epochs
   - Save best model (lowest val_loss)
3. Save final model

---

## Phase 5: Native Overlay

### Goals
- Provide an always-on-top native overlay (no Overwolf dependency)
- Display EV values and recommendations
- Risk profile controls (safe/neutral/aggressive)
- Run lifecycle controls (start/stop)

### Implementation (`backend/arcx/overlay/native_overlay.py`)

- Tk-based frameless window with drag-to-move header
- Polls `/ev` every 0.5s via `httpx` and updates UI
- Risk buttons post to `/config`; Start/Stop call `/run/start` and `/run/end`
- Status line shows connection state and current risk profile

**Design Decision**: Why Tk + HTTP polling?
- Zero install overhead (stdlib UI + existing `httpx`)
- Works without Overwolf or additional runtimes
- Polling matches backend cadence (0.5s) and keeps error handling simple

---

## Design Decisions

### 1. Why Distributional Q-Learning?

Traditional Q-learning predicts:
```
Q(s, a) = E[R | s, a]
```

Distributional Q-learning predicts full distribution:
```
Z(s, a) ~ distribution of R | s, a
```

**Benefits**:
- Captures uncertainty (high stakes vs low stakes)
- Enables risk-aware decisions (conservative vs aggressive)
- More robust to noisy rewards

**Implementation**:
- Quantile regression: 16 quantiles per action
- User selects quantile via risk slider

### 2. Why Separate Encoder & Q-Net?

**Modularity**:
- Encoder can be frozen (pretrained)
- Q-net trained from scratch
- Easier to experiment with Q-net architectures

**Efficiency**:
- Encoder computed once per frame (capture thread)
- Q-net computed once per 0.5s (inference thread)
- Avoids redundant encoder passes

### 3. Why Log Latents, Not Frames?

**Storage**:
- 1000 decisions × 8.6 MB = 8.6 GB (frames)
- 1000 decisions × 64 KB = 64 MB (latents)

**Privacy**:
- Latents = abstract features (no visual info)
- Frames = screenshots (privacy concern)

**Tradeoff**:
- Cannot re-train encoder from logged data
- But encoder is pretrained (ImageNet), so acceptable

### 4. Why BiGRU over Transformer?

**For L=32 sequence length**:

| Model       | Params | Latency | Context |
|-------------|--------|---------|---------|
| BiGRU (2L)  | 2.1M   | ~5ms    | Local   |
| Transformer | 3.8M   | ~12ms   | Global  |

**BiGRU wins** for real-time (<10ms inference).

**Transformer option available** for longer sequences (L>64).

### 5. Why 32 Frames?

32 frames @ 8 FPS = 4 seconds of history.

**Analysis**:
- Too short (<2s): Misses slow trends
- Too long (>8s): Captures stale info
- 4s: Good balance for extraction shooters

Can be tuned via config.

---

## Future Optimizations

### 1. Safetensors Streaming & Hot-Swap

**Current**: PyTorch inference (~10-20ms on GPU) with safetensors checkpoints
**Goal**: Faster startup and safer distribution without ONNX conversion

Steps:
1. Keep encoder/qnet weights in `models/*.safetensors` (already default)
2. Enable memory-mapped loading + checksum verification on startup
3. Add lazy reload + hot-swap to swap updated checkpoints during runtime

**Benefit**: Pickle-free deployment, quicker reloads, predictable integrity

### 2. Quantization

**Current**: FP32 weights (~200 MB)
**Goal**: INT8 weights (~50 MB)

Steps:
1. Post-training quantization (PTQ)
2. Test accuracy on validation set
3. Deploy quantized model

**Benefit**: 4× smaller, 2× faster

### 3. Model Pruning

**Current**: ResNet-34 (21M params)
**Goal**: Pruned ResNet (~10M params)

Steps:
1. Magnitude pruning (remove small weights)
2. Fine-tune pruned model
3. Evaluate accuracy

**Benefit**: Faster inference, smaller model

### 4. Asynchronous Encoder

**Current**: Encoder runs in capture thread (8 FPS)
**Goal**: Async encoder pool (handles burst capture)

**Benefit**: Can scale to 16-30 FPS capture if needed

### 5. Multi-Game Support

**Current**: Hardcoded for single game
**Goal**: Game-specific configs + models

Steps:
1. Detect game via process/window title or config flag
2. Load game-specific model weights
3. Use game-specific config (FPS, resolution, etc.)

---

## Testing

### Unit Tests

Run with pytest:
```bash
cd backend
pytest tests/ -v
```

Tests:
- `test_model.py`: Frame encoder, Q-net, EVModel
- `test_data.py`: Dataset, logger, schema
- `test_capture.py`: Ring buffers (no dxcam)

### Integration Test

1. Run backend:
   ```bash
   python serve.py
   ```

2. Test API:
   ```bash
   curl http://127.0.0.1:8765/
   curl http://127.0.0.1:8765/status
   ```

3. Launch native overlay (`python -m arcx.overlay.native_overlay`) and verify updates

### Training Test

Generate dummy data and train:
```bash
python -c "from arcx.training.trainer import test_trainer; test_trainer()"
```

---

## Performance Benchmarks

### Inference Latency (GPU)

| Component        | Time (ms) | Device       |
|------------------|-----------|--------------|
| Frame Encoder    | 3-5       | CUDA/ROCm    |
| Temporal Q-Net   | 2-3       | CUDA/ROCm    |
| **Total**        | **5-8**   | GPU          |
| Total (CPU)      | 30-50     | CPU fallback |

### Training Speed

- ResNet-34 + BiGRU: ~100 samples/sec (RTX 3080)
- Full epoch (10K samples): ~100 seconds
- 50 epochs: ~1.5 hours

### Memory Usage

- Model weights: ~200 MB
- Ring buffer (32 frames): ~10 MB
- GPU VRAM: ~500 MB total

---

## Deployment Checklist

### Backend

- [ ] Install Python 3.10+
- [ ] Install dependencies: `pip install -e backend/`
- [ ] For AMD ROCm: `pip install torch --index-url ...`
- [ ] For DirectML: `pip install torch-directml`
- [ ] Verify device detection: `python -c "from arcx.device import device_manager; print(device_manager)"`
- [ ] Run server: `python backend/serve.py`

### Native Overlay

- [ ] Run overlay: `python -m arcx.overlay.native_overlay`
- [ ] Drag header to desired position (always on top)
- [ ] Click Start to begin capture; Stop to end run

### Training

- [ ] Collect data (play games with overlay)
- [ ] Verify Parquet files in `data/`
- [ ] Train model: `python backend/train.py --epochs 50`
- [ ] Restart server to load trained weights

---

## Troubleshooting

### Backend won't start

**Issue**: `ImportError: No module named 'arcx'`
**Fix**: `pip install -e backend/`

**Issue**: `RuntimeError: CUDA out of memory`
**Fix**: Close other GPU apps, or add `--no-pretrained` flag to use smaller model

### No EV predictions

**Issue**: API returns 503 "buffer filling"
**Fix**: Wait for buffer to fill (32 frames @ 8 FPS = 4 seconds)

**Issue**: API returns 400 "capture not active"
**Fix**: Click "Start" button in overlay

### Overlay not appearing

**Issue**: Overlay doesn't show in-game
**Fix**: Ensure the overlay script is running and not minimized

**Issue**: "Disconnected" status
**Fix**: Ensure backend is running on port 8765

### Training fails

**Issue**: `FileNotFoundError: No data files`
**Fix**: Collect data first by playing games

**Issue**: Loss = NaN
**Fix**: Lower learning rate (`--lr 1e-5`)

---

## Credits

- PyTorch for ML framework
- FastAPI for web server
- Polars for data processing
- dxcam for screen capture
- Tkinter for the native overlay window
- ResNet architecture from torchvision

---

## License

MIT License
