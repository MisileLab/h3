# ArcX Quick Start Guide

Get up and running with ArcX in 5 minutes.

## Prerequisites

- Windows 10/11
- Python 3.10 or later
- **uv** (Python package manager) - [Install guide](https://docs.astral.sh/uv/)
- NVIDIA/AMD GPU (optional, will fallback to CPU)
- (Optional, legacy only) Node.js + pnpm + Overwolf for the old Overwolf app

### Install uv

**Install uv:**
```bash
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Step 1: Install Backend

```bash
cd backend
uv pip install -e .
```

Or use `uv sync` (faster, uses lock file):
```bash
uv sync
```

For AMD GPU with ROCm (RX 7000 series):
```bash
just install-rocm
```

For AMD/Intel with DirectML (fallback):
```bash
just install-directml
```

See [AMD_ROCM_GUIDE.md](AMD_ROCM_GUIDE.md) for GPU compatibility.

## Step 2: Start Backend Server

```bash
cd backend
python serve.py
```

You should see:
```
INFO - Using backend: cuda, device: cuda:0
INFO - Model loaded on device: cuda:0
INFO - API server ready
INFO - Uvicorn running on http://127.0.0.1:8765
```

**Note**: First run may download pretrained ResNet weights (~85 MB).

## Step 3: Launch Native Overlay

```bash
cd backend
python -m arcx.overlay.native_overlay
```

- Drag the header to move the overlay
- Click **Start** to begin capture (fires `/run/start`)
- Click **Stop** to end the run

## Step 4: Launch Game

1. Start your extraction shooter game
2. Keep the overlay window visible (it's always on top)
3. If capture isn't running, click **Start** on the overlay

## Step 5: Use the Overlay

### Understanding the UI

- **EV_stay**: Expected value for continuing to farm
- **EV_extract**: Expected value for extracting now
- **Delta**: Difference (positive = stay is better)
- **Recommendation**: Color-coded suggestion
  - Green: Stay (safer to farm)
  - Yellow: Neutral (your call)
  - Red: Extract (higher risk)
- **Status line**: Shows risk profile and connection state

### Controls

- **Risk buttons**: Safe / Neutral / Aggressive (updates `/config`)
- **Start/Stop**: Manages `/run/start` and `/run/end` from the overlay

## Step 6: Train Model (Optional)

After collecting data from several games:

```bash
cd backend
python train.py --epochs 50 --batch-size 32
```

This will:
1. Load decision logs from `data/` directory
2. Train the model for 50 epochs
3. Save best model to `models/`

Next time you start the server, it will automatically load the trained weights.

## Verification

Test the API:

```bash
# Check server is running
curl http://127.0.0.1:8765/

# Check system status
curl http://127.0.0.1:8765/status

# Get EV prediction (requires active run)
curl http://127.0.0.1:8765/ev
```

## Troubleshooting

### Backend Issues

**Problem**: `No module named 'arcx'`
```bash
cd backend
pip install -e .
```

**Problem**: `CUDA out of memory`
- Close other GPU applications
- Use CPU fallback (automatic)

**Problem**: `No EV prediction available`
- Wait 4 seconds for buffer to fill (32 frames @ 8 FPS)
- Check that you clicked "시작" button

### Overlay Issues

**Problem**: Overlay doesn't appear
- Ensure `python -m arcx.overlay.native_overlay` is running
- Overlay may be off-screen; rerun the script to reset position
- Make sure transparency isn't blocked by capture overlays (e.g., screen recorders)

**Problem**: "Disconnected" status
- Ensure backend server is running
- Check firewall isn't blocking port 8765
- Try `curl http://127.0.0.1:8765/` to verify

## Default Settings

| Setting | Value |
|---------|-------|
| Capture FPS | 8 |
| Capture Resolution | 400×225 |
| Inference Interval | 0.5 seconds |
| Sequence Length | 32 frames (4 seconds) |
| API Port | 8765 |
| Risk Profile | Neutral |

Edit `backend/arcx/config.py` to change these.

## Data Collection

Data is automatically saved to `data/*.parquet` when:
- You start a run (click "Start")
- You end a run (click "Stop")
- Backend logs decisions every 100 samples

**Privacy**: Only latent vectors are saved, not raw screenshots.

## Next Steps

- Collect more data by playing games
- Train custom model on your playstyle
- Adjust risk profile to your preference
- (Optional) Post `/feedback` via API or extend the overlay for ratings

## Support

For issues or questions:
- Check `IMPLEMENTATION.md` for detailed documentation
- Review logs in backend console
- Check the overlay console output (stdout)

## Performance Tips

1. **GPU Recommended**: 5-8ms inference vs 30-50ms on CPU
2. **Close Background Apps**: Reduces GPU contention
3. **Lower Game Settings**: If FPS drops below capture rate
4. **Train on Your Data**: Model improves with your playstyle

## Advanced

### Custom Model Config

Edit `backend/arcx/config.py`:
```python
class ModelConfig(BaseModel):
    encoder_backbone: str = "resnet50"  # Larger model
    hidden_dim: int = 768  # More capacity
    num_quantiles: int = 32  # Finer distribution
```

### Safetensors Checkpoints

- Training saves `models/encoder_*.safetensors` and `models/qnet_*.safetensors`
- The API auto-loads these on startup (no ONNX export needed)

```bash
# Manually write fresh checkpoints if you tweak the model
python - <<'PY'
from pathlib import Path
from arcx.ml.model import EVModel
from arcx.ml.utils import save_model_safetensors

model = EVModel(encoder_pretrained=False)
save_model_safetensors(model.encoder, Path("models/encoder_manual.safetensors"))
save_model_safetensors(model.qnet, Path("models/qnet_manual.safetensors"))
print("Saved manual safetensors checkpoints")
PY
```

### Multiple Games

- Tweak `backend/arcx/config.py` per game (capture FPS/resolution, model paths)
- Keep per-game safetensors in `models/` and swap before starting the server

---

Happy farming! 🎮
