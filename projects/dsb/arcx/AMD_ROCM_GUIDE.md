# AMD ROCm GPU Support for Windows

Complete guide for installing PyTorch with ROCm support on Windows AMD GPUs.

## GPU Compatibility

### Supported GPUs (gfx110X)

| GPU Model | Architecture | gfx ID | Installation Command |
|-----------|--------------|---------|----------------------|
| RX 7900 XTX | RDNA 3 | gfx1100 | `just install-rocm` |
| RX 7900 XT | RDNA 3 | gfx1100 | `just install-rocm` |
| RX 7800 XT | RDNA 3 | gfx1101 | `just install-rocm` |
| RX 7700 XT | RDNA 3 | gfx1101 | `just install-rocm` |
| RX 7600 XT | RDNA 3 | gfx1102 | `just install-rocm` |
| RX 7600 | RDNA 3 | gfx1102 | `just install-rocm` |

### Your GPU: gfx1101 (RX 7800/7700 XT)

You should use:
```bash
just install-rocm
# or
just install-rocm-gfx110x
```

## Quick Start

### 1. Check Your GPU

```bash
# Windows (PowerShell)
Get-WmiObject Win32_VideoController | Select-Object Name

# Or check Device Manager → Display adapters
```

### 2. Install PyTorch with ROCm

```bash
# For RX 7000 series (gfx110X)
just install-rocm
```

This will:
1. Install ROCm libraries and development tools
2. Install PyTorch, TorchVision, TorchAudio with ROCm support (nightly builds)

### 3. Verify Installation

```bash
cd backend
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'ROCm available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

Expected output:
```
PyTorch: 2.6.0.dev20241124+rocm6.3
ROCm available: True
Device: AMD Radeon RX 7800 XT
```

## Installation Commands

### Primary Installation (Recommended for RX 7000 series)

```bash
just install-rocm
```

**What it installs:**
```bash
# ROCm libraries
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ "rocm[libraries,devel]"

# PyTorch with ROCm
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ --pre torch torchaudio torchvision
```

### Alternative Commands

**For gfx110X (RX 7000 series):**
```bash
just install-rocm-gfx110x
```

**For gfx115X (Future AMD GPUs):**
```bash
just install-rocm-gfx115x
```

**Legacy ROCm 5.7 (Linux/Older GPUs):**
```bash
just install-rocm-legacy
```

## Manual Installation

If you prefer manual installation:

```bash
cd backend

# Install ROCm libraries
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ "rocm[libraries,devel]"

# Install PyTorch with ROCm
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ --pre torch torchaudio torchvision
```

## Verification

### Check PyTorch Version

```bash
uv run python -c "import torch; print(torch.__version__)"
```

Should show something like: `2.6.0.dev20241124+rocm6.3`

### Check CUDA (ROCm) Availability

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Should print: `True`

### Check GPU Name

```bash
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Should show your GPU name: `AMD Radeon RX 7800 XT`

### Run Simple Test

```bash
uv run python -c "
import torch
device = torch.device('cuda')
x = torch.randn(100, 100).to(device)
y = torch.randn(100, 100).to(device)
z = torch.matmul(x, y)
print(f'Matrix multiplication on {torch.cuda.get_device_name(0)}: OK')
print(f'Result shape: {z.shape}')
"
```

## Performance Comparison

### Device Detection

ArcX automatically detects your GPU:

```bash
just check-device
```

Output:
```
DeviceManager(backend=rocm, device=cuda:0)
Memory: {'allocated': 0, 'reserved': 0, 'max_allocated': 0}
```

### Benchmark

```bash
just benchmark-inference
```

Expected performance (RX 7800 XT):
- Inference time: 8-15ms (vs 30-50ms on CPU)
- Training: ~80 samples/sec

## Troubleshooting

### Issue 1: `torch.cuda.is_available()` returns False

**Possible causes:**
1. Wrong GPU architecture
2. ROCm not installed correctly
3. Using CPU-only PyTorch

**Solution:**
```bash
# Uninstall existing PyTorch
uv pip uninstall torch torchvision torchaudio

# Reinstall with ROCm
just install-rocm

# Verify
uv run python -c "import torch; print(torch.__version__)"
# Should contain 'rocm' in version string
```

### Issue 2: `ImportError: cannot import name '_C'`

**Solution:**
```bash
# Clear cache
rm -rf ~/.cache/uv
rm -rf backend/.venv

# Reinstall
cd backend
uv venv
just install-rocm
```

### Issue 3: Out of Memory

**Solution:**
```python
# In backend/arcx/config.py
class ModelConfig(BaseModel):
    batch_size: int = 16  # Reduce from 32
```

Or use smaller model:
```bash
# Use ResNet-34 instead of ResNet-50
# Already default in config.py
```

### Issue 4: Slow Performance

**Check GPU usage:**
```bash
# Windows Task Manager → Performance → GPU
# Should show GPU usage when running inference
```

**Verify ROCm backend:**
```bash
uv run python -c "from arcx.device import device_manager; print(device_manager.backend)"
```

Should print: `DeviceBackend.ROCM`

## ROCm vs DirectML

| Feature | ROCm | DirectML |
|---------|------|----------|
| Performance | ⚡⚡⚡ Fast | ⚡⚡ Moderate |
| Compatibility | RX 7000 series | All AMD/Intel GPUs |
| Installation | Nightly builds | Stable |
| Windows Support | ✅ Yes (nightly) | ✅ Yes (official) |
| Inference Speed | 8-15ms | 15-25ms |
| Training Speed | Fast | Moderate |

**Recommendation:**
- **RX 7000 series**: Use ROCm (better performance)
- **RX 6000/5000 series**: Use DirectML
- **Integrated GPU**: Use DirectML

## Switching Between Backends

### From DirectML to ROCm

```bash
# Uninstall DirectML
uv pip uninstall torch-directml torch torchvision

# Install ROCm
just install-rocm

# Verify
just check-device
```

### From ROCm to DirectML

```bash
# Uninstall ROCm PyTorch
uv pip uninstall torch torchvision torchaudio

# Install CPU PyTorch first
uv pip install torch torchvision

# Install DirectML
just install-directml

# Verify
just check-device
```

## ROCm Nightly Builds

ArcX uses ROCm nightly builds from AMD:
- **Repository**: https://rocm.nightlies.amd.com/
- **gfx110X**: RX 7000 series (gfx1100, gfx1101, gfx1102)
- **Update frequency**: Daily
- **Stability**: Generally stable, but can have occasional issues

### Pinning Specific Version

To avoid breaking changes from nightly updates:

```bash
# Find current version
uv pip show torch | grep Version

# Example: 2.6.0.dev20241124+rocm6.3

# Pin in pyproject.toml
[project]
dependencies = [
    "torch==2.6.0.dev20241124+rocm6.3",
]
```

## Advanced Configuration

### Environment Variables

```bash
# Set ROCm visible devices
export ROCM_VISIBLE_DEVICES=0

# Set HIP device order
export HIP_VISIBLE_DEVICES=0

# Enable ROCm debugging
export AMD_LOG_LEVEL=3
```

### Windows PowerShell

```powershell
$env:ROCM_VISIBLE_DEVICES=0
$env:HIP_VISIBLE_DEVICES=0
```

## Additional Resources

- **AMD ROCm**: https://rocm.docs.amd.com/
- **PyTorch ROCm**: https://pytorch.org/get-started/locally/
- **ROCm Nightlies**: https://rocm.nightlies.amd.com/
- **AMD Community**: https://community.amd.com/

## Summary

For **RX 7800 XT (gfx1101)**:

```bash
# 1. Install ROCm PyTorch
just install-rocm

# 2. Verify installation
just check-device

# 3. Benchmark performance
just benchmark-inference

# 4. Start using!
just serve
```

Expected output:
```
✓ ROCm PyTorch installed for gfx110X
DeviceManager(backend=rocm, device=cuda:0)
Average inference time: 10.5ms
```

🎉 **You're ready to use ArcX with AMD ROCm acceleration!**
