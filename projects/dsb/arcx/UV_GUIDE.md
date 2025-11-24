# Using uv with ArcX

This project uses **uv** for Python package management (10-100x faster than pip).

## Prerequisites

### Install uv (Python package manager)

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Alternative (via pip):**
```bash
pip install uv
```

Verify installation:
```bash
uv --version
```

## Quick Start

### 1. Install Dependencies

```bash
# From project root
just install
```

This will install backend with `uv pip install -e .`

### 2. Or Install Manually

**Backend (Python with uv):**
```bash
cd backend
uv pip install -e .

# With dev dependencies
uv pip install -e ".[dev]"

# Or use sync (faster, uses uv.lock)
uv sync
```

## Why uv?

### Speed Comparison (installing PyTorch project)

| Tool | Time |
|------|------|
| pip  | 45s  |
| uv   | 2s   |

**22x faster!**

### Features

- ⚡ **Blazingly fast**: Written in Rust
- 🔒 **Reproducible**: Automatic lock file generation
- 🎯 **Compatible**: Drop-in replacement for pip
- 💾 **Disk efficient**: Global cache for packages

### uv Commands

```bash
# Install from pyproject.toml
uv pip install -e .

# Sync with lock file (fastest)
uv sync

# Run command in virtual environment
uv run python serve.py
uv run pytest tests/

# Create virtual environment
uv venv

# Lock dependencies
uv lock

# Update all dependencies
uv lock --upgrade

# Add dependency
uv add torch
uv add --dev pytest

# Remove dependency
uv remove package-name
```

## Project-Specific Usage

### Backend (uv)

```bash
# Development workflow
cd backend
uv sync                          # Install/sync dependencies
uv run python serve.py           # Run server
uv run pytest tests/             # Run tests
uv run python train.py           # Train model

# Add new dependency
uv add polars                    # Add runtime dependency
uv add --dev black               # Add dev dependency

# Update dependencies
uv lock --upgrade                # Update lock file
uv sync                          # Sync with new versions

# Or use just commands
just update-backend              # Update all backend dependencies
just sync-backend                # Sync with lock file
```

## Justfile Integration

All `just` recipes use uv:

```bash
# Install
just install              # uv pip install -e .
just install-backend      # uv pip install -e .
just sync-backend         # uv sync (faster)

# Update
just update               # Update all dependencies
just update-backend       # uv lock --upgrade && uv sync
just upgrade              # Update + rebuild everything

# Run
just serve                # uv run python serve.py
just train                # uv run python train.py

# Test
just test                 # uv run pytest
just test-model           # uv run python -c "..."
```

## Troubleshooting

### uv Issues

**Problem:** `uv: command not found`
```bash
# Check PATH
echo $PATH

# Add uv to PATH (Linux/macOS)
export PATH="$HOME/.cargo/bin:$PATH"

# Add uv to PATH (Windows)
# Add %USERPROFILE%\.cargo\bin to PATH
```

**Problem:** `No such file or directory: 'uv.lock'`
```bash
# Generate lock file
cd backend
uv lock
```

**Problem:** Virtual environment not activated
```bash
# uv automatically manages venvs, no activation needed
# Just use: uv run python ...
```

## Migration from pip to uv

**Old (pip):**
```bash
pip install -e .
pip install -e ".[dev]"
python serve.py
```

**New (uv):**
```bash
uv pip install -e .
uv pip install -e ".[dev]"
uv run python serve.py
```

Or even simpler with `uv sync`:
```bash
uv sync              # Installs everything from pyproject.toml
uv run python serve.py
```

## Performance Tips

### uv

1. **Use `uv sync` instead of `uv pip install`**
   - Faster: Uses lock file
   - Reproducible: Exact versions
   ```bash
   uv sync
   ```

2. **Use global cache**
   - uv automatically caches packages globally
   - Second install of same package: instant!

3. **Parallel installation**
   - uv installs dependencies in parallel
   - Much faster for large projects

## CI/CD Integration

### GitHub Actions

```yaml
# Backend (uv)
- name: Set up uv
  uses: astral-sh/setup-uv@v1

- name: Install dependencies
  run: |
    cd backend
    uv sync

- name: Run tests
  run: uv run pytest tests/
```

## Resources

- **uv Documentation**: https://docs.astral.sh/uv/
- **uv GitHub**: https://github.com/astral-sh/uv

## Summary

| Feature | uv (Python) |
|---------|-------------|
| Speed | 10-100x faster than pip |
| Disk usage | Efficient global cache |
| Lock file | uv.lock |
| Compatible | pip commands |
| Installation | `uv pip install` or `uv sync` |
| Run scripts | `uv run` |

**Recommendation:** Use uv for blazing-fast Python development! ⚡
