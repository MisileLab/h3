# Using uv and pnpm with ArcX

This project uses modern package managers for faster installation and better dependency management:
- **uv** for Python (10-100x faster than pip)
- **pnpm** for Node.js (3x faster than npm, saves disk space)

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

### Install pnpm (Node.js package manager)

**Windows:**
```powershell
iwr https://get.pnpm.io/install.ps1 -useb | iex
```

**macOS/Linux:**
```bash
curl -fsSL https://get.pnpm.io/install.sh | sh -
```

**Alternative (via npm):**
```bash
npm install -g pnpm
```

Verify installation:
```bash
pnpm --version
```

## Quick Start

### 1. Install All Dependencies

```bash
# From project root
just install
```

This will:
- Install backend with `uv pip install -e .`
- Install Overwolf with `pnpm install`

### 2. Or Install Separately

**Backend (Python with uv):**
```bash
cd backend
uv pip install -e .

# With dev dependencies
uv pip install -e ".[dev]"

# Or use sync (faster, uses uv.lock)
uv sync
```

**Overwolf (Node.js with pnpm):**
```bash
cd overwolf
pnpm install
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

## Why pnpm?

### Speed Comparison (installing typical frontend project)

| Tool | Time | Disk Space |
|------|------|------------|
| npm  | 50s  | 500 MB     |
| yarn | 35s  | 450 MB     |
| pnpm | 17s  | 200 MB     |

**3x faster, 2.5x less disk space!**

### Features

- ⚡ **Fast**: Parallel installation
- 💾 **Efficient**: Content-addressable storage (symlinks)
- 🔒 **Strict**: Flat node_modules, no phantom dependencies
- 🎯 **Compatible**: Drop-in replacement for npm

### pnpm Commands

```bash
# Install dependencies
pnpm install

# Add dependency
pnpm add package-name
pnpm add -D dev-package

# Remove dependency
pnpm remove package-name

# Run script
pnpm run build
pnpm run watch

# Update dependencies
pnpm update

# Check outdated packages
pnpm outdated

# Clean install
pnpm install --frozen-lockfile
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

### Overwolf (pnpm)

```bash
# Development workflow
cd overwolf
pnpm install                     # Install dependencies
pnpm run build                   # Build once
pnpm run watch                   # Build on changes

# Add new dependency
pnpm add library-name            # Add dependency
pnpm add -D dev-library          # Add dev dependency

# Update dependencies
pnpm update                      # Update all
pnpm update library-name         # Update specific package

# Or use just commands
just update-overwolf             # Update all Overwolf dependencies
just build-overwolf              # Build after update
```

## Justfile Integration

All `just` recipes use uv and pnpm:

```bash
# Install
just install              # Install all (uv + pnpm)
just install-backend      # uv pip install -e .
just sync-backend         # uv sync (faster)
just install-overwolf     # pnpm install

# Update
just update               # Update all dependencies
just update-backend       # uv lock --upgrade && uv sync
just update-overwolf      # pnpm update
just upgrade              # Update + rebuild everything

# Run
just serve                # uv run python serve.py
just train                # uv run python train.py
just build-overwolf       # pnpm run build

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

### pnpm Issues

**Problem:** `pnpm: command not found`
```bash
# Reinstall pnpm
npm install -g pnpm

# Or use standalone installer
curl -fsSL https://get.pnpm.io/install.sh | sh -
```

**Problem:** `ERR_PNPM_NO_MATCHING_VERSION`
```bash
# Clear cache
pnpm store prune

# Reinstall
rm -rf node_modules pnpm-lock.yaml
pnpm install
```

**Problem:** Phantom dependencies error
```bash
# This is actually a feature! pnpm is stricter.
# Add the missing dependency explicitly:
pnpm add missing-package
```

## Migration from pip/npm

### From pip to uv

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

### From npm to pnpm

**Old (npm):**
```bash
npm install
npm run build
npm run watch
```

**New (pnpm):**
```bash
pnpm install
pnpm run build
pnpm run watch
```

Exact same syntax! Just replace `npm` with `pnpm`.

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

### pnpm

1. **Use frozen lockfile in CI**
   ```bash
   pnpm install --frozen-lockfile
   ```

2. **Clean store periodically**
   ```bash
   pnpm store prune
   ```

3. **Use workspace for monorepo**
   - Shared dependencies across projects
   - Saves disk space

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

# Overwolf (pnpm)
- name: Setup pnpm
  uses: pnpm/action-setup@v2
  with:
    version: 8

- name: Install dependencies
  run: |
    cd overwolf
    pnpm install --frozen-lockfile

- name: Build
  run: pnpm run build
```

## Resources

- **uv Documentation**: https://docs.astral.sh/uv/
- **pnpm Documentation**: https://pnpm.io/
- **uv GitHub**: https://github.com/astral-sh/uv
- **pnpm GitHub**: https://github.com/pnpm/pnpm

## Summary

| Feature | uv (Python) | pnpm (Node.js) |
|---------|-------------|----------------|
| Speed | 10-100x faster | 3x faster |
| Disk usage | Efficient cache | 50% less |
| Lock file | uv.lock | pnpm-lock.yaml |
| Compatible | pip commands | npm commands |
| Installation | `uv pip install` | `pnpm install` |
| Run scripts | `uv run` | `pnpm run` |

**Recommendation:** Use both for best development experience!
