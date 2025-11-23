# Backend-specific justfile
# Can be used from backend directory: just -f .justfile <recipe>

# Install with uv
install:
    uv pip install -e .

# Install with dev dependencies
install-dev:
    uv pip install -e ".[dev]"

# Sync with uv.lock
sync:
    uv sync

# Run the API server
serve:
    uv run python serve.py

# Train the model
train:
    uv run python train.py --epochs 50

# Run tests
test:
    uv run pytest tests/ -v

# Test with coverage
test-cov:
    uv run pytest tests/ --cov=arcx --cov-report=html --cov-report=term-missing

# Format code
format:
    uv run black arcx/ tests/

# Lint code
lint:
    uv run ruff check arcx/ tests/

# Type check
typecheck:
    uv run mypy arcx/

# Check device
check-device:
    uv run python -c "from arcx.device import device_manager; print(device_manager)"

# Test encoder
test-encoder:
    uv run python -c "from arcx.ml.encoder import test_encoder; test_encoder()"

# Test qnet
test-qnet:
    uv run python -c "from arcx.ml.qnet import test_qnet; test_qnet()"

# Test model
test-model:
    uv run python -c "from arcx.ml.model import test_evmodel; test_evmodel()"

# Interactive Python shell with imports
shell:
    uv run python -i -c "from arcx.ml.model import EVModel; from arcx.device import device_manager; from arcx.config import config; print('Imports ready: EVModel, device_manager, config')"

# Lock dependencies
lock:
    uv lock

# Update dependencies
update:
    uv lock --upgrade
