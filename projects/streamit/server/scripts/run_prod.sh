#!/bin/bash

# Production server script (no auto-reload)

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-level info
