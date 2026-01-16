#!/bin/bash

# Development server script with auto-reload

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info
