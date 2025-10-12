#!/bin/bash

# LoRA Fine-tuning Script for ELECTRA-based Jailbreak Detection Model
# This script runs training with LoRA (Low-Rank Adaptation) for efficient fine-tuning

set -e  # Exit on any error

# Configuration
CONFIG_PATH="config/config.yaml"
LOG_DIR="experiments/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/lora_training_${TIMESTAMP}.log"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "Starting LoRA Fine-tuning"
echo "Timestamp: ${TIMESTAMP}"
echo "Config: ${CONFIG_PATH}"
echo "Log: ${LOG_FILE}"
echo "========================================"

# Run training with LoRA
uv run python src/train.py \
    --config "${CONFIG_PATH}" \
    --lora \
    2>&1 | tee "${LOG_FILE}"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "========================================"
    echo "LoRA training completed successfully!"
    echo "Log saved to: ${LOG_FILE}"
    echo "========================================"
else
    echo "========================================"
    echo "LoRA training failed!"
    echo "Check log file: ${LOG_FILE}"
    echo "========================================"
    exit 1
fi
