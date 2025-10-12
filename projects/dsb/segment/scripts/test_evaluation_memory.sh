#!/bin/bash

# Test script for memory-efficient evaluation
# This script runs evaluation with very small chunks to test memory management

set -e

echo "========================================"
echo "Testing Memory-Efficient Evaluation"
echo "========================================"

# Test with very small chunk size first
CONFIG_PATH="config/config.yaml"
MODEL_PATH="models/checkpoints/final_model"
OUTPUT_DIR="experiments/logs/test_memory_eval"
BATCH_SIZE=1
CHUNK_SIZE=10  # Very small for testing

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Testing with chunk_size=${CHUNK_SIZE}, batch_size=${BATCH_SIZE}"

# Run memory-efficient evaluation
python src/evaluate_memory_efficient.py \
    --model_path "${MODEL_PATH}" \
    --config "${CONFIG_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --chunk_size "${CHUNK_SIZE}" \
    --debug

if [ $? -eq 0 ]; then
    echo "========================================"
    echo "Memory-efficient test passed!"
    echo "Results saved to: ${OUTPUT_DIR}"
    echo "========================================"
else
    echo "========================================"
    echo "Memory-efficient test failed!"
    echo "========================================"
    exit 1
fi