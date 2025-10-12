#!/bin/bash

# Evaluation Script for ELECTRA-based Jailbreak Detection Model
# This script evaluates a trained model and generates comprehensive reports

set -e  # Exit on any error

# Default configuration
CONFIG_PATH="config/config.yaml"
MODEL_PATH="models/checkpoints/final_model"
OUTPUT_DIR=""
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_SIZE=1
CHUNK_SIZE=100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --chunk_size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model_path PATH    Path to trained model (default: models/checkpoints/final_model)"
            echo "  --config PATH        Path to configuration file (default: config/config.yaml)"
            echo "  --output_dir PATH    Output directory for evaluation results"
            echo "  --batch_size INT     Batch size for evaluation (default: 1)"
            echo "  --chunk_size INT     Number of samples to process at once (default: 100)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default output directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="experiments/logs/evaluation_${TIMESTAMP}"
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "Starting Model Evaluation"
echo "Timestamp: ${TIMESTAMP}"
echo "Model: ${MODEL_PATH}"
echo "Config: ${CONFIG_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================"

# Check if model exists
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: Model path does not exist: ${MODEL_PATH}"
    exit 1
fi

# Check if config exists
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Error: Configuration file does not exist: ${CONFIG_PATH}"
    exit 1
fi

# Run evaluation - use memory efficient version
python src/evaluate_memory_efficient.py \
    --model_path "${MODEL_PATH}" \
    --config "${CONFIG_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --chunk_size "${CHUNK_SIZE}"

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo "========================================"
    echo "Evaluation completed successfully!"
    echo "Results saved to: ${OUTPUT_DIR}"
    echo ""
    echo "Generated files:"
    echo "  - evaluation_results.json"
    echo "  - confusion_matrix.png"
    echo "  - roc_curve.png"
    echo "  - precision_recall_curve.png"
    echo "  - false_predictions.json"
    echo "========================================"
else
    echo "========================================"
    echo "Evaluation failed!"
    echo "Check logs in: ${OUTPUT_DIR}"
    echo "========================================"
    exit 1
fi
