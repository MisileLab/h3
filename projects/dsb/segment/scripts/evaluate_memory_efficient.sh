#!/bin/bash

# Memory-Efficient Evaluation Script
# This script sets up environment variables for better GPU memory management
# and runs evaluation with reduced batch sizes

echo "🚀 Setting up memory-efficient evaluation environment..."

# Set PyTorch memory allocation configuration for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Optional: Limit GPU memory growth (uncomment if needed)
# export CUDA_VISIBLE_DEVICES=0

echo "📊 Memory optimization settings:"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo ""

# Default batch size (reduced for memory efficiency)
BATCH_SIZE=${1:-8}
MODEL_PATH=${2:-"./models/best_model"}
CONFIG_PATH=${3:-"./config/config.yaml"}

echo "🔧 Running evaluation with:"
echo "Batch size: $BATCH_SIZE"
echo "Model path: $MODEL_PATH"
echo "Config path: $CONFIG_PATH"
echo ""

# Run the evaluation with memory-efficient settings
python src/evaluate.py \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG_PATH" \
    --batch_size $BATCH_SIZE \
    --debug

echo ""
echo "✅ Evaluation completed!"
echo ""
echo "💡 If you still encounter OOM errors, try:"
echo "1. Reducing batch size further: ./scripts/evaluate_memory_efficient.sh 4"
echo "2. Using CPU evaluation: CUDA_VISIBLE_DEVICES='' ./scripts/evaluate_memory_efficient.sh"
echo "3. Checking for other GPU processes: nvidia-smi"