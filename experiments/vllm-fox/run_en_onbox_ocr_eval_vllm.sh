#!/bin/bash

# Fox Benchmark - English On-Box OCR Evaluation with vLLM
# Usage: ./run_en_onbox_ocr_eval_vllm.sh

set -e

echo "========================================="
echo "Fox Benchmark - English On-Box OCR Evaluation (vLLM)"
echo "========================================="

# =============================================================================
# MODEL CONFIGURATION
# Uncomment ONE model configuration below, or set your own.
# =============================================================================

# --- Local vLLM (default) ---
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_FILE="./results_qwen25_3b_en_onbox_ocr_vllm.json"

# --- OpenAI-compatible API (e.g., vLLM serve) ---
# export VLLM_OPENAI_BASE_URL="http://localhost:8000/v1"
# export VLLM_OPENAI_API_KEY="EMPTY"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
# OUTPUT_FILE="./results_qwen25_7b_en_onbox_ocr_vllm.json"

# =============================================================================
# DATA PATHS
# =============================================================================
GT_FILE="./focus_benchmark_test/en_onbox_ocr.json"
IMAGE_DIR="./focus_benchmark_test/en_pdf_png_onbox/"
SYSTEM_PROMPT="./default_system_prompt.txt"

# =============================================================================
# VLLM SETTINGS
# =============================================================================
MAX_MODEL_LEN=32768
MAX_NEW_TOKENS=4096
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9

# =============================================================================
# VALIDATION
# =============================================================================
echo "Checking required files..."

if [ ! -f "$GT_FILE" ]; then
    echo "Error: Ground truth file not found: $GT_FILE"
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Image directory not found: $IMAGE_DIR"
    exit 1
fi

if [ ! -f "$SYSTEM_PROMPT" ]; then
    echo "Error: System prompt file not found: $SYSTEM_PROMPT"
    exit 1
fi

echo "✓ Ground truth: $GT_FILE"
echo "✓ Images: $IMAGE_DIR"
echo "✓ System prompt: $SYSTEM_PROMPT"
echo ""

# =============================================================================
# RUN EVALUATION
# =============================================================================
echo "vLLM Configuration:"
echo "- Model: $MODEL_NAME"
echo "- Max model length: $MAX_MODEL_LEN"
echo "- Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "- GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo ""

python fox_eval.py \
    --task onbox \
    --model-name "$MODEL_NAME" \
    --gtfile_path "$GT_FILE" \
    --image_path "$IMAGE_DIR" \
    --system_prompt_path "$SYSTEM_PROMPT" \
    --out_file "$OUTPUT_FILE" \
    --max_model_len $MAX_MODEL_LEN \
    --max_new_tokens $MAX_NEW_TOKENS \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --gpu_memory_utilization $GPU_MEMORY_UTILIZATION

echo ""
echo "========================================="
echo "Evaluation completed. Results: $OUTPUT_FILE"
echo "========================================="

# Compute metrics
if [ -f "eval_tools/eval_ocr_test.py" ]; then
    echo "Computing evaluation metrics..."
    python eval_tools/eval_ocr_test.py --out_file "$OUTPUT_FILE"
fi

echo "========================================="
