#!/bin/bash

set -e

echo "========================================="
echo "Fox Benchmark - PaddleOCR-VL via OpenAI-Compatible Server"
echo "========================================="

# Requires an OpenAI-compatible server (e.g., vLLM serve).
# Set these env vars before running if not already configured:
: "${VLLM_OPENAI_BASE_URL:=http://localhost:8000/v1}"
: "${VLLM_OPENAI_API_KEY:=EMPTY}"
export VLLM_OPENAI_BASE_URL VLLM_OPENAI_API_KEY

# Configuration
TASKS=("page" "box" "line" "onbox")
MODEL_NAME="PaddlePaddle/PaddleOCR-VL"  # or server-specific model id
SYSTEM_PROMPT="./default_system_prompt.txt"

# Generation settings (used by server if supported)
MAX_MODEL_LEN=8192
MAX_NEW_TOKENS=8192
BATCH_SIZE=4

echo "Available tasks: ${TASKS[@]}"
echo "Processing all tasks..."
echo ""

for task in "${TASKS[@]}"; do
    echo "========================================="
    echo "Processing task: $task"
    echo "========================================="

    GT_FILE="./focus_benchmark_test/en_${task}_ocr.json"
    if [ "$task" = "onbox" ]; then
        IMAGE_DIR="./focus_benchmark_test/en_pdf_png_onbox/"
    else
        IMAGE_DIR="./focus_benchmark_test/en_pdf_png/"
    fi

    OUTPUT_FILE="./results/paddleocr_vl_en_${task}_ocr_openai.json"

    # Build the command using Fox eval CLI in OpenAI backend mode
    CMD="python fox_eval.py \
        --task $task \
        --model-name \"$MODEL_NAME\" \
        --use_openai_api \
        --gtfile_path \"$GT_FILE\" \
        --image_path \"$IMAGE_DIR\" \
        --system_prompt_path \"$SYSTEM_PROMPT\" \
        --out_file \"$OUTPUT_FILE\" \
        --coord_mode relative \
        --max_model_len $MAX_MODEL_LEN \
        --max_new_tokens $MAX_NEW_TOKENS \
        --batch_size $BATCH_SIZE \
        --limit_mm_video 0 \
        --skip_mm_profiling"

    echo "Executing: $CMD"
    echo ""
    eval $CMD

    echo ""
    echo "========================================="
    echo "OpenAI-compatible evaluation completed!"
    echo "Results saved to: $OUTPUT_FILE"
    echo ""

    echo "Computing evaluation metrics..."
    python eval_tools/eval_ocr_test.py --out_file "$OUTPUT_FILE"
    echo "Metrics computation completed."
    echo "========================================="
    echo ""

done
