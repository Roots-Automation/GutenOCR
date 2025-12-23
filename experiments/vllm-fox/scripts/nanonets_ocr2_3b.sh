#!/bin/bash

set -e

echo "========================================="
echo "Fox Benchmark - English Page OCR Evaluation (vLLM Optimized)"
echo "========================================="

# Configuration
TASKS=("page" "box" "line" "onbox")

echo "Available tasks: ${TASKS[@]}"
echo "Processing all tasks..."
echo ""

for task in "${TASKS[@]}"; do
    echo "========================================="
    echo "Processing task: $task"
    echo "========================================="
    MODEL_NAME="nanonets/Nanonets-OCR2-3B"
    OUTPUT_FILE="./results/nanonets_3b_en_${task}_ocr_vllm.json"

    # Data paths
    GT_FILE="./focus_benchmark_test/en_${task}_ocr.json"
    if [ "$task" = "onbox" ]; then
        IMAGE_DIR="./focus_benchmark_test/en_pdf_png_onbox/"
    else
        IMAGE_DIR="./focus_benchmark_test/en_pdf_png/"
    fi
    SYSTEM_PROMPT="./default_system_prompt.txt"

    # vLLM settings
    MAX_MODEL_LEN=32768
    MAX_NEW_TOKENS=32768
    TENSOR_PARALLEL_SIZE=1
    GPU_MEMORY_UTILIZATION=0.9

    # Build the command using unified Fox eval CLI
    CMD="python fox_eval.py \
        --task $task \
        --model-name \"$MODEL_NAME\" \
        --gtfile_path \"$GT_FILE\" \
        --image_path \"$IMAGE_DIR\" \
        --system_prompt_path \"$SYSTEM_PROMPT\" \
        --out_file \"$OUTPUT_FILE\" \
        --max_model_len $MAX_MODEL_LEN \
        --max_new_tokens $MAX_NEW_TOKENS \
        --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
        --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
        --limit_mm_video 0 \
        --skip_mm_profiling"

    # Run the evaluation via OpenAI-compatible vLLM server
    echo "Executing: $CMD"
    echo ""
    eval $CMD

    echo ""
    echo "========================================="
    echo "vLLM Evaluation completed!"
    echo "Results saved to: $OUTPUT_FILE"
    echo ""

    # Run evaluation metrics if the eval tools are available
    echo "Computing evaluation metrics..."
    python eval_tools/eval_ocr_test.py --out_file "$OUTPUT_FILE"
    echo "Metrics computation completed."
    echo "========================================="
    echo ""
done
