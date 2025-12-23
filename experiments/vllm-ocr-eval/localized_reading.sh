#!/bin/bash
# Localized reading evaluation script
# Configure these variables for your environment

LIMIT=0
MAX_MODEL_LEN=32768
MAX_TOKENS=16384

# Data shards to evaluate
SHARD_PATHS=("pubmed.tar" "idl.tar" "synthdog_grounding.tar" "tabmepp.tar")
SHARD_BASENAMES=("pubmed" "idl" "synthdog_grounding" "tabmepp")

# Model configurations - add your model paths here
MODEL_NAMES=(
    "Qwen/Qwen2.5-VL-3B-Instruct"
)

# Output directory base - predictions will be saved here
MODEL_DIR_BASE="${MODEL_DIR_BASE:-./predictions}"

# Output directories corresponding to each model (must match MODEL_NAMES length)
MODEL_DIRS=(
    "$MODEL_DIR_BASE/qwen25vl3b"
)

# Loop over all shard paths and basenames
for shard_idx in "${!SHARD_PATHS[@]}"; do
    SHARD_PATH="${SHARD_PATHS[$shard_idx]}"
    SHARD_BASENAME="${SHARD_BASENAMES[$shard_idx]}"
    
    echo "Processing shard: $SHARD_PATH (basename: $SHARD_BASENAME)"

    for idx in "${!MODEL_NAMES[@]}"; do
        MODEL_NAME="${MODEL_NAMES[$idx]}"
        MODEL_DIR="${MODEL_DIRS[$idx]}"

        echo "Evaluating model: $MODEL_NAME"
        echo "Saving results to: $MODEL_DIR"

        mkdir -p $MODEL_DIR

        # Define parallel arrays for task types and output types (localized reading with fallback to latex)
        TASK_TYPES=("localized_reading")
        OUTPUT_TYPES=("lines")
        OUTPUT_NAMES=("localized-reading-lines")

        # Loop through the combinations
        for i in "${!TASK_TYPES[@]}"; do
            task_type="${TASK_TYPES[$i]}"
            output_type="${OUTPUT_TYPES[$i]}"
            output_name="${OUTPUT_NAMES[$i]}"
            
            echo "Running evaluation for task: $task_type, output: $output_type"
            
            CSV_FILE="$MODEL_DIR/${SHARD_BASENAME}-${output_name}-results.csv"
            
            uv run python run_evaluation.py \
                --model-name "$MODEL_NAME" \
                --max-model-len $MAX_MODEL_LEN \
                --shard-path "$SHARD_PATH" \
                --limit $LIMIT \
                --task-types "$task_type" \
                --output-types "$output_type" \
                --csv-output "$CSV_FILE"
        done
    done
done

echo "Localized reading evaluation and scoring completed!"
