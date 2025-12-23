#!/bin/bash
# Detection scoring script
# Configure these variables for your environment

LIMIT=0
MAX_MODEL_LEN=32768
MAX_TOKENS=16384

# Data shards to score
SHARD_PATHS=("pubmed.tar" "idl.tar" "latex.tar" "synthdog_grounding.tar" "tabmepp.tar")
SHARD_BASENAMES=("pubmed" "idl" "latex" "synthdog_grounding" "tabmepp")

# Model configurations - add your model paths here
MODEL_NAMES=(
    "Qwen/Qwen2.5-VL-3B-Instruct"
)

# Output directory base - predictions should be here
MODEL_DIR_BASE="${MODEL_DIR_BASE:-./predictions}"

# Output directories corresponding to each model (must match MODEL_NAMES length)
MODEL_DIRS=(
    "$MODEL_DIR_BASE/qwen25vl3b"
)

# Loop over all shard paths and basenames
for shard_idx in "${!SHARD_PATHS[@]}"; do
    SHARD_PATH="${SHARD_PATHS[$shard_idx]}"
    SHARD_BASENAME="${SHARD_BASENAMES[$shard_idx]}"
    
    for idx in "${!MODEL_NAMES[@]}"; do
        MODEL_NAME="${MODEL_NAMES[$idx]}"
        MODEL_DIR="${MODEL_DIRS[$idx]}"

        # Define parallel arrays for task types and output types
        TASK_TYPES=("detection")
        OUTPUT_TYPES=("box")
        OUTPUT_NAMES=("detection")

        # Loop through the combinations
        for i in "${!TASK_TYPES[@]}"; do
            task_type="${TASK_TYPES[$i]}"
            output_type="${OUTPUT_TYPES[$i]}"
            output_name="${OUTPUT_NAMES[$i]}"
            
            CSV_FILE="$MODEL_DIR/${SHARD_BASENAME}-${output_name}-results.csv"
            
            # Score the detection results if CSV was created
            if [ -f "$CSV_FILE" ]; then
                echo "Scoring detection results for $CSV_FILE"
                uv run python score_detection.py "$CSV_FILE" --overwrite
            fi
        done
    done
done

echo "Detection scoring complete."
