#!/bin/bash
#
# Simple SynthDoG Statistics Processor
#
# A lightweight version of the batch statistics processor that focuses on
# processing only the remaining tar files that don't have statistics yet.
# This script is useful for resuming interrupted batch processing operations.
#
# Features:
# - Quick check for existing stats files
# - Processes only missing statistics
# - Minimal logging for faster execution
# - Progress reporting every 20 files
#
# Usage:
#   ./simple_batch_process.sh
#
# Output:
#   Creates .stats.csv files for tar files that don't have them yet
#
# Requirements:
#   - generate_stats.py in the same directory
#   - uv package manager
#
# Environment Variables:
#   SYNTHDOG_DATA_DIR - Override the default dataset directory
#

# Configuration - use arg, then env var, then default
DATASET_DIR="${1:-${SYNTHDOG_DATA_DIR:-./outputs}}"
SCRIPT_DIR="$(dirname "$0")"
PROGRESS_INTERVAL=20

echo "Simple SynthDoG Statistics Processor"
echo "===================================="
echo "Dataset directory: $DATASET_DIR"
echo "Processing remaining tar files..."
echo ""

# Change to script directory to ensure relative imports work
cd "$SCRIPT_DIR" || exit 1

# Initialize counters
count=0
processed=0
skipped=0

# Process each tar file
for tar_file in "$DATASET_DIR"/*.tar; do
    count=$((count + 1))
    filename=$(basename "$tar_file")
    stats_file="${tar_file%.tar}.stats.csv"
    
    # Check if stats file already exists
    if [[ -f "$stats_file" ]]; then
        echo "[$count] $filename → SKIP (stats exist)"
        skipped=$((skipped + 1))
    else
        echo "[$count] $filename → PROCESSING..."
        if uv run generate_stats.py "$tar_file"; then
            echo "  ✓ SUCCESS"
            processed=$((processed + 1))
        else
            echo "  ✗ FAILED"
        fi
    fi
    
    # Progress update
    if (( count % PROGRESS_INTERVAL == 0 )); then
        echo ""
        echo "--- Progress Update ---"
        echo "Files checked: $count"
        echo "Newly processed: $processed"
        echo "Already existed: $skipped"
        echo "Completion rate: $(( (processed + skipped) * 100 / count ))%"
        echo ""
    fi
done

echo ""
echo "=== PROCESSING COMPLETE ==="
echo "Total checked: $count"
echo "Newly processed: $processed" 
echo "Already existed: $skipped"
if [ $((count - skipped)) -gt 0 ]; then
    echo "Success rate: $(( processed * 100 / (count - skipped) ))%"
fi
