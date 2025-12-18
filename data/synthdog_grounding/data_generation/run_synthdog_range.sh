#!/usr/bin/env bash
#
# SynthDoG Range Generator
#
# This script generates synthetic document data using SynthDoG for a specified range of directory IDs.
# Each directory contains approximately 16,784 synthetic document samples with grounding annotations.
#
# Usage:
#   ./run_synthdog_range.sh [START] [END]
#
# Arguments:
#   START   Starting directory ID (default: 35)
#   END     Ending directory ID (default: 75)
#
# Examples:
#   ./run_synthdog_range.sh          # Generate directories 0035-0075
#   ./run_synthdog_range.sh 10 20    # Generate directories 0010-0020
#
# Output:
#   Creates numbered directories (e.g., 0035, 0036, ...) in the base output directory.
#   Each directory contains:
#   - Image files (.jpg): Synthetic document images
#   - Annotation files (.json): Grounding annotations with text and bounding boxes
#
# Configuration:
#   - Uses config_en-pdfs.yaml for English PDF-style documents
#   - Generates 16,784 samples per directory with 128 workers
#   - Skips directories that already contain data
#
# Environment Variables:
#   SYNTHDOG_DATA_DIR - Override the default output directory
#
# Requirements:
#   - synthtiger package installed
#   - template.py and config files in parent directory
#   - Sufficient disk space (each directory ~2-5GB)
#
set -euo pipefail

# Configuration - use env var or default to ./outputs
BASE="${SYNTHDOG_DATA_DIR:-./outputs}"
SAMPLES_PER_DIR=16784
WORKERS=128
CONFIG_FILE="../config/config_en-pdfs.yaml"

# Parse command line arguments with defaults
START="${1:-35}"
END="${2:-75}"

echo "SynthDoG Range Generator"
echo "========================"
echo "Output directory: $BASE"
echo "Range: $(printf '%04d' "$START") to $(printf '%04d' "$END")"
echo "Samples per directory: $SAMPLES_PER_DIR"
echo "Workers: $WORKERS"
echo "Config: $CONFIG_FILE"
echo ""

# Validate configuration file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Please ensure you're running this script from the data_generation directory"
    exit 1
fi

# Create base directory if it doesn't exist
mkdir -p "$BASE"

# Generate data for each directory in the range
for i in $(seq "$START" "$END"); do
    NUM="$(printf '%04d' "$i")"
    OUT="${BASE}/${NUM}"

    echo "=== [$NUM] Generating into: $OUT ==="
    mkdir -p "$OUT"

    # Skip if directory already contains data
    if [ -n "$(ls -A "$OUT" 2>/dev/null || true)" ]; then
        echo "    → Skipping $NUM (directory not empty)"
        continue
    fi

    # Generate synthetic documents
    echo "    → Generating $SAMPLES_PER_DIR samples with $WORKERS workers"
    if synthtiger -o "$OUT" -c "$SAMPLES_PER_DIR" -w "$WORKERS" -v ../template.py SynthDoG "$CONFIG_FILE"; then
        echo "    → Successfully generated data for $NUM"
    else
        echo "    → Error generating data for $NUM"
        exit 1
    fi
done

echo ""
echo "=== Generation Complete ==="
echo "Successfully generated directories: $(printf '%04d' "$START") to $(printf '%04d' "$END")"
