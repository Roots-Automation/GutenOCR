#!/bin/bash
# Alternative approach using GNU parallel for maximum efficiency
# This can be even faster than Python for pure downloading tasks

set -euo pipefail

# Configuration
CSV_FILE="oa_non_comm_use_pdf.csv"
OUTPUT_DIR="${OUTPUT_DIR:-./pubmed_data}"
BASE_URL="https://ftp.ncbi.nlm.nih.gov/pub/pmc/"
MAX_JOBS=4
LOG_DIR="$OUTPUT_DIR/logs"

# Create directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Function to download a single file
download_file() {
    local rel_url="$1"
    local full_url="${BASE_URL}${rel_url}"
    local output_path="${OUTPUT_DIR}/${rel_url}"
    local output_dir=$(dirname "$output_path")
    
    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"
    
    # Skip if file already exists and is not empty
    if [[ -f "$output_path" && -s "$output_path" ]]; then
        echo "SKIP: $rel_url (already exists)"
        return 0
    fi
    
    # Download with wget, multiple retries, and proper error handling
    if wget -q --timeout=30 --tries=3 --retry-connrefused \
            --user-agent="PubMed-Downloader/1.0" \
            --output-document="$output_path.tmp" "$full_url" 2>/dev/null; then
        mv "$output_path.tmp" "$output_path"
        echo "OK: $rel_url"
    else
        rm -f "$output_path.tmp"
        echo "FAIL: $rel_url" >&2
        return 1
    fi
}

# Export function for parallel
export -f download_file
export BASE_URL OUTPUT_DIR

# Extract URLs from CSV (skip header) and run downloads in parallel
tail -n +2 "$CSV_FILE" | \
    cut -d',' -f1 | \
    parallel -j "$MAX_JOBS" \
             --bar \
             --joblog "$LOG_DIR/parallel.log" \
             --resume-failed \
             --retries 2 \
             download_file {} \
    > "$LOG_DIR/success.log" \
    2> "$LOG_DIR/failed.log"

echo "Download completed. Check logs in $LOG_DIR"
