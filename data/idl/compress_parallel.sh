#!/usr/bin/env bash
# Parallel compression script using GNU parallel for maximum efficiency
# Each compression job is independent, so we can run many in parallel
#
# Environment variables:
#   PREFIX  - Path prefix for directories (REQUIRED, e.g., /path/to/idl-train)
#   WIDTH   - Zero-padding width for directory numbers (default: 5)
#   MAX_JOBS - Maximum parallel jobs (default: 80)
#   LOG_DIR - Directory for log files (default: <PREFIX parent>/logs)
#   DELETE_AFTER_COMPRESS - Set to 'false' to keep original directories (default: true)
#
# Usage:
#   PREFIX=/path/to/idl-train ./compress_parallel.sh 1 100
#   PREFIX=/path/to/idl-train DELETE_AFTER_COMPRESS=false ./compress_parallel.sh 1 100

set -euo pipefail

if [[ -z "${PREFIX:-}" ]]; then
  echo "Error: PREFIX environment variable is required." >&2
  echo "Example: PREFIX=/path/to/idl-train ./compress_parallel.sh 1 100" >&2
  exit 1
fi

WIDTH="${WIDTH:-5}"
MAX_JOBS="${MAX_JOBS:-80}"  # Adjust based on your system's capabilities
LOG_DIR="${LOG_DIR:-$(dirname "$PREFIX")/logs}"
DELETE_AFTER_COMPRESS="${DELETE_AFTER_COMPRESS:-true}"

if [[ $# -ne 2 ]]; then
  echo "Usage: PREFIX=/path/to/prefix $0 LO HI"
  echo "Example: PREFIX=/data/idl-train $0 1 100"
  echo "Environment variables:"
  echo "  PREFIX (required): ${PREFIX}"
  echo "  WIDTH (default: 5): ${WIDTH}"
  echo "  MAX_JOBS (default: 80): ${MAX_JOBS}"
  echo "  LOG_DIR: ${LOG_DIR}"
  echo "  DELETE_AFTER_COMPRESS (default: true): ${DELETE_AFTER_COMPRESS}"
  exit 1
fi

lo="$1"
hi="$2"

# Basic validation
if ! [[ "$lo" =~ ^[0-9]+$ && "$hi" =~ ^[0-9]+$ ]]; then
  echo "Error: LO and HI must be integers." >&2
  exit 1
fi
if (( lo > hi )); then
  echo "Error: LO must be <= HI." >&2
  exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Function to compress a single directory
compress_dir() {
    local i="$1"
    local num=$(printf "%0${WIDTH}d" "$i")
    local dir="${PREFIX}-${num}"
    local out="${dir}.tar.gz"

    # Check if directory exists
    if [[ ! -d "$dir" ]]; then
        echo "SKIP: directory '${dir}' not found (index: $i)"
        return 0
    fi

    # Check if archive already exists
    if [[ -f "$out" ]]; then
        echo "SKIP: archive '${out}' already exists (index: $i)"
        return 0
    fi

    # Compress the directory
    echo "START: Compressing ${dir} -> ${out} (index: $i)"
    if tar czf "$out" "$dir" 2>/dev/null; then
        echo "COMPRESSED: ${out} (index: $i)"
        
        # Remove the original directory after successful compression (if enabled)
        if [[ "${DELETE_AFTER_COMPRESS}" == "true" ]]; then
            rm -rf "$dir"
            echo "REMOVED: ${dir} (index: $i)"
        fi
        echo "OK: $i"
    else
        echo "FAIL: Failed to compress ${dir} (index: $i)" >&2
        return 1
    fi
}

# Export function and variables for parallel
export -f compress_dir
export PREFIX WIDTH DELETE_AFTER_COMPRESS

# Generate sequence and run compressions in parallel
echo "Starting parallel compression of directories ${lo} to ${hi}"
echo "Using ${MAX_JOBS} parallel jobs, logging to ${LOG_DIR}"

seq "$lo" "$hi" | \
    parallel -j "$MAX_JOBS" \
             --bar \
             --joblog "$LOG_DIR/compression_parallel.log" \
             --resume-failed \
             --retries 1 \
             compress_dir {} \
    > "$LOG_DIR/compression_success.log" \
    2> "$LOG_DIR/compression_failed.log"

echo "Compression completed. Check logs in $LOG_DIR"
echo "Success log: $LOG_DIR/compression_success.log"
echo "Failed log: $LOG_DIR/compression_failed.log"
echo "Job log: $LOG_DIR/compression_parallel.log"
