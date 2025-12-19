#!/usr/bin/env bash
# Sequential compression of IDL directories into tar.gz archives
#
# Environment variables:
#   PREFIX  - Path prefix for directories (REQUIRED, e.g., /path/to/idl-train)
#   WIDTH   - Zero-padding width for directory numbers (default: 5)
#   DELETE_AFTER_COMPRESS - Set to 'false' to keep original directories (default: true)
#
# Usage:
#   PREFIX=/path/to/idl-train ./compress.sh 1 100
#   PREFIX=/path/to/idl-train DELETE_AFTER_COMPRESS=false ./compress.sh 1 100

set -euo pipefail

if [[ -z "${PREFIX:-}" ]]; then
  echo "Error: PREFIX environment variable is required." >&2
  echo "Example: PREFIX=/path/to/idl-train ./compress.sh 1 100" >&2
  exit 1
fi

WIDTH="${WIDTH:-5}"
DELETE_AFTER_COMPRESS="${DELETE_AFTER_COMPRESS:-true}"

if [[ $# -ne 2 ]]; then
  echo "Usage: PREFIX=/path/to/prefix $0 LO HI"
  echo "Example: PREFIX=/data/idl-train $0 1 100"
  echo "Environment variables:"
  echo "  PREFIX (required): ${PREFIX}"
  echo "  WIDTH (default: 5): ${WIDTH}"
  echo "  DELETE_AFTER_COMPRESS (default: true): ${DELETE_AFTER_COMPRESS}"
  exit 1
fi

lo="$1"
hi="$2"

# basic validation
if ! [[ "$lo" =~ ^[0-9]+$ && "$hi" =~ ^[0-9]+$ ]]; then
  echo "Error: LO and HI must be integers." >&2
  exit 1
fi
if (( lo > hi )); then
  echo "Error: LO must be <= HI." >&2
  exit 1
fi

for ((i = lo; i <= hi; i++)); do
  num=$(printf "%0${WIDTH}d" "$i")
  dir="${PREFIX}-${num}"
  out="${dir}.tar.gz"

  if [[ ! -d "$dir" ]]; then
    echo "Skip: directory '${dir}' not found."
    continue
  fi
  if [[ -f "$out" ]]; then
    echo "Skip: archive '${out}' already exists."
    continue
  fi

  echo "Compressing ${dir} -> ${out}"
  tar czvf "$out" "$dir"

  if [[ "${DELETE_AFTER_COMPRESS}" == "true" ]]; then
    echo "Removing directory ${dir}"
    rm -rf "$dir"
  fi

done
