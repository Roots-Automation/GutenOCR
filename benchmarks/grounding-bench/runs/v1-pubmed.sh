#!/usr/bin/env bash
# Reproduce GroundingBench v1 — PubMed tasks 1–4.
#
# Usage:
#   WORK_DIR=/mnt/research bash runs/v1-pubmed.sh
#
# Environment variables:
#   WORK_DIR     Root directory for data and outputs (default: /mnt/research)
#   PUBMED_TAR   Path to the pubmed tar file
#                (default: $WORK_DIR/data/public/gutenocr-test/pubmed.tar)

set -euo pipefail

# Root of all data and generated outputs on this machine.
WORK_DIR=${WORK_DIR:-/mnt/research}

# Absolute path to benchmarks/grounding-bench/ — used to locate repo scripts.
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Source archive of PubMed pages to build the benchmark from.
PUBMED_TAR=${PUBMED_TAR:-$WORK_DIR/data/public/gutenocr-test/pubmed.tar}

# Temporary directory where the tar is unpacked before sampling.
STAGING=$WORK_DIR/grounding-bench/staging/pubmed

# Root output directory for the v1 benchmark: rankings, task splits, etc.
OUT=$WORK_DIR/grounding-bench/v1

# CSV recording each image's diversity rank and task assignment (1–4).
# Written by step 2, consumed by steps 3 and 4.
RANKINGS=$OUT/rankings.csv

# ---------------------------------------------------------------------------
# 1. Extract pubmed tar into staging
# ---------------------------------------------------------------------------
mkdir -p "$STAGING"
echo "[1/4] Extracting $PUBMED_TAR -> $STAGING ..."
tar --touch -xf "$PUBMED_TAR" -C "$STAGING"
echo "[1/4] Extraction complete."

# ---------------------------------------------------------------------------
# 2. Rank all images by visual diversity
# ---------------------------------------------------------------------------
mkdir -p "$OUT"
cd "$REPO_DIR"
echo "[2/4] Ranking images by visual diversity ..."
uv run python3 -m diversity.rank "$STAGING" "$RANKINGS"
echo "[2/4] Ranking complete. Rankings written to $RANKINGS"

# ---------------------------------------------------------------------------
# 3. Assign tasks
# ---------------------------------------------------------------------------
echo "[3/4] Assigning images to tasks ..."
uv run python3 build.py assign "$STAGING" "$RANKINGS" --per-task 100 --seed 42
echo "[3/4] Task assignment complete."

# ---------------------------------------------------------------------------
# 4. Populate tasks
# ---------------------------------------------------------------------------
echo "[4/4] Populating task directories ..."

echo "  [4/4] Sampling task 1 ..."
mkdir -p "$OUT/t1-pubmed-100"
uv run python3 build.py sample "$STAGING" "$RANKINGS" "$OUT/t1-pubmed-100" --task 1

echo "  [4/4] Sampling task 2 ..."
mkdir -p "$OUT/t2-pubmed-100"
uv run python3 build.py sample "$STAGING" "$RANKINGS" "$OUT/t2-pubmed-100" --task 2

echo "  [4/4] Sampling task 3 ..."
mkdir -p "$OUT/t3-pubmed-100"
uv run python3 build.py sample "$STAGING" "$RANKINGS" "$OUT/t3-pubmed-100" --task 3

echo "  [4/4] Sampling task 4 ..."
mkdir -p "$OUT/t4-pubmed-100"
uv run python3 build.py sample "$STAGING" "$RANKINGS" "$OUT/t4-pubmed-100" --task 4

echo "[4/4] All tasks populated."

echo ""
echo "v1 build complete."
echo "  Rankings:       $RANKINGS"
echo "  Task 1 output:  $OUT/t1-pubmed-100"
echo "  Task 2 output:  $OUT/t2-pubmed-100"
echo "  Task 3 output:  $OUT/t3-pubmed-100"
echo "  Task 4 output:  $OUT/t4-pubmed-100"
