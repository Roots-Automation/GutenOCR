#!/usr/bin/env bash
# Reproduce GroundingBench v1 — PubMed tasks 1–4.
#
# Usage:
#   WORK_DIR=/mnt/research bash runs/v1-pubmed.sh
#
# Required environment variables:
#   WORK_DIR     Root directory for data and outputs (default: /mnt/research)
#   PUBMED_TAR   Path to the pubmed tar file
#                (default: $WORK_DIR/data/public/gutenocr-test/pubmed.tar)

set -euo pipefail

WORK_DIR=${WORK_DIR:-/mnt/research}
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # benchmarks/grounding-bench
PUBMED_TAR=${PUBMED_TAR:-$WORK_DIR/data/public/gutenocr-test/pubmed.tar}
STAGING=$WORK_DIR/grounding-bench/staging/pubmed
RANKINGS=$REPO_DIR/v1/rankings.csv
OUT=$WORK_DIR/grounding-bench/v1

# ---------------------------------------------------------------------------
# 1. Extract pubmed tar into staging
# ---------------------------------------------------------------------------
mkdir -p "$STAGING"
tar -xf "$PUBMED_TAR" -C "$STAGING"

# ---------------------------------------------------------------------------
# 2. Rank all images by visual diversity
# ---------------------------------------------------------------------------
cd "$REPO_DIR/diversity"
uv run python rank.py "$STAGING" "$RANKINGS"

# ---------------------------------------------------------------------------
# 3. Assign tasks
# ---------------------------------------------------------------------------
cd "$REPO_DIR"
python build.py assign "$STAGING" "$RANKINGS" --per-task 100 --seed 42

# ---------------------------------------------------------------------------
# 4. Populate tasks
# ---------------------------------------------------------------------------
mkdir -p "$OUT/t1-pubmed-100"
cd "$REPO_DIR"
python build.py sample "$STAGING" "$RANKINGS" "$OUT/t1-pubmed-100" --task 1

mkdir -p "$OUT/t2-pubmed-100"
python build.py sample "$STAGING" "$RANKINGS" "$OUT/t2-pubmed-100" --task 2

mkdir -p "$OUT/t3-pubmed-100"
python build.py sample "$STAGING" "$RANKINGS" "$OUT/t3-pubmed-100" --task 3

mkdir -p "$OUT/t4-pubmed-100"
python build.py sample "$STAGING" "$RANKINGS" "$OUT/t4-pubmed-100" --task 4

echo ""
echo "v1 build complete."
echo "  Task 1 output: $OUT/t1-pubmed-100"
echo "  Task 2 output: $OUT/t2-pubmed-100"
echo "  Task 3 output: $OUT/t3-pubmed-100"
echo "  Task 4 output: $OUT/t4-pubmed-100"
