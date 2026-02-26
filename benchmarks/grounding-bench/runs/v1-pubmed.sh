#!/usr/bin/env bash
# Reproduce GroundingBench v1 — PubMed task 1 (grounded line reading).
#
# Usage:
#   bash runs/v1-pubmed.sh [WORK_DIR]
#
# WORK_DIR defaults to /mnt/research and must contain:
#   data/public/gutenocr-test/pubmed.tar
#
# Rankings and task assignments are already committed to the repo, so steps 2
# and 3 are commented out. Re-running this script produces the same output.

set -euo pipefail

WORK_DIR=${1:-/mnt/research}
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # benchmarks/grounding-bench
PUBMED_TAR=$WORK_DIR/data/public/gutenocr-test/pubmed.tar
STAGING=$WORK_DIR/grounding-bench/staging/pubmed
RANKINGS=$REPO_DIR/diversity/rankings.csv
OUT=$WORK_DIR/grounding-bench/v1

# ---------------------------------------------------------------------------
# 1. Extract pubmed tar into staging
# ---------------------------------------------------------------------------
mkdir -p "$STAGING"
tar -xf "$PUBMED_TAR" -C "$STAGING"

# ---------------------------------------------------------------------------
# 2. Rank all images by visual diversity
#    (skip — rankings.csv already committed for v1)
# ---------------------------------------------------------------------------
# cd "$REPO_DIR/diversity"
# uv run python rank.py "$STAGING" "$RANKINGS"

# ---------------------------------------------------------------------------
# 3. Assign tasks
#    (skip — task column already populated in rankings.csv for v1)
# ---------------------------------------------------------------------------
# cd "$REPO_DIR"
# python build.py assign "$STAGING" "$RANKINGS" --per-task 100 --seed 42

# ---------------------------------------------------------------------------
# 4. Populate task 1: full-page grounded line reading
# ---------------------------------------------------------------------------
mkdir -p "$OUT/t1-pubmed-100"
cd "$REPO_DIR"
python build.py sample "$STAGING" "$RANKINGS" "$OUT/t1-pubmed-100" --task 1

# Future tasks (uncomment when ready):
# python build.py sample "$STAGING" "$RANKINGS" "$OUT/t2-pubmed-100" --task 2
# python build.py sample "$STAGING" "$RANKINGS" "$OUT/t3-pubmed-100" --task 3
# python build.py sample "$STAGING" "$RANKINGS" "$OUT/t4-pubmed-100" --task 4

echo ""
echo "v1 build complete."
echo "  Task 1 output: $OUT/t1-pubmed-100"
