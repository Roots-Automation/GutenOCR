# GroundingBench

**GroundingBench** is an open benchmark for evaluating models on *grounded OCR* — the joint task of reading text and localizing it on the page. Unlike plain OCR benchmarks that only check whether text is recognized, GroundingBench requires a model to return both the text content and the bounding box for every element it reads.

---

## Task Taxonomy

GutenOCR defines four grounded OCR task families:

| Task | Input | Output |
|---|---|---|
| **reading** | Full-page image | Text transcription with bounding boxes per unit |
| **detection** | Full-page image | Bounding boxes only (no text) |
| **localized reading** | Image + region box | Text inside a given region |
| **conditional detection** | Image + text query | Boxes for regions matching the query |

Bounding boxes are in absolute pixel coordinates `[x1, y1, x2, y2]` with the origin at the top-left corner.

---

## Roadmap

Benchmark splits are named `t{N}-{dataset}-{count}` (e.g. `t1-pubmed-100`), where `N` is the task number from the taxonomy above.

**PubMed** is covered first across all four tasks, then additional datasets will be added to broaden layout and domain coverage.

| Split | Task | Status |
|---|---|---|
| `t1-pubmed-100` | reading — full-page grounded line reading | in progress |
| `t2-pubmed-100` | detection — bounding boxes only | todo |
| `t3-pubmed-100` | localized reading — text inside a given region | todo |
| `t4-pubmed-100` | conditional detection — boxes matching a text query | todo |

---

## v1.0 Scope

- **100 samples** from PubMed Open Access (`t1-pubmed-100`)
- Task: full-page grounded line reading
- Documents: CC-licensed academic articles not used during GutenOCR training
- Every sample has been **manually verified** by a human annotation team for annotation correctness

---

## Data Format

The benchmark is a flat directory of paired files:

```
t1-pubmed-100/
  {id}.png     ← document page image
  {id}.json    ← grounded annotation
```

JSON schema:

```json
{
  "text": {
    "lines": [
      {"text": "First line on the page", "box": [x1, y1, x2, y2]},
      {"text": "Second line", "box": [x1, y1, x2, y2]}
    ],
    "words": [...],
    "paragraphs": [...],
    "text": "Full document text",
    "text2d": "Layout-preserved text"
  },
  "image": {"width": 1000, "height": 1400}
}
```

> **Note:** The annotation field for bounding boxes uses `"box"` (not `"bbox"`).

---

## Diversity Sampling and Task Assignment

Benchmark samples are selected by visual diversity rather than random sampling, ensuring the selected pages span the full visual space of the PubMed test pool — covering narrow single-column papers, wide multi-column layouts, tables, figures, and mixed pages.

**Method:** SigLIP visual embeddings + k-center greedy algorithm

1. Each candidate page is embedded with [SigLIP](https://huggingface.co/google/siglip2-so400m-patch16-naflex).
2. The k-center greedy algorithm iteratively picks the image most distant from all previously selected images (measured by cosine distance in embedding space).
3. The result is ranked by decreasing diversity contribution.

**`$WORK_DIR/grounding-bench/v1/rankings.csv` is the single source of truth for the v1 benchmark.** It encodes both the diversity order (the `rank` column) and the task assignment (the `task` column, values 1–4). Task assignments were made once with a fixed random seed (42):

- Scan rankings in rank order; pre-validate each sample (check `text.lines` is non-empty).
- Collect the first 400 valid samples (4 tasks × 100).
- Shuffle the 400 indices with seed 42, then assign task 1/2/3/4 to each group of 100.

This ensures the four task splits are **mutually exclusive** and collectively cover a maximally diverse set of 400 document pages.

---

## How to Reproduce

The benchmark data is not hosted in this repository. Run `runs/v1-pubmed.sh` to produce it locally.

```bash
# From the repo root:
WORK_DIR=/mnt/research bash benchmarks/grounding-bench/runs/v1-pubmed.sh
```

You can also override the tar path explicitly:

```bash
WORK_DIR=/mnt/research PUBMED_TAR=/path/to/pubmed.tar bash benchmarks/grounding-bench/runs/v1-pubmed.sh
```

The script:
1. Extracts `pubmed.tar` from `$PUBMED_TAR` into a staging directory.
2. Ranks all staged images by visual diversity and writes `v1/rankings.csv`.
3. Assigns each of the 400 top-ranked samples to one of four tasks (seed 42).
4. Copies the 100 image+JSON pairs for each task to `$WORK_DIR/grounding-bench/v1/t{N}-pubmed-100/`.

Install dependencies before running:

```bash
cd benchmarks/grounding-bench
uv sync
```

---

## Evaluating a Model

After producing the benchmark samples, run evaluation with the `vllm-ocr-eval` experiment:

```bash
cd experiments/vllm-ocr-eval
uv run python run_evaluation.py \
    --model-name "rootsautomation/GutenOCR-3B" \
    --shard-path /path/to/benchmark.tar \
    --task-types reading \
    --output-types "[lines, box]" \
    --csv-output predictions.csv

uv run python score_lines_reading.py predictions.csv --overwrite
```
