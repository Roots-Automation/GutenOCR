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

**v1.0** covers a single task: **full-page grounded line reading**.

- Task: `reading`
- Input: full-page document image
- Output: one entry per line — `{"text": "...", "bbox": [x1, y1, x2, y2]}`

Bounding boxes are in absolute pixel coordinates `[x1, y1, x2, y2]` with the origin at the top-left corner.

Future releases will add detection-only, paragraph reading, conditional detection, and localized reading — as well as additional tasks not yet defined in the original GutenOCR paper.

---

## v1.0 Scope

- **100 samples** from PubMed Open Access
- Task: full-page grounded line reading
- Documents: CC-licensed academic articles not used during GutenOCR training
- Every sample has been **manually verified** by a human annotation team for annotation correctness

Future releases will incorporate samples from additional public document datasets to broaden layout and domain coverage.

---

## Data Format

The benchmark is a flat directory of paired files:

```
output/
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

## Diversity Sampling

Benchmark samples are selected by visual diversity rather than random sampling. This ensures the 100 samples span the full visual space of the PubMed test pool — covering narrow single-column papers, wide multi-column layouts, tables, figures, and mixed pages — rather than clustering around the most common document style.

**Method:** SigLIP visual embeddings + k-center greedy algorithm

1. Each candidate page is embedded with [SigLIP](https://huggingface.co/google/siglip2-so400m-patch16-naflex).
2. The k-center greedy algorithm iteratively picks the image most distant from all previously selected images (measured by cosine distance in embedding space).
3. The result is ranked by decreasing diversity contribution.

---

## How to Reproduce

These scripts produce the benchmark dataset locally. The benchmark data is not hosted in this repository.

### 1. Install dependencies

```bash
# benchmarks/grounding-bench/diversity
cd benchmarks/grounding-bench/diversity
uv sync
```

### 2. Rank all images by visual diversity

```bash
# benchmarks/grounding-bench/diversity
uv run python rank.py /data/pubmed rankings.csv
```

This embeds every image in the test pool with SigLIP and ranks them by diversity. For a quick smoke-test, add `--limit 50` to process only 50 images.

### 3. (Optional) Inspect the diversity curve

```bash
# benchmarks/grounding-bench/diversity
uv run python analyze_threshold.py rankings.csv
```

Prints recommended sampling thresholds (elbow point, 50%/80%/90% coverage) to help choose `--top-k`.

### 4. Extract the top-100 diverse samples

```bash
# benchmarks/grounding-bench
cd ..
python sample.py /data/pubmed diversity/rankings.csv ./output --top-k 100
```

`sample.py` (in `benchmarks/grounding-bench/`) reads the rankings produced by `diversity/rank.py`, skips any sample whose `text.lines` annotation is missing or empty, and copies the image + JSON pair to `./output`. It stops when 100 valid samples have been collected and prints a summary of how many were skipped.

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

