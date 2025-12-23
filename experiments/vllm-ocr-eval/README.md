# OCR Evaluation with vLLM

Batch OCR evaluation framework using vLLM for Vision Language Models. Supports text reading, detection, localized reading, and conditional detection tasks on document images.

## Installation

```bash
uv sync
```

## Data Format

Evaluation data should be packaged as `.tar` files containing image-JSON pairs:

```
shard.tar/
├── doc001.png
├── doc001.json
├── doc002.jpg
├── doc002.json
└── ...
```

Each JSON file should contain:
```json
{
  "text": "Full document text",
  "text2d": "Text with layout whitespace preserved",
  "lines": [{"text": "Line text", "bbox": [x1, y1, x2, y2]}, ...],
  "words": [{"text": "word", "bbox": [x1, y1, x2, y2]}, ...],
  "paragraphs": [{"text": "Paragraph text", "bbox": [x1, y1, x2, y2]}, ...],
  "image_width": 1000,
  "image_height": 1400
}
```

## Usage

### 1. Generate Predictions

```bash
# Text reading (plain text output)
uv run python run_evaluation.py \
    --model-name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --shard-path data.tar \
    --task-types reading \
    --output-types text \
    --csv-output predictions.csv

# Detection (bounding boxes only)
uv run python run_evaluation.py \
    --model-name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --shard-path data.tar \
    --task-types detection \
    --output-types box \
    --csv-output detection.csv

# Structured reading (text + bounding boxes)
uv run python run_evaluation.py \
    --model-name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --shard-path data.tar \
    --task-types reading \
    --output-types "[lines, box]" \
    --csv-output reading-lines.csv
```

### 2. Score Predictions

```bash
# Score text reading
uv run python score_text_reading.py predictions.csv --overwrite

# Score detection
uv run python score_detection.py detection.csv --overwrite

# Score structured reading (lines/paragraphs with boxes)
uv run python score_lines_reading.py reading-lines.csv --overwrite
```

## Task Types

| Task | Description | Output Types |
|------|-------------|--------------|
| `reading` | Full document OCR | `text`, `text2d`, `[lines, box]`, `[paragraphs, box]` |
| `detection` | Detect text regions (no transcription) | `box` |
| `localized_reading` | Read text within a specified bounding box | `lines`, `latex` |
| `conditional_detection` | Find bounding boxes for given text query | `box` |

## Metrics

**Text Reading:** Character Error Rate (CER), Word Error Rate (WER), ANLS, Exact Match

**Detection:** Precision, Recall, F1 at IoU thresholds (0.5, 0.75, etc.)

**Structured Reading:** Bbox metrics + text metrics for spatially matched regions

## Shell Script Examples

See `detection.sh`, `reading_text.sh`, `reading_lines.sh`, etc. for batch evaluation examples across multiple shards and models.
