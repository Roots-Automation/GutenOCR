# TabMe++ Dataset Standardization Tools

Tools for downloading, converting, and standardizing the [TabMe++ (TABME++)](https://huggingface.co/datasets/rootsautomation/TABMEpp) table detection dataset into standard tar shards for OCR training pipelines.

## Overview

This package provides utilities to:

-   Download the TabMe++ dataset from HuggingFace
-   Convert raw TabMe++ directories into standardized tar shards
-   Process parquet-format TabMe++ data with parallel processing support

## Installation

```bash
pip install -r requirements.txt
```

For optional text processing and token counting (Qwen model):

```bash
pip install transformers
```

## Quick Start

### Download the Dataset

```bash
# Download from HuggingFace (requires huggingface_hub with hf_transfer)
./download.sh
```

### Convert Raw Directory Format

For datasets stored as directories with image/JSON pairs:

```bash
# Validate structure without producing output
python standardize.py /path/to/tabmepp /path/to/output --dry-run

# Convert with default settings (4096 documents per shard)
python standardize.py /path/to/tabmepp /path/to/output

# Convert with custom shard size
python standardize.py /path/to/tabmepp /path/to/output --documents-per-shard 1024
```

### Convert Parquet Format (Parallel)

For datasets in parquet format (HuggingFace downloads):

```bash
# Analyze without processing
python standardize_parquet.py "data/*.parquet" /path/to/output --dry-run

# Process all parquet files in parallel
python standardize_parquet.py "data/*.parquet" /path/to/output

# Process with custom settings
python standardize_parquet.py "data/*.parquet" /path/to/output \
    --pages-per-shard 4096 \
    --max-workers 8 \
    --no-text-processing
```

## Input Formats

### Directory Format (standardize.py)

    tabmepp/
    ├── document_id_1/
    │   ├── 0000.jpg
    │   ├── 0000.json
    │   ├── 0001.jpg
    │   ├── 0001.json
    │   └── ...
    ├── document_id_2/
    │   └── ...
    └── ...

### Parquet Format (standardize_parquet.py)

Parquet files with columns:

-   `doc_id` (str): Document identifier
-   `pg_id` (int): Page number within document
-   `ocr` (str): JSON string with OCR data (words_data, lines_data)
-   `img` (bytes): Raw image bytes

## Output Format

Output tar shards contain:

-   `{document_id}_{page_id}.jpg` - Page images
-   `{document_id}_{page_id}.json` - Standardized per-page JSON

### Standard Per-Page JSON Schema

```json
{
  "text": {
    "words": [{"text": "string", "box": [x1, y1, x2, y2]}],
    "lines": [{"text": "string", "box": [x1, y1, x2, y2]}],
    "text": "1D text representation (reading order)",
    "text2d": "2D text representation (spatial layout preserved)"
  },
  "image": {
    "path": "{document_id}_{page_id}.jpg",
    "width": 1700,
    "height": 2200,
    "dpi": 200
  },
  "metadata": {
    "qwen_tokens": {
      "text": 150,
      "text2d": 200,
      "words": 180,
      "lines": 160,
      "image": 483
    }
  }
}
```

## CLI Reference

### standardize.py (Directory Input)

| Argument                | Default    | Description                            |
| ----------------------- | ---------- | -------------------------------------- |
| `input_path`            | _required_ | Path to TabMe++ dataset root           |
| `output_path`           | _required_ | Output directory for tar shards        |
| `--documents-per-shard` | 4096       | Max documents per shard                |
| `--allow-unpaired`      | false      | Write images even when JSON is missing |
| `--dry-run`             | false      | Validate and count without writing     |

### standardize_parquet.py (Parquet Input)

| Argument                | Default                     | Description                     |
| ----------------------- | --------------------------- | ------------------------------- |
| `input_patterns`        | _required_                  | Glob patterns for parquet files |
| `output_path`           | _required_                  | Output directory for tar shards |
| `--pages-per-shard`     | 2048                        | Max pages per shard             |
| `--max-workers`         | min(files, cores)           | Parallel processing workers     |
| `--no-text-processing`  | false                       | Skip text/token processing      |
| `--dry-run`             | false                       | Analyze without writing         |
| `--model-id`            | Qwen/Qwen2.5-VL-7B-Instruct | Model for tokenization          |
| `--no-local-files-only` | false                       | Allow downloading model         |

## Text Processing

The standardization scripts include text processing to generate:

-   **text (1D)**: Linear text in reading order (top-to-bottom, left-to-right)
-   **text2d**: Spatial text layout preserving 2D positioning using character density estimation

Token counts are computed using the Qwen2.5-VL tokenizer when available.

## Dependencies

### Required

-   Python 3.8+
-   Pillow (`pillow`)
-   tqdm
-   pandas (for parquet processing)
-   pyarrow (for parquet reading)

### Optional

-   transformers (for token counting)
-   huggingface_hub with hf_transfer (for fast downloads)
