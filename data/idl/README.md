# IDL Dataset Standardization Tools

Tools for converting and compressing Industry Document Library (IDL) datase into standardized tar shards for OCR training pipelines.

## Overview

This package provides utilities to:

-   Convert IDL document directories into standardized tar shards with consistent JSON format
-   Batch process multiple IDL directories in parallel
-   Compress raw IDL directories into `.tar.gz` archives

## Installation

```bash
pip install -r requirements.txt
```

For optional text processing and token counting (Qwen model):

```bash
pip install transformers
```

## Quick Start

### Convert a Single IDL Directory

```bash
# Validate structure without producing output
python standardize.py \
    --input-dir /path/to/idl-train-00001 \
    --output /path/to/output/train-00001.tar \
    --dry-run

# Convert with text processing and token counting
python standardize.py \
    --input-dir /path/to/idl-train-00001 \
    --output /path/to/output/train-00001.tar

# Convert without text processing (faster)
python standardize.py \
    --input-dir /path/to/idl-train-00001 \
    --output /path/to/output/train-00001.tar \
    --no-text-processing
```

### Batch Convert Multiple Directories

```bash
# Process all idl-train-* folders
python batch_standardize.py \
    --input-base /path/to/idl/samples \
    --output-base /path/to/output \
    --pattern "idl-train-*"

# Process with custom settings and parallelism
python batch_standardize.py \
    --input-base /path/to/idl/samples \
    --output-base /path/to/output \
    --pattern "idl-val-*" \
    --dpi 150 \
    --image-ext jpg \
    --max-workers 4
```

### Compress Raw Directories

```bash
# Sequential compression
PREFIX=/path/to/idl-train ./compress.sh 1 100

# Parallel compression (requires GNU parallel)
PREFIX=/path/to/idl-train MAX_JOBS=16 ./compress_parallel.sh 1 100
```

## Input Format

IDL directories are expected to have the following structure:

    idl-train-00001/
    ├── document_id_1/
    │   ├── document_id_1.pdf
    │   └── document_id_1.json
    ├── document_id_2/
    │   ├── document_id_2.pdf
    │   └── document_id_2.json
    └── ...

Each document JSON contains page-level OCR data:

```json
{
  "pages": [
    {
      "bbox": [[x, y, w, h], ...],
      "poly": [...],
      "score": [...],
      "text": ["PHILIP MORRIS MANAGEMENT CORP.", ...]
    }
  ]
}
```

## Output Format

The output tar shards contain:

-   `{document_id}.pdf` - Original PDF
-   `{document_id}_{page_id}.png` - Rasterized page images (72 DPI by default)
-   `{document_id}_{page_id}.json` - Standardized per-page JSON

### Standard Per-Page JSON Schema

```json
{
  "text": {
    "lines": [{"text": "string", "box": [x1, y1, x3, y3]}],
    "text": "1D text representation (reading order)",
    "text2d": "2D text representation (spatial layout preserved)"
  },
  "image": {
    "path": "{document_id}_{page_id}.png",
    "width": 612,
    "height": 792,
    "dpi": 72
  },
  "metadata": {
    "qwen_tokens": {
      "text": 150,
      "text2d": 200,
      "lines": 180,
      "image": 483
    }
  }
}
```

## CLI Reference

### standardize.py

| Argument                | Default                     | Description                                     |
| ----------------------- | --------------------------- | ----------------------------------------------- |
| `--input-dir`           | _required_                  | Path to IDL directory (e.g., `idl-train-00001`) |
| `--output`              | _required_                  | Output tar file path                            |
| `--dpi`                 | 72                          | Rasterization DPI                               |
| `--image-ext`           | png                         | Image format: `png`, `jpg`, `tif`               |
| `--page-base`           | 0                           | Page index base (0 or 1)                        |
| `--bbox-format`         | xywh                        | Source bbox format: `xywh` or `x1y1x2y2`        |
| `--bbox-space`          | as_is                       | Bbox scaling: `as_is` or `pdf_to_pixel`         |
| `--dry-run`             | false                       | Validate only, no output                        |
| `--no-text-processing`  | false                       | Skip text processing (faster)                   |
| `--model-id`            | Qwen/Qwen2.5-VL-7B-Instruct | Model for tokenization                          |
| `--no-local-files-only` | false                       | Allow downloading model from HuggingFace        |

### batch_standardize.py

| Argument        | Default    | Description                          |
| --------------- | ---------- | ------------------------------------ |
| `--input-base`  | _required_ | Base path containing IDL directories |
| `--output-base` | _required_ | Output directory for tar files       |
| `--pattern`     | _required_ | Glob pattern (e.g., `idl-train-*`)   |
| `--max-workers` | CPU count  | Number of parallel workers           |
| `--dpi`         | 72         | Rasterization DPI                    |
| `--image-ext`   | png        | Image format                         |

### Shell Scripts

Both `compress.sh` and `compress_parallel.sh` accept environment variables:

| Variable   | Default    | Description                                              |
| ---------- | ---------- | -------------------------------------------------------- |
| `PREFIX`   | _required_ | Path prefix for directories (e.g., `/path/to/idl-train`) |
| `WIDTH`    | 5          | Zero-padding width for directory numbers                 |
| `MAX_JOBS` | 80         | Max parallel jobs (parallel script only)                 |

**⚠️ Warning**: The compression scripts delete original directories after successful compression. Set `DELETE_AFTER_COMPRESS=false` to disable this behavior.

## Dependencies

### Required

-   Python 3.8+
-   PyMuPDF (`pymupdf`)
-   tqdm

### Optional

-   transformers (for token counting)
-   GNU parallel (for `compress_parallel.sh`)
