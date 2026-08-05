# Table Generation

Synthetic table image generator for training Table Structure Recognition (TSR) models. Produces JPEG images with JSON sidecars encoding word-level bounding boxes and OTSL table structure annotations, packaged into WebDataset-compatible tar shards.

No browser, geckodriver, or scikit-image required — rendering is done entirely with Pillow.

## Overview

The generator:

1. Randomly constructs a table structure (rows × cols, colspan/rowspan spans, header layout, border style)
2. Fills cells with text sampled from a content distribution (bundled word list or a custom UNLV-derived distribution)
3. Renders the table to a JPEG image using Pillow
4. Encodes the structure as an OTSL token sequence (`fcel`, `ecel`, `lcel`, `ucel`, `xcel`, `nl`) matching the docling-project FinTabNet_OTSL vocabulary (arXiv 2305.03393)
5. Optionally applies affine augmentations (shear, rotation) via Pillow — no scikit-image

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Generate samples

```bash
# 100 samples, reproducible seed
python generate.py --num-samples 100 --seed 42 --output-dir ./out/raw

# With geometric augmentation (2 variants per base table)
python generate.py --num-samples 100 --seed 42 --output-dir ./out/raw \
  --augment --augment-count 2

# Custom table dimensions and span probability
python generate.py --num-samples 500 --seed 0 --output-dir ./out/raw \
  --min-rows 2 --max-rows 15 --min-cols 2 --max-cols 10 --span-prob 0.3
```

### Package into tar shards

```bash
python sharding.py ./out/raw ./out/shards --samples-per-shard 1000

# Dry run to check counts without writing
python sharding.py ./out/raw ./out/shards --dry-run
```

## CLI Reference

### `generate.py`

| Argument | Default | Description |
|---|---|---|
| `--num-samples` | 100 | Number of base table images to generate |
| `--seed` | 42 | Random seed for reproducibility |
| `--output-dir` | `./out/raw` | Output directory for `.jpg` + `.json` pairs |
| `--min-rows` | 2 | Minimum table row count |
| `--max-rows` | 12 | Maximum table row count |
| `--min-cols` | 2 | Minimum table column count |
| `--max-cols` | 8 | Maximum table column count |
| `--span-prob` | 0.2 | Per-cell probability of starting a colspan/rowspan |
| `--augment` | off | Produce augmented variants of each base sample |
| `--augment-count` | 2 | Number of augmented variants per base sample |
| `--distribution` | bundled | Path to a custom UNLV distribution pickle |

### `sharding.py`

| Argument | Default | Description |
|---|---|---|
| `raw_dir` | — | Directory containing `.jpg` + `.json` pairs |
| `output_dir` | — | Destination for `train-NNNNN.tar` shards |
| `--samples-per-shard` | 1000 | Samples per tar shard |
| `--dry-run` | off | Report counts without writing |

## Output Format

### File layout

```
out/raw/
    00000000.jpg
    00000000.json
    00000001.jpg
    ...

out/shards/
    train-00000.tar
        00000000.jpg
        00000000.json
        ...
    train-00001.tar
    ...
```

### JSON sidecar schema

```json
{
  "image": {
    "path": "00000000.jpg",
    "width": 480,
    "height": 320
  },
  "text": {
    "words": [{"text": "Revenue", "box": [12, 8, 76, 24]}],
    "lines": [{"text": "Revenue Q1 Q2", "box": [12, 8, 200, 24]}]
  },
  "table": {
    "otsl": "fcel lcel nl fcel fcel nl",
    "html": "<table>...</table>",
    "rows": 2,
    "cols": 2
  }
}
```

Bounding boxes are `[x1, y1, x2, y2]` in absolute pixels. The `table.otsl` field uses the docling-project OTSL vocabulary (arXiv 2305.03393).

### OTSL token vocabulary

Matches the docling-project vocabulary from FinTabNet_OTSL (arXiv 2305.03393).

| Token | Meaning |
|---|---|
| `fcel` | Primary cell with content |
| `ecel` | Primary cell without content |
| `lcel` | Horizontal span extension (left-looking, colspan) |
| `ucel` | Vertical span extension (up-looking, rowspan) |
| `xcel` | 2D span extension (cell is both a colspan and rowspan extension) |
| `nl` | End of row |

## Using a Custom UNLV Distribution

The bundled `resources/sample_words.json` is sufficient for structural training but lacks real-table content fidelity. To substitute the UNLV distribution:

1. Obtain the UNLV ground-truth XML files and run the original `src/distribution.py` from `yosh-2022/TableGeneration` to produce a pickle.
2. Pass `--distribution /path/to/unlv_dist.pkl` to `generate.py`.

The pickle must be either:
- A `list[str]` of words, or
- A `dict` with `"words"`, `"numbers"`, and `"symbols"` keys (lists of strings)

## Running Tests

```bash
pytest tests/ -v
```

## OTSL Compatibility

The `table.otsl` field in every sidecar is validated at generation time against the structure dimensions. The vocabulary matches docling-project's FinTabNet_OTSL, enabling direct comparison and mixed training with that dataset.
