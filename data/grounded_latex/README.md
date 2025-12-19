# LaTeX Equation Dataset Tools

This repository contains two main Python scripts for preparing and annotating LaTeX equation datasets.

---

## 1. `shard_dataset.py`
_Prepares standardized dataset shards from LaTeX PDFs and JSONs._

### What it does
- Reads pairs of `<doc_id>.pdf` and `<doc_id>.json` files from a source directory.
- Converts each PDF into a PNG at 72 DPI.
- Converts bounding boxes from absolute PDF coordinates into pixel coordinates.
- Packages each sample into three files inside a tar shard:
  - `<doc_id>.pdf`
  - `<doc_id>.png`
  - `<doc_id>.json` (updated schema with image info and pixel bbox coordinates)
- Outputs shards as `train-00000.tar`, `train-00001.tar`, etc., with up to 2048 samples each.

### Usage
```bash
python shard_dataset.py --src-dir ./input --out-dir ./output
```

### Options
- `--src-dir`: Source directory containing PDF/JSON pairs (default: `./input`)
- `--out-dir`: Output directory for tar shards (default: `./output`)
- `--shard-size`: Number of documents per shard (default: 2048)
- `--dpi`: DPI for PNG rendering (default: 72)
- `--max-docs`: Maximum documents to process (default: all)

---

## 2. `generate_equations.py`
_Compiles LaTeX expressions into PDFs and creates rotated variants._

### What it does
- Reads LaTeX math expressions from `wmf_texvc_inputs.json`.
- Compiles each expression to PDF using `pdflatex` (with retries and common fixes).
- Crops equations (`pdfcrop`, with ghostscript fallback).
- Places each equation on a blank page with safe margins.
- Creates a rotated variant (90°, 180°, 270°, or small random tilt).
- Writes paired PDF + JSON files into `dataset/<folder>/`.
- Runs in parallel with up to 70 processes for throughput.
- Logs errors to `equation_errors.log`.

### Usage
Download wmf_texvc_inputs.json from https://zenodo.org/records/15162182 (A JSON file containing unique LaTeX expressions).
```bash
python generate_equations.py --input wmf_texvc_inputs.json --output ./dataset
```

### Options
- `--input`: Path to JSON file with LaTeX expressions (default: `wmf_texvc_inputs.json`)
- `--output`: Output directory for generated PDFs/JSONs (default: `./dataset`)
- `--workers`: Number of parallel processes (default: auto-detected)
- `--skip-existing`: Skip equations that already exist (default: enabled)

---

## Quickstart

### 1. Install system dependencies
On **Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install -y texlive-latex-base texlive-latex-extra \
    texlive-fonts-recommended texlive-latex-recommended \
    pdfcrop ghostscript poppler-utils
```

On **macOS** (with Homebrew):
```bash
brew install mactex ghostscript poppler
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

Typical requirements include:
- `reportlab`
- `pdfrw`
- `pdf2image`
- `Pillow`

### 3. Run the scripts
Prepare shards from an existing dataset:
```bash
python shard_dataset.py --src-dir ./dataset --out-dir ./shards
```

Generate new equations from raw LaTeX expressions:
```bash
python generate_equations.py --input wmf_texvc_inputs.json --output ./dataset
```

---

## 3. `cleanup_incomplete.sh`
_Removes incomplete equation sets and temporary files from the dataset._

### What it does
- Scans each folder in `dataset/` for equation sets.
- Removes any equation that doesn't have all 4 expected files (base PDF/JSON + rotated variant PDF/JSON).
- Cleans up temporary `equation_*` files left over from LaTeX compilation.
- Reports per-folder statistics and overall completion percentage.

### Usage
```bash
bash cleanup_incomplete.sh [dataset_dir]
```

---

## 4. `count_success.sh`
_Dataset integrity checker that monitors processing status._

### What it does
- Counts PDF and JSON files in the dataset directory.
- Detects mismatches between PDF/JSON pairs.
- Monitors recent file creation to estimate processing rate.
- Reports system load and identifies potential bottlenecks.

### Usage
```bash
bash count_success.sh [dataset_dir]
```

---

## 5. `visualize_bbox.ipynb`
_Jupyter notebook for visualizing bounding boxes on generated images._

### What it does
- Loads samples from tar shards.
- Displays images with bounding box overlays.
- Useful for validating dataset quality and debugging annotation issues.

### Usage
Open in Jupyter and set `TAR_PATH` to point to your shard file.

---

## Notes
- All scripts assume LaTeX/PDF toolchain is installed and available on `PATH`.
- Use **`shard_dataset.py`** when standardizing an existing dataset of PDFs + JSONs.
- Use **`generate_equations.py`** when generating new equations from raw LaTeX expressions.
- Use **`cleanup_incomplete.sh`** to clean up failed or incomplete equation generations.
- Use **`count_success.sh`** to monitor dataset integrity during processing.
