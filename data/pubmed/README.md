# PubMed OCR Processing Pipeline

-   Download the list of PDFs [here](https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_non_comm_use_pdf.csv)
-   Iterate through the PDFs and download them using `download_parallel.sh`

This will generate ~2M PDFs.

This directory contains scripts for processing PubMed PDFs with Google Vision OCR, handling failures, and visualizing results.

## Prerequisites

### 1. Google Cloud Authentication

Before running any OCR scripts, you must set up Google Cloud authentication:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/google-serviceauth.json"
```

### 2. Python Dependencies

Install required packages:

```bash
pip install google-cloud-vision pillow pdf2image
```

## Scripts Overview

### 1. `process_pubmed_ocr.py` - Main OCR Processing Script

**Purpose**: Processes PubMed PDFs using Google Vision OCR, creating both raw OCR data and processed images with metadata.

**Input Structure**: 

    input_dir/
    ├── [hex]/
    │   └── [hex]/
    │       └── filename.pdf

**Output Structure** (Compressed Shards):

    output_dir/
    ├── raw/
    │   └── shard_w{worker}_{shard}.tar.gz    # 2048 PDF+JSON pairs per shard
    ├── data/
    │   └── shard_w{worker}_{shard}.tar.gz    # 2048 PNG+JSON pairs per shard
    └── shard_manifest.json                   # Maps shard IDs to filenames

**Shard Format**: Each tar.gz contains exactly 2048 document pairs (4096 files total). Worker-specific naming ensures parallel safety.

**Usage Examples**:

```bash
# Basic processing
python3 process_pubmed_ocr.py \
    --input-dir /path/to/pubmed/oa_pdf \
    --output-dir /path/to/output \
    --num-workers 50

# Retry failed files
python3 process_pubmed_ocr.py \
    --input-dir /path/to/pubmed/oa_pdf \
    --output-dir /path/to/output \
    --failures-list /path/to/failures_list.txt \
    --num-workers 50

# Incremental processing (skip existing)
python3 process_pubmed_ocr.py \
    --input-dir /path/to/pubmed/oa_pdf \
    --output-dir /path/to/output \
    --skip-existing \
    --num-workers 50

# Limited test run
python3 process_pubmed_ocr.py \
    --input-dir /path/to/pubmed/oa_pdf \
    --output-dir /path/to/output \
    --max-files 100 \
    --num-workers 10
```

**Key Options**: `--input-dir`, `--output-dir`, `--num-workers`, `--failures-list`, `--skip-existing`, `--max-files`, `--start-index`

### 2. `retry_failed_ocr.py` - Failure Recovery Script

**Purpose**: Analyzes logs from previous OCR runs to identify failures and orchestrates their reprocessing.

**What it identifies as failures**:

-   PDFs that were never processed
-   PDFs that failed during processing
-   PDFs that were "successfully" processed but returned empty OCR content (rate limiting cases)

**Usage Examples**:

```bash
# Analyze and retry failures
python3 retry_failed_ocr.py \
    --log-dir /path/to/output/logs \
    --input-dir /path/to/pubmed/oa_pdf \
    --output-dir /path/to/output \
    --num-workers 80

# Dry run (analyze only)
python3 retry_failed_ocr.py \
    --log-dir /path/to/output/logs \
    --input-dir /path/to/pubmed/oa_pdf \
    --output-dir /path/to/output \
    --dry-run
```

**How it works**: Analyzes logs to identify unprocessed PDFs and files with empty OCR content (rate limiting), generates a failure list, then orchestrates retry processing.

**Key Options**: `--log-dir`, `--input-dir`, `--output-dir`, `--num-workers`, `--dry-run`

### 3. `visualize_pubmed_ocr.py` - OCR Visualization Script

**Purpose**: Creates visual overlays of OCR bounding boxes on page images to verify OCR accuracy.

**Features**:

-   Overlays detected lines, words, and paragraphs as colored bounding boxes
-   Supports processing individual PDFs, specific pages, or entire datasets
-   Generates high-quality visualization images for quality assessment

**Usage Examples**:

```bash
# Specific pages
python3 visualize_pubmed_ocr.py --input /path/to/output --output /tmp/viz \
    --pdf-id IJEM-16-117.PMC3354931 --pages 1,2,3

# All pages of one PDF
python3 visualize_pubmed_ocr.py --input /path/to/output --output /tmp/viz \
    --pdf-id IJEM-16-117.PMC3354931 --all-pages

# All PDFs (resource intensive)
python3 visualize_pubmed_ocr.py --input /path/to/output --output /tmp/viz --all-pdfs
```

**Visual Elements**: Red (lines), Blue (words), Green (paragraphs)

### 4. `format_ocr_json.py` - JSON Formatting Module

**Purpose**: Provides utilities for standardizing Google Vision OCR JSON responses.

**Key Functions**:

-   Converts 8-coordinate bounding boxes (X1,Y1,X2,Y2,X3,Y3,X4,Y4) to simplified 4-coordinate format [X1,Y1,X3,Y3]
-   Standardizes JSON structure across different OCR processing stages
-   Handles coordinate system transformations (e.g., DPI scaling)

**Usage**: This is primarily used as a module by other scripts, not run directly.

## Typical Workflow

1.  Set up authentication: `export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"`
2.  Run initial processing: `process_pubmed_ocr.py` with desired worker count
3.  Analyze failures: `retry_failed_ocr.py --dry-run` to check statistics
4.  Retry failures: `retry_failed_ocr.py` without dry-run flag
5.  Quality check: `visualize_pubmed_ocr.py` on sample PDFs

## Working with Sharded Output

### Shard Manifest Format

The `shard_manifest.json` maps shard IDs to contained files:

```json
{
  "raw": {
    "w000_00000": ["doc1.pdf", "doc1.json", "doc2.pdf", "doc2.json", ...]
  },
  "data": {
    "w000_00000": ["doc1.png", "doc1.json", "doc2.png", "doc2.json", ...]
  }
}
```

### Accessing Sharded Files

```bash
# Find shard containing a file
grep "filename.json" shard_manifest.json

# Extract specific files
tar -xzf raw/shard_w000_00000.tar.gz filename.pdf filename.json

# Extract entire shard
tar -xzf raw/shard_w000_00000.tar.gz -C output_dir/
```
