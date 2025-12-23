# SynthDoG Grounding 🐶: Synthetic Document Generator with Grounding

This module provides a complete pipeline for generating, processing, and analyzing synthetic document data using SynthDoG (Synthetic Document Generator) for visual document understanding (VDU) with grounding annotations.

> **Attribution**: This module is a fork of [SynthDoG](https://github.com/clovaai/donut/tree/master/synthdog) from the [Donut](https://github.com/clovaai/donut) project by NAVER Corp., released under the MIT License. We've extended it with grounding annotations, coherent text generation, and additional tooling.

## Overview

The SynthDoG Grounding pipeline consists of several key components:

1.  **Data Generation**: Creating synthetic documents with grounding annotations
2.  **Data Packaging**: Converting generated data into efficient tar archives
3.  **Quality Analysis**: Statistical analysis and validation of generated data
4.  **Data Extraction**: Tools for extracting and inspecting dataset contents

### Extensions to SynthDoG

-   Grounding Annotations: Each generated document includes detailed bounding box annotations for text lines.
-   Coherent text: Words are never split across lines, ensuring readability.
-   Novel data support: Hugging Face datasets for text corpora, both static and streaming.

## Directory Structure

    synthdog_grounding/
    ├── README.md                    # This documentation
    ├── requirements-synthdog.txt    # Python dependencies
    │
    ├── config/                      # SynthDoG configuration files
    │   ├── config_en.yaml          # English documents
    │   ├── config_en-pdfs.yaml     # English PDF-based documents
    │   ├── config_zh.yaml          # Chinese documents
    │   ├── config_ja.yaml          # Japanese documents
    │   └── config_ko.yaml          # Korean documents
    │
    ├── elements/                    # SynthDoG document elements
    ├── layouts/                     # SynthDoG layout definitions
    ├── resources/                   # SynthDoG resources (fonts, backgrounds, etc.)
    │
    ├── template.py                  # Main SynthDoG template
    │
    ├── data_generation/             # Data generation scripts
    │   └── run_synthdog_range.sh   # Generate data for ID ranges
    │
    ├── data_packaging/              # Data packaging and archiving
    │   ├── build_tar.py            # Create tar archives from generated data
    │   └── build_tars_parallel.py  # Parallel tar creation
    │
    ├── data_analysis/               # Data analysis and statistics
    │   ├── generate_stats.py       # Generate statistics for tar files
    │   ├── aggregate_stats.py      # Aggregate statistics across datasets
    │   └── simple_batch_process.sh # Simple batch processing script
    │
    ├── data_extraction/             # Data extraction and inspection
    │   ├── check_sample.py         # Extract and visualize samples
    │   └── extract_finepdfs.py     # Extract text from FinePDFs dataset
    │
    └── outputs/                    # Generated data (gitignored)

## Prerequisites

-   Python >= 3.8
-   [synthtiger](https://github.com/clovaai/synthtiger)
-   Additional dependencies: see `requirements-synthdog.txt`

```bash
# Install synthtiger
pip install synthtiger

# Install additional dependencies
pip install -r requirements-synthdog.txt
```

## Quick Start

### 1. Generate Synthetic Data

Generate synthetic documents for a specific range of IDs:

```bash
# Set environment variable (required for macOS)
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Generate data for directories 0035-0075 (configurable)
./run_synthdog_range.sh 35 75
```

### 2. Package Data into Archives

Convert generated directories into efficient tar archives:

```bash
# Create a single tar archive from a data directory
python data_packaging/build_tar.py /path/to/data/directory -o output.tar

# Or create tar archives in parallel for multiple directories
python data_packaging/build_tars_parallel.py --core-dir /path/to/data
```

### 3. Generate Statistics

Analyze the generated data and create statistical summaries:

```bash
# Process all tar files in a directory
./data_analysis/simple_batch_process.sh /path/to/data/directory

# Or process a single tar file
python data_analysis/generate_stats.py /path/to/data.tar
```

### 4. Inspect Samples

Extract and visualize samples from the dataset:

```bash
# Extract first 10 samples with bounding box annotations
python data_extraction/check_sample.py /path/to/data.tar -n 10

# Extract specific samples with text labels
python data_extraction/check_sample.py /path/to/data.tar --ids 00087 00042 --label-with-text
```

## Detailed Usage

### Data Generation

The data generation process uses SynthDoG with custom configurations for different languages and document types.

#### Basic Generation

```bash
# Generate English documents
synthtiger -o ./outputs/SynthDoG_en -c 50 -w 4 -v template.py SynthDoG config/config_en.yaml

# Generate multilingual documents
synthtiger -o ./outputs/SynthDoG_zh -c 50 -w 4 -v template.py SynthDoG config/config_zh.yaml  # Chinese
synthtiger -o ./outputs/SynthDoG_ja -c 50 -w 4 -v template.py SynthDoG config/config_ja.yaml  # Japanese
synthtiger -o ./outputs/SynthDoG_ko -c 50 -w 4 -v template.py SynthDoG config/config_ko.yaml  # Korean
```

#### Key Arguments

-   `-o` : Output directory path
-   `-c` : Number of documents to generate
-   `-w` : Number of worker processes
-   `-s` : Random seed for reproducibility
-   `-v` : Verbose output (print error messages)

### Data Packaging

#### Single Archive Creation

```bash
# Create a tar archive from a data directory
python data_packaging/build_tar.py /path/to/data/directory -o output.tar

# Options:
# -o, --output: Output tar file path (defaults to <directory>.tar)
```

#### Parallel Archive Creation

```bash
# Process all directories in parallel (uses SYNTHDOG_DATA_DIR env var or ./outputs default)
python data_packaging/build_tars_parallel.py --core-dir /path/to/data

# Options:
# --core-dir: Directory containing numbered subdirectories
# --workers: Maximum number of parallel workers (default: CPU count)
# --start/--end: Directory range to process
```

### Statistical Analysis

#### Generate Statistics for Single Archive

```bash
# Generate comprehensive statistics for a tar file
python data_analysis/generate_stats.py /path/to/data.tar

# Output: Creates data.stats.csv with detailed metrics
```

#### Batch Statistics Processing

```bash
# Process all tar files in a directory (creates .stats.csv files)
./data_analysis/simple_batch_process.sh /path/to/data/directory
```

#### Aggregate Statistics

```bash
# Combine statistics from multiple tar files in a directory
python data_analysis/aggregate_stats.py -d /path/to/directory -o aggregated_stats

# Or specify tar files directly
python data_analysis/aggregate_stats.py file1.tar file2.tar -o aggregated_stats

# Options:
# -d, --directory: Directory to scan for tar files with .stats.csv files
# -o, --output: Output path prefix (creates .csv and .json files)
```

### Data Inspection and Validation

#### Extract Sample Images

```bash
# Extract and visualize first 25 samples with bounding box annotations
python data_extraction/check_sample.py /path/to/data.tar

# Extract specific samples with custom options
python data_extraction/check_sample.py /path/to/data.tar --ids 00087 00042 -o ./my_samples --label-with-text --line-width 5
```

CLI options:

-   `tar_file`: Path to tar archive to inspect (required)
-   `-o, --output`: Output directory (default: `./check_sample`)
-   `-n, --first-n`: Number of samples to extract (default: 25)
-   `--ids`: Specific sample IDs to extract
-   `--line-width`: Annotation line thickness (default: 3)
-   `--label-with-text`: Include text content in annotation labels
-   `--font-path`: Custom TTF font for labels

#### Extract Text Corpus

```bash
# Extract text from FinePDFs dataset for training corpus
# Note: This script streams from HuggingFace and extracts 1M ASCII samples
python data_extraction/extract_finepdfs.py
```

## Configuration Files

### Language-Specific Configs

-   `config_en.yaml`: English documents with standard layouts
-   `config_en-pdfs.yaml`: English documents with PDF-style layouts
-   `config_zh.yaml`: Chinese documents
-   `config_ja.yaml`: Japanese documents
-   `config_ko.yaml`: Korean documents

Each config file specifies:

-   Text corpus sources
-   Font selections
-   Layout parameters
-   Image quality settings
-   Document dimensions

### Custom Configuration

To create custom configurations:

1.  Copy an existing config file
2.  Modify corpus paths, fonts, and layout parameters
3.  Update resource paths as needed
4.  Test with small sample generation first

## Output Format

### Generated Data Structure

Each generated sample consists of:

-   **Image file**: `{id}.jpg` - The synthetic document image
-   **Annotation file**: `{id}.json` - Grounding annotations with:
    -   Text lines with bounding boxes
    -   Character-level annotations
    -   Layout information
    -   Metadata (DPI, dimensions, etc.)

### Statistics Output

Statistics files contain comprehensive metrics:

-   Sample counts and distributions
-   Image dimensions and quality metrics
-   Text statistics (character counts, word counts)
-   Bounding box analysis (overlap, density)
-   Layout complexity measures

## Performance Optimization

### Generation Performance

-   Use appropriate worker count (`-w`) based on CPU cores
-   Optimize batch sizes for memory usage
-   Use SSD storage for faster I/O
-   Monitor memory usage during generation

### Processing Performance

-   Use parallel processing scripts for large datasets
-   Set appropriate timeout values for batch operations
-   Process statistics incrementally to avoid recomputation
-   Use compression for archive storage

## Troubleshooting

### Common Issues

1.  **Memory errors during generation**

    -   Reduce worker count (`-w`)
    -   Decrease batch size
    -   Monitor system memory usage

2.  **Missing dependencies**

    -   Install synthtiger: `pip install synthtiger`
    -   Install additional deps: `pip install -r requirements-synthdog.txt`

3.  **Archive creation failures**

    -   Check disk space availability
    -   Verify input directory structure
    -   Use appropriate compression levels

4.  **Statistics processing timeouts**
    -   Increase timeout values in batch scripts
    -   Process files individually if needed
    -   Check for corrupted tar files

### Environment Variables

```bash
# Required for macOS
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Optional performance tuning
export OMP_NUM_THREADS=1  # Limit OpenMP threads
export MKL_NUM_THREADS=1  # Limit MKL threads
```

## Development

### Adding New Analysis Tools

1.  Create new Python scripts in appropriate subdirectories
2.  Follow the existing argument parsing patterns
3.  Add comprehensive docstrings and type hints
4.  Update this README with usage instructions

### Extending Configuration

1.  Copy existing config files as templates
2.  Modify corpus sources and parameters
3.  Test with small sample generations
4.  Document new config options

For questions or issues, please refer to the [SynthTiger documentation](https://github.com/clovaai/synthtiger) or create an issue in this repository.
