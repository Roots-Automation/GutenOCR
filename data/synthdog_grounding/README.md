# SynthDoG Grounding: Synthetic Document Generator with Grounding Annotations

This module generates synthetic document images with line-, word-, and block-level bounding-box annotations for visual document understanding (VDU) tasks.

> **Attribution**: Fork of [SynthDoG](https://github.com/clovaai/donut/tree/master/synthdog) from the [Donut](https://github.com/clovaai/donut) project by NAVER Corp. (MIT License). Extended with grounding annotations, contrast-aware rendering, HuggingFace dataset support, and additional tooling.

## Key Features

- **Multi-level grounding**: Line, word, and block bounding boxes in normalized `[x1, y1, x2, y2]` coordinates
- **Coherent text**: Words are never split across lines
- **Contrast-aware rendering**: Text color adapts to background luminance using the WCAG relative-luminance formula
- **Color paper backgrounds**: Random RGB paper colors with optional texture overlay
- **Dual text sources**: Local corpus files or HuggingFace datasets (streaming supported)
- **Multilingual**: English, Chinese, Japanese, Korean with language-specific fonts
- **Document effects**: Perspective transforms, elastic distortion, Gaussian noise, color/shadow/blur post-processing
- **Per-sample quality metrics**: Contrast, bbox area, textbox fill rate, and degenerate bbox counts
- **HuggingFace-compatible output**: JSONL metadata format ready for dataset loading

> **Word boundary caveat**: Word-level bounding boxes are computed by splitting on whitespace. This works well for space-delimited languages (English, etc.) but will not produce meaningful word segments for CJK languages, where the `words` field will typically contain single characters or entire lines.

> **AABB after perspective**: Line-level bounding boxes are axis-aligned bounding rectangles (AABBs) of the transformed quad — slightly loose after perspective/elastic distortion. Word-level AABBs are derived from quad interpolation and tightly enclose each word even after distortion. Enable `emit_quads: true` for exact 4-corner polygon coordinates at both levels. All normalized coordinates are clamped to `[0, 1]`. Samples with no visible text are silently skipped (not saved). Lines whose AABB area falls below `min_bbox_area` pixels (default 16) are excluded from annotations (but remain in the rendered image).

## Directory Structure

```
synthdog_grounding/
├── template.py                  # Main SynthDoG template (generation + save)
├── pillow_compat.py             # Pillow 10+ compatibility patches
├── pyproject.toml               # Project metadata and dependencies
├── requirements-synthdog.txt    # Lightweight pip requirements
│
├── config/                      # Language and source configurations
│   ├── config_en.yaml           # English (file-based corpus)
│   ├── config_en-pdfs.yaml      # English (PDF-style layout)
│   ├── config_huggingface.yaml  # English (HuggingFace streaming)
│   ├── config_zh.yaml           # Chinese
│   ├── config_ja.yaml           # Japanese
│   └── config_ko.yaml           # Korean
│
├── elements/                    # Document generation components
│   ├── background.py            # Background texture generation
│   ├── paper.py                 # Paper color and texture (with luminance output)
│   ├── content.py               # Text readers (file-based and HuggingFace)
│   ├── document.py              # Document orchestration and geometric effects
│   └── textbox.py               # Single-line text rendering and word tracking
│
├── layouts/                     # Text layout engines
│   ├── grid.py                  # Single grid layout (rows x columns)
│   └── grid_stack.py            # Stacked multi-section grid layout
│
├── resources/                   # Fonts, backgrounds, paper textures, corpora
│   ├── font/{en,ja,ko,zh}/      # Language-specific font directories
│   ├── background/              # Background texture images
│   ├── paper/                   # Paper texture images
│   └── corpus/                  # Text corpus files (e.g. enwiki.txt)
│
├── data_generation/
│   └── run_synthdog_range.sh    # Batch generation for directory ID ranges
│
├── data_packaging/
│   ├── build_tar.py             # Single tar archive from a data directory
│   └── build_tars_parallel.py   # Parallel tar creation across directories
│
├── data_analysis/
│   ├── generate_stats.py        # Statistics for individual tar files
│   ├── aggregate_stats.py       # Aggregate statistics across datasets
│   └── simple_batch_process.sh  # Batch processing wrapper
│
├── data_extraction/
│   ├── check_sample.py          # Extract and visualize annotated samples
│   └── extract_finepdfs.py      # Extract text from FinePDFs dataset
│
└── outputs/                     # Generated data (git-ignored)
```

## Prerequisites

- Python >= 3.8
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [SynthTiger](https://github.com/clovaai/synthtiger)

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Quick Start

### Generate Synthetic Data

```bash
# Required on macOS
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Generate English documents (50 samples, 4 workers)
uv run python -m synthtiger -o ./outputs/SynthDoG_en -c 50 -w 4 -v template.py SynthDoG config/config_en.yaml

# Generate using HuggingFace streaming corpus
uv run python -m synthtiger -o ./outputs/SynthDoG_hf -c 50 -w 4 -v template.py SynthDoG config/config_huggingface.yaml

# Batch generation for directory ranges (e.g., dirs 0035-0075)
data_generation/run_synthdog_range.sh 35 75
```

#### SynthTiger CLI Arguments

| Flag | Description |
|------|-------------|
| `-o` | Output directory path |
| `-c` | Number of documents to generate |
| `-w` | Number of worker processes |
| `-s` | Random seed for reproducibility |
| `-v` | Verbose output (print error messages) |

### Inspect Samples

```bash
# Extract first 25 samples with bounding box annotations
uv run python data_extraction/check_sample.py /path/to/data.tar

# Extract specific samples with text labels
uv run python data_extraction/check_sample.py /path/to/data.tar --ids 00087 00042 --label-with-text
```

| Flag | Description | Default |
|------|-------------|---------|
| `-o, --output` | Output directory | `./check_sample` |
| `-n, --first-n` | Number of samples to extract | 25 |
| `--ids` | Specific sample IDs to extract | — |
| `--line-width` | Bbox annotation line width | 3 |
| `--label-with-text` | Include text content in labels | off |
| `--font-path` | Custom TTF font for labels | — |

### Package Data into Archives

```bash
# Single tar archive
uv run python data_packaging/build_tar.py /path/to/data/directory -o output.tar

# Parallel tar creation across directories
uv run python data_packaging/build_tars_parallel.py --core-dir /path/to/data
```

### Generate Statistics

```bash
# Single tar file
uv run python data_analysis/generate_stats.py /path/to/data.tar

# Batch process all tar files in a directory
data_analysis/simple_batch_process.sh /path/to/data/directory

# Aggregate across multiple tar files
uv run python data_analysis/aggregate_stats.py -d /path/to/directory -o aggregated_stats
```

## Output Format

Each generation run produces a split directory structure:

```
outputs/<run_name>/
├── train/
│   ├── image_0.jpg
│   ├── image_3.jpg
│   ├── ...
│   └── metadata.jsonl
├── validation/
│   └── ...
└── test/
    └── ...
```

Split assignment follows the configured ratio (default 80/10/10 train/val/test).

### Metadata Schema

Each line in `metadata.jsonl` is a JSON object:

```json
{
  "file_name": "image_0.jpg",
  "ground_truth": "{\"gt_parse\": {\"text_lines\": [...], \"text_bboxes\": [...], \"text_blocks\": [...], \"text_words\": [...], \"quality_metrics\": {...}}}"
}
```

The `ground_truth` field is a JSON string containing a `gt_parse` object with the following keys:

#### `text_lines`

Per-line text with bounding box and identifiers:

```json
{"text": "hello world", "bbox": [0.1, 0.2, 0.5, 0.25], "line_id": 0, "block_id": 0}
```

When `emit_quads` is enabled, each entry also includes a `"quad"` field (see [Quad Coordinates](#quad-coordinates) below).

#### `text_bboxes`

Flat list of `[x1, y1, x2, y2]` bounding boxes (one per line, same order as `text_lines`). All coordinates are normalized to `[0, 1]` relative to image dimensions.

#### `text_blocks`

Block-level groupings (lines sharing the same visual column/section):

```json
{"block_id": 0, "bbox": [0.1, 0.2, 0.5, 0.4], "line_ids": [0, 1, 2]}
```

#### `text_words`

Word-level grounding with line association:

```json
{"text": "hello", "bbox": [0.1, 0.2, 0.3, 0.25], "word_id": 0, "line_id": 0}
```

When `emit_quads` is enabled, each entry also includes a `"quad"` field (see [Quad Coordinates](#quad-coordinates) below).

#### `text_quads` (when `emit_quads: true`)

Flat list of quad coordinates (one per line, same order as `text_bboxes`). Each quad is `[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]` in TL, TR, BR, BL order, normalized to `[0, 1]`.

#### Quad Coordinates

When `emit_quads: true` is set in the config, line and word entries include a `"quad"` field containing the 4-corner polygon coordinates after perspective/elastic distortion. This provides tighter ground truth than the axis-aligned bounding box.

Format: `[[x, y], [x, y], [x, y], [x, y]]` — corners in **TL, TR, BR, BL** order, normalized to `[0, 1]` relative to image dimensions.

Word quads are computed via bilinear interpolation of the word's `x1_ratio`/`x2_ratio` along the parent line's quad edges. When no perspective is applied, quads degenerate to the same rectangle as the AABB.

This flag is backwards compatible: when `emit_quads` is `false` (the default), output is identical to the current format.

#### `quality_metrics`

Per-sample quality and integrity signals:

| Field | Type | Description |
|-------|------|-------------|
| `min_line_contrast` | `float \| null` | Minimum RMS contrast (std of grayscale pixel values) across all surviving line bbox regions. `null` if no measurable lines. |
| `mean_line_contrast` | `float \| null` | Mean RMS contrast across all surviving line bbox regions. |
| `min_line_bbox_area_px` | `int \| null` | Smallest surviving line bbox area in pixels. |
| `min_word_bbox_area_px` | `int \| null` | Smallest surviving word bbox area in pixels. |
| `degenerate_line_count` | `int` | Number of lines removed because their bbox area was below `min_bbox_area`. |
| `degenerate_word_count` | `int` | Number of words removed (belonging to degenerate lines). |
| `textbox_null_count` | `int` | Number of textbox layout slots that produced no text (e.g., text didn't fit). |
| `textbox_total_count` | `int` | Total textbox layout slots attempted. |
| `image_size` | `[int, int]` | Image dimensions `[width, height]` in pixels. |

## Generation Pipeline

The `SynthDoG` template in `template.py` orchestrates this pipeline per image:

1. **Size**: Randomize dimensions (short side 720-1024px, aspect ratio 1:1 to 2:1, 50% landscape)
2. **Background**: Generate background texture layer
3. **Paper**: Generate paper layer (50% chance of random RGB color, optional texture overlay)
4. **Content**: Render text onto the paper using the selected text reader and layout
   - Text color adapts to paper luminance for contrast
   - Words are never split across lines
5. **Document effects**: Apply elastic distortion, perspective transform, and Gaussian noise to the document group
6. **Bbox capture**: Compute line/word/block bounding boxes from the transformed text layers (AABBs)
7. **Degenerate filtering**: Remove annotations for lines whose bbox area is below `min_bbox_area` pixels
8. **Compositing**: Merge document group with background
9. **Pixel effects**: Apply color shift, shadow, contrast, brightness, motion blur, and Gaussian blur
10. **Quality metrics**: Measure RMS contrast per line, compute bbox area stats, record textbox fill rate
11. **Save**: Write JPEG image and append metadata (including `quality_metrics`) to `metadata.jsonl`

> **Multi-worker safety**: SynthTiger calls `save()` from the main process only, so there is no risk of concurrent writes corrupting the JSONL metadata file, regardless of `-w` worker count.

## Known Limitations

- **Contrast-aware color is pre-effects**: Text color is chosen based on the base paper RGB color *before* texture overlay and all pixel-level effects (shadow, brightness, contrast, blur). Actual post-effects contrast may be lower than what the WCAG-based color selection targeted. Use `min_line_contrast` in `quality_metrics` to filter low-contrast samples.
- **Elastic distortion warps pixels but not quads**: Quad coordinates reflect perspective transforms only. Elastic distortion modifies the rendered pixels without updating annotation coordinates. For the moderate distortion levels used (alpha ≤ 1, sigma ≤ 0.5), this is a good approximation.
- **Motion blur shifts apparent text position**: Motion blur (k=3–5) is applied after bbox capture and can shift apparent text position by ~1–2px beyond annotated boundaries.
- **Shadow/brightness reduce legibility**: These pixel-level effects can reduce text legibility below what the WCAG-based color selection targeted. The `min_line_contrast` metric captures this.

## Configuration

Each YAML config file controls the full pipeline. Key sections:

| Section | Controls |
|---------|----------|
| `quality` | JPEG quality range (e.g. `[50, 95]`) |
| `landscape` | Probability of landscape orientation |
| `emit_quads` | Emit quad (4-corner) coordinates for lines and words (default `false`) |
| `min_bbox_area` | Minimum line bbox area in pixels; lines below this are excluded from annotations (default `16`) |
| `short_size` | Short dimension range in pixels |
| `aspect_ratio` | Min/max aspect ratio |
| `background` | Background texture source |
| `document.paper` | Paper color ranges and texture source |
| `document.content.text` | Text corpus path or HuggingFace dataset config |
| `document.content.font` | Font paths and size ranges |
| `document.content.layout` | Grid dimensions, text scale, alignment |
| `document.effect` | Geometric effects (elastic, perspective, noise) |
| `effect` | Pixel-level effects (color, shadow, blur, contrast) |

### HuggingFace Text Source

To use a HuggingFace dataset as the text corpus, set the `text` section in the config:

```yaml
text:
  use_huggingface: true
  dataset_name: "allenai/c4"
  subset: "en"
  split: "train"
  streaming: true
  buffer_size: 1000
```

See `config/config_huggingface.yaml` for a complete example.

## Environment Variables

```bash
# Required on macOS for multiprocessing
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Optional performance tuning
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## License

MIT License. See [LICENSE](LICENSE) for details.

For questions or issues, please refer to the [SynthTiger documentation](https://github.com/clovaai/synthtiger) or create an issue in the [GutenOCR repository](https://github.com/Roots-Automation/GutenOCR/issues).
