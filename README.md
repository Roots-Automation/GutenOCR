<p align="center">
  <img src="assets/guten-mascot.png" alt="GutenOCR Logo" width="300"/>
</p>

<h1 align="center">GutenOCR</h1>

<p align="center">
  <strong>Open-source tools for training and evaluating Vision Language Models for OCR</strong>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
    <a href="https://huggingface.co/collections/rootsautomation/gutenocr"><img src="https://img.shields.io/badge/🤗-Models-yellow.svg" alt="HuggingFace"></a>
  <a href="https://arxiv.org/abs/2601.14490"><img src="https://img.shields.io/badge/Paper-ArXiv-red.svg" alt="Paper"></a>
  <a href="https://ocr.roots.ai/"><img src="https://img.shields.io/badge/Demo-Live-green.svg" alt="Demo"></a>
</p>

* * *

## Overview

GutenOCR provides a complete toolkit for building OCR systems powered by Vision Language Models (VLMs).

<p align="center">
  <img src="assets/guten-radar.png" alt="Model Comparison" width="500"/>
</p>

### What's Inside

| Component                             | Description                                       |
| ------------------------------------- | ------------------------------------------------- |
| **[Data Pipelines](#data-pipelines)** | Data utilities for 6+ document sources            |
| **[Multi-GPU Training](#training)**   | Full-weight VLM fine-tuning with DeepSpeed ZeRO-3 |
| **[vLLM Evaluation](#evaluation)**    | OCR benchmarking framework                        |

* * *

## Models

We release state-of-the-art OCR models trained using this toolkit:

| Model | Parameters | Link |
| :--- | :--- | :--- |
| **GutenOCR-3B** | 3B | [🤗 rootsautomation/GutenOCR-3B](https://huggingface.co/rootsautomation/GutenOCR-3B) |
| **GutenOCR-7B** | 7B | [🤗 rootsautomation/GutenOCR-7B](https://huggingface.co/rootsautomation/GutenOCR-7B) |

> **Note:** These model weights are released under the **CC-BY-NC** license.

Try our **[Live Demo](https://ocr.roots.ai/)** to see them in action.

* * *

## Model Usage

You can easily use the released models with the `transformers` library.

### Quick Start

```python
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

# 1. Load model and processor
model_id = "rootsautomation/GutenOCR-3B"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# 2. Prepare inputs
image = Image.open("document.png")

# Example: Read all text
prompt = "Read all text in {image} and return a single TEXT string, linearized left-to-right/top-to-bottom."

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }
]

# 3. Process and Generate
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=4096)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text[0])
```

### Task Gallery

GutenOCR is steered by specific prompt templates. Below are common tasks:

#### Full OCR Reading
Extract text from the entire page.

**Prompt:** `Return a layout-sensitive TEXT2D representation of the image.`

**Output:**
```text
This is the text found in the document.
It preserves line breaks.
```

#### Text Detection
Locate regions of text (lines, paragraphs, math) without transcribing them.

**Prompt:** `Highlight all math in the image by returning their bounding boxes as a JSON array.`

**Output:**
```json
[[100, 200, 400, 250], [500, 600, 800, 650]]
```

#### Localized Reading
Read the text contained strictly within a specific bounding box.

**Prompt:** `What does it say in [100, 200, 500, 600] of the image?`

**Output:**
```text
Content of the specific box.
```

#### Conditional Detection (Search)
Find the bounding box locations of a specific query string.

**Prompt:** `Ground "Invoice #12345" in the image.`

**Output:**
```json
[[100, 200, 400, 250]]
```

### System Prompt

The model relies on a specific system prompt to enforce output formats (JSON, bounding box normalization, etc.). This is **automatically injected by the chat template**, but included below for reference.

<details>
<summary>Click to view the full System Prompt</summary>

```text
Your task is to read and localize text data from documents and images.

GEOMETRY:
    - Coordinates: integer pixels; origin (0,0) top-left; [x1,y1,x2,y2] with x1<x2, y1<y2.
    - Clip all boxes to the image bounds; drop boxes with zero/negative area.
    - Reading order: read text in natural reading order: top-to-bottom, left-to-right.
    - Rotated/angled text: return the axis-aligned bounding box of the minimal enclosing rectangle (no rotated boxes).

TASK TYPES:
    - reading: a full-text reading task on the entire image.
    - localized_reading: read text within a specified bounding box in the image.
    - detection: detect text regions in the image without transcription.
    - conditional_detection: detect text regions in the image based on a provided text query.

OUTPUT TYPES:
    - TEXT: one plain string; collapse multiple spaces to one; preserve line breaks. Non-grounded output only.
    - TEXT2D: one plain string; preserve whitespace as layout cue (spaces + `\n` only; no coordinates). Non-grounded output only.
    - LINES: JSON array of objects, corresponding to line-by-line OCR: `{"text": string, "bbox": [x1,y1,x2,y2]}`. When locally reading, only return the text: `string`.
    - WORDS: JSON array of objects, corresponding to word-by-word OCR: `{"text": string, "bbox": [x1,y1,x2,y2]}`.
    - PARAGRAPHS: JSON array of objects, corresponding to paragraph-wise OCR: `{"text": string, "bbox": [x1,y1,x2,y2]}`. When locally reading, only return the text: `string`.
    - LATEX: JSON array of objects, corresponding to LaTeX expressions: `{"text": string, "bbox": [x1,y1,x2,y2]}`. When locally reading, only return the latex: `string`.
    - BOX: JSON array of bounding boxes only: `[ [x1,y1,x2,y2], ... ]`. For detection and conditional_detection tasks only.

OUTPUT FORMAT
    - For non-grounded outputs, return a string.
    - For grounded outputs, return a JSON array of objects when performing reading tasks.
    - For detection tasks, return a JSON array of bounding boxes only.
    - For localized reading tasks, return the recognized text within the specified bounding box.
```

</details>

* * *

## Data Pipelines

Standardized data processing pipelines that output a unified format with bounding boxes and text at word, line, and paragraph levels.

| Pipeline                                           | Source            | Description                                              |
| -------------------------------------------------- | ----------------- | -------------------------------------------------------- |
| [**Google Vision OCR**](data/google_vision_ocr/)   | Cloud API         | Text/document/PDF modes with polygon coordinates         |
| [**Grounded LaTeX**](data/grounded_latex/)         | LaTeX sources     | Math equations with rotation variants & bbox annotations |
| [**IDL**](data/idl/)                               | Industry docs     | Industry Document Library standardization                |
| [**PubMed**](data/pubmed/)                         | Scientific papers | ~2M paper processing with failure recovery               |
| [**SynthDoG Grounding**](data/synthdog_grounding/) | Synthetic         | Multi-language doc generation (EN/ZH/JA/KO), text+bbox   |
| [**TabMe++**](data/tabmepp/)                       | Industry docs     | Azure-based OCR                                          |

All pipelines output **WebDataset tar shards** with standardized JSON:

```json
{
  "text": {
    "lines": [{"text": "Hello world", "bbox": [x1, y1, x2, y2]}],
    "text": "Full document text",
    "text2d": "Layout-preserved\ntext rendering"
  },
  "image": {"width": 1000, "height": 1400}
}
```

* * *

## Training

Full-weight fine-tuning of Vision Language Models with multi-GPU support.

**Location:** [`experiments/qwen-multigpu-sft/`](experiments/qwen-multigpu-sft/)

### Features

-   **DeepSpeed ZeRO-3** optimization for memory-efficient training
-   **WebDataset streaming** for large-scale data handling  
-   **Flexible task system** via CSV configuration
-   **Checkpoint resume** with model-only loading option

### Supported Tasks

| Task                    | Description                                                   |
| ----------------------- | ------------------------------------------------------------- |
| `reading`               | Full document OCR (text, text2d, lines+bbox, paragraphs+bbox) |
| `detection`             | Locate text regions without transcription                     |
| `localized_reading`     | Read text within a specified bounding box                     |
| `conditional_detection` | Find bounding boxes for given text queries                    |

### Quick Start

```bash
cd experiments/qwen-multigpu-sft

# Install dependencies
uv sync
uv pip install flash-attn --no-build-isolation

# Launch training (8 GPUs)
accelerate launch --config_file acc.yaml sft_clean.py \
    --train-shards "path/to/train/*.tar" \
    --val-shards "path/to/val/*.tar" \
    --output-dir ./checkpoints
```

* * *

## Evaluation

Batch OCR evaluation using vLLM for fast inference across multiple GPUs.

| Framework                                       | Description                                                                   |
| ----------------------------------------------- | ----------------------------------------------------------------------------- |
| [**vLLM OCR Eval**](experiments/vllm-ocr-eval/) | General OCR benchmarking (CER, WER, detection)                                |
| [**vLLM Fox**](experiments/vllm-fox/)           | [Fox benchmark](https://github.com/ucaslcl/Fox) for fine-grained document OCR |

**Location:** [`experiments/vllm-ocr-eval/`](experiments/vllm-ocr-eval/)

### Metrics

| Category         | Metrics                                |
| ---------------- | -------------------------------------- |
| **Text Quality** | CER, WER, ANLS, Exact Match            |
| **Detection**    | Precision, Recall, F1 @ IoU thresholds |
| **Structured**   | Combined bbox + text matching          |

### Quick Start

```bash
cd experiments/vllm-ocr-eval

# Generate predictions
uv run python run_evaluation.py \
    --model-name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --shard-path data.tar \
    --task-types reading \
    --output-types text \
    --csv-output predictions.csv

# Score results
uv run python score_text_reading.py predictions.csv
```

### Multi-GPU Evaluation

```bash
uv run python run_evaluation.py \
    --num-workers 4 \
    --shard-path "data/*.tar" \
    ...
```

* * *

## Installation

Each component has its own dependencies. We recommend using [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

```bash
# Clone the repository
git clone https://github.com/your-org/GutenOCR.git
cd GutenOCR

# Install component-specific dependencies
cd experiments/qwen-multigpu-sft && uv sync
cd experiments/vllm-ocr-eval && uv sync
```

* * *

## Citation

```bibtex
@misc{heidenreich2026gutenocrgroundedvisionlanguagefrontend,
      title={GutenOCR: A Grounded Vision-Language Front-End for Documents}, 
      author={Hunter Heidenreich and Ben Elliott and Olivia Dinica and Yosheb Getachew},
      year={2026},
      eprint={2601.14490},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.14490}, 
}

@misc{heidenreich2026pubmedocrpmcopenaccess,
      title={PubMed-OCR: PMC Open Access OCR Annotations}, 
      author={Hunter Heidenreich and Yosheb Getachew and Olivia Dinica and Ben Elliott},
      year={2026},
      eprint={2601.11425},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.11425}, 
}
```

* * *

## Acknowledgments

-   [Fox](https://github.com/ucaslcl/Fox) from UCAS --- fine-grained document understanding benchmark
-   [SynthDoG](https://github.com/clovaai/donut/tree/master/synthdog) from NAVER Corp (MIT License) --- basis for our synthetic document generation
-   Built with [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [vLLM](https://github.com/vllm-project/vllm), [DeepSpeed](https://github.com/microsoft/DeepSpeed)

* * *

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE). The released model weights are licensed under **CC-BY-NC**.

* * *

<p align="center">
  <strong>Built with ❤️ for the open-source community</strong>
</p>
