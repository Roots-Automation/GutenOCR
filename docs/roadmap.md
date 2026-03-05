# Roadmap

This roadmap breaks down planned features and improvements we're thinking about.
If anything strikes your interest, please reach out on GitHub Discussions or open an issue!

## Modeling - Training

> Anything that improves training efficiency, stability, or quality. Not for new tasks, but core modeling improvements.

- Sequence packing for more efficient training
- Variable resolution sampling at training time (resizing images to different resolutions)
- Training with Transformers v5+
- Recipe demonstration with other bases (e.g., Qwen-3)
- Smoother multi-stage training recipes

## Modeling - Inference

> Anything that improves inference efficiency, stability, or quality. Primarily, different backends or inference scenarios.

- Support for llama.cpp

## Data and Tasks

> New data sources, new tasks, or improvements to existing data and tasks.

- SynthDoG Grounding
  - Different color paper backgrounds
  - Different color fonts
  - More fonts, backgrounds, etc.
  - Expose words (in additional to line boxes)
  - Expose block boxes (groups of lines)
- Semantic conditional detection support
- Higher-order layout features w/ grounding in underlying text
- Natural language descriptions for images, figures

## Benchmarking

> New benchmarks, improvements to existing benchmarks, or better benchmarking infrastructure.

- Standardized throughput benchmarks (on key hardware configurations: T4, A100, H100, other?)
- Reproducable OmniDocBench evals in-repo
- Other benchmarks?
- Other models?
- Public leaderboard

---

## Modeling — Training (additional items)

> Additions to the training roadmap: reproducibility improvements.

- **Data mixing & curriculum strategy** `[medium]`
  - Documented recipe: synthetic → clean real → noisy real; simple tasks → complex tasks
  - Mixing ratios across data sources (SynthDoG, PubMed, IDL, etc.)
  - Configurable in training args, not hardcoded

- **Experiment tracking integration** `[medium]`
  - W&B / MLflow callbacks in training loop
  - Automatic logging of loss curves, eval metrics, hyperparameters
  - Low-effort, high-value for reproducibility

## Modeling — Inference (additional items)

> Additions to the inference roadmap: quantization, confidence, serving, and export.

- **Quantization recipes with quality benchmarks** `[high]`
  - INT4 (AWQ/GPTQ), INT8 for 3B/7B/1B models
  - Published tradeoff table: CER/WER delta vs. throughput gain on T4/A100/H100
  - Quantized variants on HuggingFace Hub

- **Confidence / uncertainty estimation** `[high]`
  - Token-level logprobs → aggregate to word/line/page confidence scores
  - Critical for production: flag low-confidence outputs for human review
  - Add `--return-confidence` flag to evaluation and inference scripts

- **Reference serving example** `[medium]`
  - Simple FastAPI script wrapping vLLM (`examples/serve.py`)
  - Not production-grade, but shows: image upload → JSON response
  - Dockerfile for easy local testing

- **ONNX / TensorRT export** `[future]`
  - Especially valuable for distilled 0.5B/1B models
  - Enables deployment in environments without Python/PyTorch

## Document Understanding Tasks

> New task types that expand GutenOCR from basic OCR to full document understanding.

- **Table structure recognition** `[high]`
  - New task type: `table_reading` — output HTML `<table>` / markdown / CSV from table regions
  - Cell-level grounding: row/column indices, merged cell spans, header detection
  - Synthetic table generation pipeline in SynthDoG (borders, borderless, merged cells, nested headers)
  - Scoring: TEDS (Tree-Edit-Distance-based Similarity), cell-level F1
  - Training data: synthetic tables + PubTables-1M + FinTabNet + SciTSR

- **Key-value / field extraction** `[high]`
  - Task: given a form image, extract `{"field": "value"}` pairs with grounding
  - Frame as conditional detection variant + reading
  - Training data: FUNSD, CORD, SROIE, synthetic forms
  - Especially relevant for insurance (claims forms), financial (invoices), healthcare (intake forms)

- **Reading order prediction** `[high]`
  - Explicit task: given a page image, output ordered region sequence with bounding boxes
  - Implicit: improve text2d output for complex layouts (multi-column, sidebars, footnotes)
  - Training data: DocLayNet, PubLayNet reading order annotations
  - Scoring: Kendall's tau or Spearman correlation on region ordering

- **Handwritten text support** `[high]`
  - Training data: IAM Handwriting Database, RIMES, CVL, synthetic handwriting
  - SynthDoG extension: handwriting-style fonts + stroke variation augmentation
  - Especially important for healthcare (prescriptions, notes) and insurance (claim annotations)
  - Start with printed+handwritten mixed documents (the real-world case)

- **Scene text / in-the-wild OCR** `[medium]`
  - Training data: ICDAR 2015/2019, TextOCR, TotalText, CTW1500, Open Images text
  - Different augmentation profile: perspective distortion, curved text, partial occlusion, variable lighting
  - Expands GutenOCR from document-only to universal OCR

- **Document classification** `[medium]`
  - Page-level: table of contents / body / bibliography / figure / form / table
  - Document-level: invoice / receipt / contract / letter / scientific paper / medical record
  - Prompt-based (no architecture change needed), new task type in `tasks.csv`

- **Multi-page document context** `[future]`
  - Cross-page awareness: running headers/footers, page numbers, table continuations
  - Approach: page-level OCR + lightweight aggregation model, or multi-image prompts
  - "Extract all tables from this 50-page PDF" as a first-class workflow

- **Figure/chart awareness** `[future]`
  - Model must learn to identify and skip (or describe) non-text regions
  - Generate synthetic documents with placeholder figures + captions
  - Ties into existing roadmap item "Natural language descriptions for images/figures"

## Data Augmentation (SynthDoG)

> Improvements to the synthetic data generation pipeline to better match real-world document diversity.

- **Realistic document noise augmentation** `[high]`
  - Scanning: moiré, skew, perspective, book spine shadow, uneven lighting
  - Physical: coffee stains, crumples, hole punches, staple shadows, fax artifacts, aged paper
  - Copy: photocopier noise, low-toner streaks, compression artifacts
  - Current effects (blur, contrast) are basic compared to real-world degradation

- **Complex layout generation** `[medium]`
  - Beyond grid/grid_stack: headers, footers, page numbers, footnotes, sidebars, watermarks, stamps
  - Use real PDF layouts as templates (extract structure from PubMed/IDL, re-render with synthetic text)
  - Multi-font per document (heading sans-serif, body serif, code monospace)

## Benchmarking (additional items)

> Additions to the benchmarking roadmap: robustness, comparisons, and quality gates.

- **Robustness benchmarks** `[high]`
  - Systematic perturbation suite: blur, noise, rotation, low DPI, partial occlusion, compression
  - Applied to GroundingBench images → measure degradation curves
  - No existing OCR benchmark does this well — potential research contribution

- **Commercial API comparison** `[high]`
  - Benchmark against Google Document AI, Azure Document Intelligence, AWS Textract
  - Same test set, same metrics, apples-to-apples
  - Publish results — this is what adoption decisions hinge on

- **Cross-dataset generalization eval** `[medium]`
  - Evaluate on data completely outside training distribution
  - Historical newspapers, handwritten notes, scene text, forms
  - Honestly measures domain gaps

- **Regression testing in CI** `[medium]`
  - Automated quality gates: new model ≥ previous on benchmark suite
  - Small eval suite runs on PRs touching training/inference code
  - Prevents silent quality regressions

- **Failure mode taxonomy** `[medium]`
  - Systematic error analysis: small text, dense tables, math, rotated, low contrast, overlapping text, stamps/watermarks
  - Drives targeted data collection
  - Publish as part of model card

- **Domain-specific benchmarks** `[future]`
  - Insurance: policy documents, claims forms, certificates of insurance
  - Financial: invoices, bank statements, tax forms
  - Healthcare: lab reports, prescriptions, discharge summaries
  - Use GroundingBench's diversity sampling methodology to build these

## Developer Experience

> Improvements to onboarding, usability, and integration with existing document processing ecosystems.

- **Unified CLI** — `gutenocr {generate, train, eval, serve}` `[medium]`
  - Single entry point instead of scattered scripts with separate environments
  - Dramatically improves onboarding

- **End-to-end tutorials** `[medium]`
  - "PDF to structured JSON in 10 minutes" (inference with released models)
  - "Fine-tune on your domain in 30 minutes" (LoRA on single GPU)
  - "Build a custom benchmark" (using GroundingBench's diversity sampling)

- **Model card / datasheet** `[medium]`
  - Formal model card: intended use, limitations, known failure modes, performance by document type
  - Dataset documentation for each data source
  - Expected for responsible AI release

- **Output format interoperability** `[future]`
  - hOCR, ALTO XML, PAGE XML converters (standard document processing formats)
  - Searchable PDF generation (overlay invisible OCR text on original)
  - Enables integration with existing DMS, search indices, archival systems

## Suggested Priority Order

> No timelines — just impact ranking. Priority annotations: `[high]`, `[medium]`, `[future]`.

**Highest impact — tackle first:**
1. Table structure recognition (task + synthetic data + scoring) — critical for all three verticals
2. LoRA / QLoRA support — unlocks community + domain adaptation
3. Realistic document noise augmentation — immediate training quality boost
4. Confidence / uncertainty estimation — production requirement
5. Key-value / field extraction — direct vertical value (forms, invoices, claims)

**High impact — natural follow-ons:**
6. Robustness benchmarks — validates noise augmentation work
7. Knowledge distillation (1B GPU model) — depends on having a strong 7B teacher
8. Reading order (explicit + implicit) — complex layouts are a real pain point
9. Handwritten text — healthcare vertical needs this badly
10. Quantization recipes — pairs with distillation work
11. Commercial API comparison — adoption driver

**Important but less urgent:**
12. Complex layout generation (SynthDoG) — supports reading order work
13. Domain-specific benchmarks — insurance, financial, healthcare
14. Scene text / in-the-wild — broadest scope expansion
15. Distilled 0.5B edge model — depends on 1B success
16. Multi-page context, document classification, unified CLI, output format interop, ONNX export, figure/chart awareness

**Ongoing (do whenever convenient):**
- Experiment tracking integration
- Regression testing in CI
- Failure mode taxonomy
- Tutorials and model cards
- Reference serving example
