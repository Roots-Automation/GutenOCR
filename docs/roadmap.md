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
