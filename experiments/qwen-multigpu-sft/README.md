# Multi-GPU Vision-Language Model Fine-Tuning

This directory contains a production-ready implementation for full-weight fine-tuning of vision-language models (VLMs) across multiple GPUs. The code supports distributed training with DeepSpeed ZeRO-3, streaming data loading via WebDataset, and flexible task configuration for OCR and document understanding tasks.

Tested on DGX systems with 8×H100 GPUs.

## Features

-   **Full-weight fine-tuning** of both vision and language components
-   **Multi-GPU training** with DeepSpeed ZeRO-3 optimization
-   **Streaming data loading** from WebDataset tar archives
-   **Task filtering** by type and output format
-   **Checkpoint management** with resume support
-   **Dry-run mode** for debugging prompts and data pipelines

## Install Dependencies

```bash
uv sync
uv pip install "flash-attn>=2.6.0,<2.7.0" --no-build-isolation
```

## Data Preparation

Prepare your training data as WebDataset tar archives with paired files:

-   `.json` files containing structured OCR data (text, bounding boxes, metadata)
-   `.png`/`.jpg` image files

Each JSON should follow the schema expected by `prompt_builder.py` (see `tasks.csv` for supported task types).

Use `split_wds_pairs_tar.py` to shard large datasets into multiple tar files for efficient streaming.

## Configuration

Edit [run.sh](run.sh) to set:

1.  `TRAIN_DATA` - glob pattern for training tar files
2.  `EVAL_DATA` - glob pattern for evaluation tar files
3.  Model name, batch size, learning rate, and other hyperparameters

Edit [acc.yaml](acc.yaml) to configure the number of GPUs (`num_processes`).

## Run Training

```bash
uv run sh run.sh
```

To resume from a checkpoint:

```bash
# List available checkpoints
python args.py --list-checkpoints <output_dir>

# Resume training
accelerate launch --config_file acc.yaml sft_clean.py \
  --resume-from-checkpoint <checkpoint_path> \
  <other args...>
```

## Debugging

Test your data pipeline without loading the full model:

```bash
python sft_clean.py \
  --tar-pattern "/path/to/train-*.tar" \
  --dry-run \
  --dry-run-samples 10 \
  --no-model
```
