"""Command line argument parsing for OCR training scripts."""

import argparse
import os
import glob


def list_checkpoints(output_dir: str) -> list[str]:
    """List available checkpoint directories in the output directory."""
    if not os.path.exists(output_dir):
        return []
    
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    # Sort by checkpoint number
    def extract_step(path):
        try:
            return int(os.path.basename(path).split('-')[1])
        except (IndexError, ValueError):
            return 0
    
    return sorted(checkpoints, key=extract_step)


def parse_sft_args():
    """Parse command line arguments for SFT training script."""
    parser = argparse.ArgumentParser(description="Multi-GPU OCR/Vision-to-Text SFT training")

    # Checkpoint management
    parser.add_argument("--list-checkpoints", default=None,
                        help="List available checkpoints in the specified output directory and exit")

    # Model and data
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-length", type=int, default=5000)
    parser.add_argument("--train-vision", action="store_true",
                        help="Whether to train the vision tower. By default, vision is frozen.")
    
    # Task filtering
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Restrict training to specific task types (e.g., --tasks reading localized_reading)")
    parser.add_argument("--output-types", nargs="+", default=None,
                        help="Restrict training to specific output types (e.g., --output-types text markdown)")
    
    # Checkpoint resumption
    parser.add_argument("--resume-from-checkpoint", default=None, 
                        help="Path to checkpoint directory to resume training from")
    parser.add_argument("--load-model-only", action="store_true",
                        help="Only load model weights, don't resume optimizer/scheduler state")

    # Data source - WebDataset streaming with tar pattern
    parser.add_argument("--tar-pattern", default="",
                        help="Glob or brace pattern for many tars, e.g. /data/shards/train-{0000..0999}.tar or /data/shards/*.tar")
    parser.add_argument("--eval-tar-pattern", default=None,
                        help="Separate tar pattern for evaluation. If not provided, uses --tar-pattern")

    # Validation samples per epoch for streaming
    parser.add_argument("--val-samples-per-epoch", type=int, default=0,
                        help="Samples used for each eval in streaming")

    # Dataloader performance settings
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--persistent-workers", type=bool, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=4)

    # Training hyperparameters
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--output-dir", default="ocr_model_out_2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation during training")
    
    # Learning rate and warmup settings
    parser.add_argument("--learning-rate", type=float, default=5e-6, 
                        help="Learning rate for training")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Number of warmup steps")

    # Dry-run and debugging options
    parser.add_argument("--dry-run", action="store_true", help="Do a dataset smoke test; print prompts and optionally generate.")
    parser.add_argument("--dry-run-samples", type=int, default=32, help="How many random prompts to preview in dry-run mode.")
    parser.add_argument("--dry-run-generate", action="store_true", help="In dry-run, also run a tiny model.generate() on a few samples.")
    parser.add_argument("--dry-run-max-new-tokens", type=int, default=24, help="Max new tokens for dry-run generation.")
    parser.add_argument("--no-model", action="store_true", help="Skip model load entirely (dry-run prompt preview only).")
    
    # Optional DeepSpeed configuration
    parser.add_argument("--deepspeed-config", type=str, default=None, help="Path to a Deepspeed JSON config (optional).")

    args = parser.parse_args()
    
    # Handle checkpoint listing
    if args.list_checkpoints:
        checkpoints = list_checkpoints(args.list_checkpoints)
        if checkpoints:
            print(f"Available checkpoints in {args.list_checkpoints}:")
            for i, checkpoint in enumerate(checkpoints, 1):
                step = os.path.basename(checkpoint).split('-')[1]
                print(f"  {i}. {checkpoint} (step {step})")
            print(f"\nTo resume from latest checkpoint, use:")
            print(f"  --resume-from-checkpoint {checkpoints[-1]}")
        else:
            print(f"No checkpoints found in {args.list_checkpoints}")
        exit(0)
    
    return args
