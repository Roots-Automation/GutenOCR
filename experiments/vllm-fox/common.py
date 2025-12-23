#!/usr/bin/env python3
"""
Shared utilities for Fox vLLM OCR evaluations.
Contains IO helpers, progress reporting, and standardized printing.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


def load_ground_truth(gtfile_path: str) -> list[dict]:
    """Load ground truth data from JSON file with logging."""
    if not os.path.exists(gtfile_path):
        raise FileNotFoundError(f"Ground truth file not found: {gtfile_path}")
    with open(gtfile_path, 'r', encoding='utf-8') as f:
        gts = json.load(f)
    logger.info(f"Loaded {len(gts)} test samples")
    return gts


def save_results(results: list[dict], output_path: str):
    """Save evaluation results to JSON file with logging."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file_obj:
        json.dump(results, file_obj, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Total samples processed: {len(results)}")


def load_system_prompt(system_prompt_path: str) -> str | None:
    """Read system prompt file and return its content; log basic info."""
    if not os.path.exists(system_prompt_path):
        logger.error(f"System prompt file not found: {system_prompt_path}")
        return None
    with open(system_prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read().strip()
    print("System prompt loaded:")
    print(f"Length: {len(system_prompt)} characters")
    print("=" * 50)
    return system_prompt


def progress_callback(current: int, total: int):
    """Progress callback for batch processing (logs every 10 items and at end)."""
    if current % 10 == 0 or current == total:
        logger.info(f"Progress: {current}/{total} ({100.0 * current / total:.1f}%)")


def print_run_header(args, task_name: str):
    """Standardized header for evaluations."""
    print("=" * 60)
    print(f"Fox Benchmark - English {task_name} OCR Evaluation (vLLM Optimized)")
    print(f"Model: {args.model_name}")
    print("=" * 60)
    print(f"Ground truth file: {args.gtfile_path}")
    print(f"Images directory: {args.image_path}")
    print(f"System prompt: {args.system_prompt_path}")
    print(f"Output file: {args.out_file}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Batch size: {'Auto (vLLM managed)' if getattr(args, 'batch_size', None) is None else args.batch_size}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print("=" * 60)


def print_results_preview(results: list[dict], limit: int = 3, truncate: int | None = None):
    """Print a preview of first N results, optionally truncating long text fields."""
    print("\nSample Results (first %d):" % min(limit, len(results)))
    print("=" * 50)
    for i, result in enumerate(results[:limit]):
        print(f"\nSample {i+1}:")
        print(f"Image: {result.get('image')}")
        print(f"Question: {result.get('question')}")
        answer = result.get('answer', '')
        label = result.get('label', '')
        print(f"Generated length: {len(answer)} chars")
        print(f"Ground truth length: {len(label)} chars")
        print("-" * 30)
        if truncate:
            print(f"Generated answer: {answer[:truncate]}{'...' if len(answer) > truncate else ''}")
        else:
            print(f"Generated answer: {answer}")
        print("-" * 30)
        if truncate:
            print(f"Ground truth label: {label[:truncate]}{'...' if len(label) > truncate else ''}")
        else:
            print(f"Ground truth label: {label}")
        print("=" * 50)
