#!/usr/bin/env python3
"""
Unified evaluation runner for Fox OCR tasks using vLLM.
"""

import logging
import os

from vllm_ocr_predictor import OCRVLMPredictor
from common import (
    load_ground_truth,
    save_results,
    load_system_prompt,
    progress_callback,
    print_run_header,
    print_results_preview,
)
from prompt_extractors import (
    create_prompt_extractor,
)

logger = logging.getLogger(__name__)


def _init_predictor_for_task(task_type: str, args) -> OCRVLMPredictor:
    """Initialize predictor with unified kwargs across tasks (MM limits for all)."""
    limit_mm: dict = {"image": 1}
    # Apply video limit if provided (>0) for any task; 0 disables video
    if getattr(args, "limit_mm_video", 0):
        limit_mm["video"] = args.limit_mm_video

    kwargs = dict(
        model_name=args.model_name,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        system_prompt_path=args.system_prompt_path,
        limit_mm_per_prompt=limit_mm,
        skip_mm_profiling=getattr(args, "skip_mm_profiling", False),
        use_openai_api=getattr(args, "use_openai_api", False),
    )
    return OCRVLMPredictor(**kwargs)


def run_evaluation(task_type: str, args) -> None:
    """Run a complete evaluation for the given task type using shared utilities."""
    print_run_header(args, task_type.capitalize())

    system_prompt = load_system_prompt(args.system_prompt_path)
    if system_prompt is None:
        return

    logger.info("Initializing vLLM predictor...")
    predictor = _init_predictor_for_task(task_type, args)

    gts = load_ground_truth(args.gtfile_path)

    if not os.path.exists(args.image_path):
        logger.error(f"Image directory not found: {args.image_path}")
        return

    # Route all tasks through the prompt extractor with explicit coord mode
    coord_mode = getattr(args, "coord_mode", "absolute")
    extract_prompt_fn = create_prompt_extractor(coord_mode=coord_mode, model_name=args.model_name, task=task_type)

    print("Starting batch evaluation with vLLM...")

    try:
        results = predictor.evaluate_dataset(
            gt_data=gts,
            image_dir=args.image_path,
            extract_prompt_fn=extract_prompt_fn,
            batch_size=args.batch_size,
            temperature=0.0,
            max_tokens=args.max_new_tokens,
            system_prompt=system_prompt,
            progress_callback=progress_callback,
        )

        truncate = 200 if task_type == "page" else None
        print_results_preview(results, limit=3, truncate=truncate)

        save_results(results, args.out_file)

        print("\n" + "=" * 60)
        print("vLLM Evaluation completed successfully!")
        print(f"Processed {len(results)} samples")
        print(f"Results saved to: {args.out_file}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise
