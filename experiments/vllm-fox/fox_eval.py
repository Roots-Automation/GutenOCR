#!/usr/bin/env python3
"""
Unified CLI for Fox vLLM OCR evaluations.
Use --task {box,line,page,onbox} to select the evaluation type.
"""

import argparse

from eval_runner import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified vLLM-optimized evaluation for Fox benchmark English OCR")

    # Task selection
    parser.add_argument("--task", type=str, choices=["box", "line", "page", "onbox"], required=True,
                        help="OCR task type to evaluate")

    # Model arguments
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help="Qwen model name or path")
    parser.add_argument("--max_model_len", type=int, default=8192,
                       help="Maximum model sequence length")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                       help="GPU memory utilization ratio (0.0-1.0)")
    parser.add_argument("--use_openai_api", action="store_true",
                        help="Use OpenAI-compatible API (e.g., vLLM serve) instead of local vLLM. Useful for models like Nanonets.")

    # Coordinate handling
    parser.add_argument("--coord_mode", type=str, choices=["absolute", "relative"], default="absolute",
                        help="Coordinate interpretation for prompts: 'absolute' converts [0,1000] to pixels using image size; 'relative' keeps [0,1000] unchanged.")

    # Data arguments
    parser.add_argument("--gtfile_path", type=str, required=True,
                       help="Path to ground truth JSON file")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to images directory")
    parser.add_argument("--system_prompt_path", type=str, default="./default_system_prompt.txt",
                       help="Path to system prompt file")

    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size for processing (None for auto-batching)")

    # Page-only extras (accepted for all tasks; used only when task=page)
    parser.add_argument("--limit_mm_video", type=int, default=0,
                       help="Limit number of videos per prompt (use 0 to disable video)")
    parser.add_argument("--skip_mm_profiling", action="store_true",
                       help="Skip memory profiling for multimodal inputs to save memory")

    # Output arguments
    parser.add_argument("--out_file", type=str, default="./results_vllm.json",
                       help="Output file path for results")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_evaluation(args.task, args)


if __name__ == "__main__":
    main()
