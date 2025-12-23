"""Example script demonstrating OCR evaluation with vLLM."""

import argparse
import logging
import sys
import os
import multiprocessing as mp
from typing import List

import pandas as pd

from dataset import TarShardDataset
from prompt_builder import generate_prompt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def worker_process(gpu_id: int, tasks_chunk: List, args, return_dict: dict):
    """Worker process for data parallel evaluation on a specific GPU."""
    # Set CUDA device for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Setup logging for worker
    worker_logger = logging.getLogger(f"worker-{gpu_id}")
    worker_logger.info(f"Worker {gpu_id} starting with {len(tasks_chunk)} tasks")
    
    try:
        from predictor import OCRVLMPredictor
        from dataset import TarShardDataset
        
        # Initialize predictor for this GPU
        predictor = OCRVLMPredictor(
            model_name=args.model_name,
            max_model_len=args.max_model_len,
            tensor_parallel_size=1,  # Single GPU per worker
            gpu_memory_utilization=args.gpu_memory_utilization,
            system_prompt_path=args.system_prompt
        )
        
        # Get unique entries and initialize dataset
        seen_entries = {}
        for task in tasks_chunk:
            entry = task['entry']
            seen_entries[entry.filename] = entry
        
        all_entries = list(seen_entries.values())
        if all_entries:
            dataset = TarShardDataset(all_entries[0].shard_file)
            temp_paths = dataset.extract_images_to_temp(all_entries)
        else:
            temp_paths = {}
        
        # Prepare prompts for batch processing
        batch_prompts = []
        for task in tasks_chunk:
            entry = task['entry']
            temp_path = temp_paths.get(entry.filename)
            
            if temp_path:
                batch_prompts.append({
                    'image_path': temp_path,
                    'text_prompt': task['question'],
                    'task_info': task
                })
        
        # Run batch prediction
        predictions = predictor.predict_batch(
            batch_prompts,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        # Cleanup
        if all_entries:
            dataset.cleanup_temp_files()
        
        # Store results
        worker_results = []
        for prompt, prediction in zip(batch_prompts, predictions):
            result = {
                'gpu_id': gpu_id,
                'task_info': prompt['task_info'],
                'prediction': prediction.strip()
            }
            worker_results.append(result)
        
        return_dict[gpu_id] = worker_results
        worker_logger.info(f"Worker {gpu_id} completed {len(worker_results)} tasks")
        
    except Exception as e:
        worker_logger.error(f"Worker {gpu_id} failed: {e}")
        import traceback
        traceback.print_exc()
        return_dict[gpu_id] = []


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OCR evaluation script with vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset configuration
    parser.add_argument(
        "--shard-path", 
        type=str, 
        default="shard-00000.tar",
        help="Path to the tar shard dataset file"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=5,
        help="Number of samples to evaluate (0 for all)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Name of the model to evaluate"
    )
    parser.add_argument(
        "--system-prompt", 
        type=str, 
        default="system_prompt.txt",
        help="Path to system prompt file"
    )
    parser.add_argument(
        "--max-model-len", 
        type=int, 
        default=8192,
        help="Maximum model length"
    )
    parser.add_argument(
        "--tensor-parallel-size", 
        type=int, 
        default=1,
        help="Tensor parallel size for multi-GPU"
    )
    parser.add_argument(
        "--gpu-memory-utilization", 
        type=float, 
        default=0.9,
        help="GPU memory utilization ratio"
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=1,
        help="Number of parallel worker processes (data parallelism)"
    )
    
    # Generation configuration
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.0,
        help="Sampling temperature (0.0 for deterministic)"
    )
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=2048,
        help="Maximum tokens to generate"
    )
    
    # Task configuration
    parser.add_argument(
        "--task-types", 
        nargs="+", 
        default=['reading', 'detection', 'conditional_detection', 'localized_reading'],
        choices=['reading', 'detection', 'conditional_detection', 'localized_reading'],
        help="Task types to evaluate"
    )
    parser.add_argument(
        "--input-types",
        nargs="+",
        default=None,
        help="Input types to restrict tasks (e.g., 'image', 'image+text')"
    )
    parser.add_argument(
        "--output-types", 
        nargs="+", 
        default=['text', '[lines, box]', 'box', '[latex, box]', 'text2d', 'lines'],
        help="Output types to evaluate"
    )
    parser.add_argument(
        "--questions-per-task",
        type=int,
        default=4,
        help="Number of questions to generate per task for conditional_detection and localized_reading (deterministic based on data)"
    )
    
    # Output configuration
    parser.add_argument(
        "--csv-output", 
        type=str, 
        default="ocr_eval_results.csv",
        help="Output CSV file for detailed results"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )


def main():
    """Main evaluation script."""
    args = parse_args()
    setup_logging(args.verbose)
    
    logger.info("Starting OCR evaluation with vLLM")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Shard: {args.shard_path}")
    logger.info(f"Task types: {args.task_types}")
    logger.info(f"Input types: {args.input_types}")
    logger.info(f"Output types: {args.output_types}")
    logger.info(f"Questions per conditional/localized task: {args.questions_per_task}")
    logger.info(f"Sample limit: {args.limit if args.limit > 0 else 'all'}")

    
    try:
        # Initialize dataset
        logger.info("Loading dataset...")
        dataset = TarShardDataset(args.shard_path)
        logger.info(f"Dataset loaded with {len(dataset)} entries")
        
        # Data parallelism setup - we'll distribute tasks across multiple GPU workers
        num_workers = min(args.num_workers, 8)  # Cap at 8 GPUs
        logger.info(f"Using data parallelism with {num_workers} GPU workers")
        
        # Generate evaluation tasks using prompt_builder
        logger.info("Generating evaluation tasks...")
        limit = args.limit if args.limit > 0 else None
        entries = dataset.get_entries(limit=limit)
        tasks = []
        
        for entry in entries:
            example = entry.to_prompt_builder_format()
            
            # Generate multiple tasks per entry
            for task_type in args.task_types:
                for output_type in args.output_types:
                    for input_type in (args.input_types or [None]):
                        # Determine number of questions based on task type
                        num_questions = args.questions_per_task if task_type in ['conditional_detection', 'localized_reading'] else 1
                        
                        for question_idx in range(num_questions):
                            try:
                                # Generate deterministic seed based on entry filename, task type, output type, and question index
                                # This ensures the same questions are generated for the same data across different model runs
                                deterministic_seed = hash(f"{entry.filename}_{task_type}_{input_type}_{output_type}_{question_idx}") % (2**31)
                                
                                question, expected_answer = generate_prompt(
                                    example,
                                    allowed_tasks=[task_type],
                                    allowed_input_types=[input_type] if input_type else None,
                                    allowed_output_types=[output_type],
                                    seed=deterministic_seed
                                )
                                
                                # Skip tasks with empty box answers to avoid invalid evaluations
                                if output_type == 'box' and (not expected_answer or expected_answer == [] or expected_answer == '[]'):
                                    logger.debug(f"Skipping {task_type}/{output_type} task {question_idx} for {entry.filename}: empty box answer")
                                    continue

                                if '{box}' in question or '{text}' in question:
                                    logger.debug(f"Skipping {task_type}/{output_type} task {question_idx} for {entry.filename}: placeholder in question")
                                    continue
                                
                                tasks.append({
                                    'entry': entry,
                                    'task_type': task_type,
                                    'input_type': input_type,
                                    'output_type': output_type,
                                    'question': question,
                                    'expected_answer': expected_answer,
                                    'question_idx': question_idx  # Track which question this is
                                })
                                
                            except Exception as e:
                                logger.warning(f"Failed to generate {task_type}/{output_type} task {question_idx} for {entry.filename}: {e}")
            
        logger.info(f"Generated {len(tasks)} evaluation tasks")
        
        # Split tasks across workers
        tasks_per_worker = len(tasks) // num_workers
        task_chunks = []
        for i in range(num_workers):
            start_idx = i * tasks_per_worker
            if i == num_workers - 1:  # Last worker gets remainder
                end_idx = len(tasks)
            else:
                end_idx = (i + 1) * tasks_per_worker
            task_chunks.append(tasks[start_idx:end_idx])
        
        logger.info(f"Split {len(tasks)} tasks across {num_workers} workers: {[len(chunk) for chunk in task_chunks]}")
        
        # Run evaluation with multiprocessing
        logger.info("Running evaluation with data parallelism...")
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []
        
        # Start worker processes
        for gpu_id in range(num_workers):
            if gpu_id < len(task_chunks) and len(task_chunks[gpu_id]) > 0:
                p = mp.Process(
                    target=worker_process,
                    args=(gpu_id, task_chunks[gpu_id], args, return_dict)
                )
                p.start()
                processes.append(p)
                logger.info(f"Started worker process for GPU {gpu_id}")
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        logger.info("All worker processes completed")
        
        # Collect and process results
        all_worker_results = []
        for gpu_id in range(num_workers):
            if gpu_id in return_dict:
                all_worker_results.extend(return_dict[gpu_id])
        
        logger.info(f"Collected {len(all_worker_results)} results from workers")
        
        # Process results similar to the original evaluate_tasks method
        processed_results = []        
        for worker_result in all_worker_results:
            task_info = worker_result['task_info']
            predicted_str = worker_result['prediction']
            
            # Create result entry
            result = {
                'filename': task_info['entry'].filename,
                'task_type': task_info['task_type'],
                'output_type': task_info['output_type'],
                'question_idx': task_info.get('question_idx', 0),  # Add question index
                'question': task_info['question'],
                'expected_answer': str(task_info['expected_answer']),
                'predicted_answer': predicted_str,
            }
            processed_results.append(result)
        
        # Create results dataframe
        results_df = pd.DataFrame(processed_results)
        
        # Save to CSV
        if args.csv_output:
            results_df.to_csv(args.csv_output, index=False)
            logger.info(f"Results saved to {args.csv_output}")

        logger.info("Evaluation complete!")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Set multiprocessing start method to spawn for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    main()
