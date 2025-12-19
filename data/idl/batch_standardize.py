#!/usr/bin/env python3
"""
Batch IDL ➜ Standard Shard Converter

Processes multiple IDL directories at once, wrapping the standardize.py functionality.
Automatically discovers and converts multiple IDL sub-directories into standard tar shards.

Usage Examples:
  # Process all idl-train-* folders in a directory
  python batch_standardize.py \
      --input-base /mnt/mldata/idl/samples \
      --output-base /mnt/mldata/datasets/idl \
      --pattern "idl-train-*"

  # Process specific range with custom naming
  python batch_standardize.py \
      --input-base /mnt/mldata/idl/samples \
      --output-base /mnt/mldata/datasets/idl \
      --pattern "idl-train-0000[1-5]"

  # Dry run to validate all folders first
  python batch_standardize.py \
      --input-base /mnt/mldata/idl/samples \
      --output-base /mnt/mldata/datasets/idl \
      --pattern "idl-*" \
      --dry-run

  # Process with different settings
  python batch_standardize.py \
      --input-base /mnt/mldata/idl/samples \
      --output-base /mnt/mldata/datasets/idl \
      --pattern "idl-val-*" \
      --dpi 150 \
      --image-ext jpg \
      --max-workers 4

Dependencies:
  All dependencies from standardize.py plus:
  - concurrent.futures (built-in)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Import the standardize module
from standardize import Config, process_shard, validate_structure


def discover_input_folders(input_base: Path, pattern: str) -> List[Path]:
    """Discover input folders matching the pattern."""
    search_pattern = str(input_base / pattern)
    matching_paths = glob.glob(search_pattern)
    return sorted([Path(p) for p in matching_paths if Path(p).is_dir()])


def generate_output_path(input_dir: Path, input_base: Path, output_base: Path, 
                        suffix: str = ".tar") -> Path:
    """Generate output path by mapping input structure to output structure.
    
    Example:
      input_dir: /mnt/mldata/idl/samples/idl-train-00001
      input_base: /mnt/mldata/idl/samples
      output_base: /mnt/mldata/datasets/idl
      -> /mnt/mldata/datasets/idl/train-00001.tar
    """
    rel_path = input_dir.relative_to(input_base)
    # Remove 'idl-' prefix if present for cleaner output names
    output_name = str(rel_path).replace('idl-', '', 1)
    return output_base / f"{output_name}{suffix}"


def process_single_folder(args: Tuple[Path, Path, Config]) -> Tuple[bool, str, dict]:
    """Process a single input folder. Returns (success, message, stats)."""
    input_dir, output_path, base_config = args
    
    # Create config for this specific folder
    config = Config(
        input_dir=input_dir,
        output_path=output_path,
        dpi=base_config.dpi,
        image_ext=base_config.image_ext,
        page_base=base_config.page_base,
        bbox_format=base_config.bbox_format,
        bbox_space=base_config.bbox_space,
        dry_run=base_config.dry_run,
    )
    
    try:
        if config.dry_run:
            # Quick validation
            doc_dirs = [p for p in input_dir.iterdir() if p.is_dir()]
            valid_count = 0
            total_docs = len(doc_dirs)
            
            for doc_dir in doc_dirs:
                valid, _ = validate_structure(doc_dir, config.bbox_format)
                if valid:
                    valid_count += 1
            
            return True, f"Validation: {valid_count}/{total_docs} documents valid", {
                "docs": total_docs, "valid": valid_count, "pages": 0, "lines": 0
            }
        else:
            # Actual processing
            start_time = time.time()
            process_shard(config)
            end_time = time.time()
            
            return True, f"Completed in {end_time - start_time:.1f}s", {}
            
    except Exception as e:
        return False, f"Error: {str(e)}", {}


class BatchProcessor:
    def __init__(self, input_base: Path, output_base: Path, base_config: Config,
                 max_workers: Optional[int] = None, sequential: bool = False):
        self.input_base = input_base
        self.output_base = output_base
        self.base_config = base_config
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 4)
        self.sequential = sequential
        
    def process_folders(self, input_folders: List[Path]) -> None:
        """Process multiple folders either sequentially or in parallel."""
        if not input_folders:
            logging.warning("No input folders found to process")
            return
            
        # Ensure output base directory exists
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # Prepare arguments for processing
        tasks = []
        for input_dir in input_folders:
            output_path = generate_output_path(
                input_dir, self.input_base, self.output_base
            )
            tasks.append((input_dir, output_path, self.base_config))
        
        logging.info(f"Processing {len(tasks)} folders...")
        
        if self.sequential or len(tasks) == 1:
            self._process_sequential(tasks)
        else:
            self._process_parallel(tasks)
    
    def _process_sequential(self, tasks: List[Tuple[Path, Path, Config]]) -> None:
        """Process folders one by one."""
        successful = 0
        total_time = 0
        
        for i, (input_dir, output_path, _) in enumerate(tasks, 1):
            logging.info(f"[{i}/{len(tasks)}] Processing {input_dir.name} -> {output_path.name}")
            
            start_time = time.time()
            success, message, stats = process_single_folder((input_dir, output_path, self.base_config))
            end_time = time.time()
            
            elapsed = end_time - start_time
            total_time += elapsed
            
            if success:
                successful += 1
                logging.info(f"✓ {input_dir.name}: {message}")
            else:
                logging.error(f"✗ {input_dir.name}: {message}")
        
        logging.info(f"Batch complete: {successful}/{len(tasks)} successful, total time: {total_time:.1f}s")
    
    def _process_parallel(self, tasks: List[Tuple[Path, Path, Config]]) -> None:
        """Process folders in parallel."""
        logging.info(f"Using {self.max_workers} parallel workers")
        
        successful = 0
        start_time = time.time()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(process_single_folder, task): task 
                for task in tasks
            }
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_task):
                input_dir, output_path, _ = future_to_task[future]
                
                try:
                    success, message, stats = future.result()
                    if success:
                        successful += 1
                        logging.info(f"✓ {input_dir.name}: {message}")
                    else:
                        logging.error(f"✗ {input_dir.name}: {message}")
                        
                except Exception as exc:
                    logging.error(f"✗ {input_dir.name}: Exception occurred: {exc}")
        
        end_time = time.time()
        logging.info(f"Batch complete: {successful}/{len(tasks)} successful, total time: {end_time - start_time:.1f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch process multiple IDL directories into standard tar shards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument("--input-base", required=True, type=Path,
                       help="Base directory containing IDL folders to process")
    parser.add_argument("--output-base", required=True, type=Path,
                       help="Base directory for output tar files")
    parser.add_argument("--pattern", required=True,
                       help="Glob pattern to match input folders (e.g., 'idl-train-*')")
    
    # Processing options
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Maximum number of parallel workers (default: min(cpu_count, 4))")
    parser.add_argument("--sequential", action="store_true",
                       help="Process folders sequentially instead of in parallel")
    
    # Options passed through to standardize.py
    parser.add_argument("--dpi", type=int, default=72,
                       help="Rasterization DPI (default: 72)")
    parser.add_argument("--image-ext", choices=["png", "jpg", "tif"], default="png",
                       help="Image extension/format (default: png)")
    parser.add_argument("--page-base", type=int, choices=[0, 1], default=0,
                       help="Filename page index base (0 or 1; default: 0)")
    parser.add_argument("--bbox-format", choices=["xywh", "x1y1x2y2"], default="xywh",
                       help="Source bbox format in JSON (default: xywh)")
    parser.add_argument("--bbox-space", choices=["as_is", "pdf_to_pixel"], default="as_is",
                       help="Coordinate space for bounding boxes")
    parser.add_argument("--dry-run", action="store_true",
                       help="Validate folders only; don't create output files")
    
    # Logging
    parser.add_argument("--log-level", default="INFO",
                       help="Logging level (default: INFO)")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    # Validate input directory
    if not args.input_base.exists() or not args.input_base.is_dir():
        logging.error(f"Input base directory does not exist: {args.input_base}")
        sys.exit(1)
    
    # Discover input folders
    input_folders = discover_input_folders(args.input_base, args.pattern)
    if not input_folders:
        logging.error(f"No folders found matching pattern '{args.pattern}' in {args.input_base}")
        sys.exit(1)
    
    logging.info(f"Found {len(input_folders)} folders matching pattern '{args.pattern}':")
    for folder in input_folders:
        output_path = generate_output_path(folder, args.input_base, args.output_base)
        logging.info(f"  {folder.name} -> {output_path.name}")
    
    # Create base config
    base_config = Config(
        input_dir=Path(),  # Will be overridden per folder
        output_path=Path(),  # Will be overridden per folder
        dpi=args.dpi,
        image_ext=args.image_ext,
        page_base=args.page_base,
        bbox_format=args.bbox_format,
        bbox_space=args.bbox_space,
        dry_run=args.dry_run,
    )
    
    # Process folders
    processor = BatchProcessor(
        input_base=args.input_base,
        output_base=args.output_base,
        base_config=base_config,
        max_workers=args.max_workers,
        sequential=args.sequential
    )
    
    processor.process_folders(input_folders)


if __name__ == "__main__":
    main()
