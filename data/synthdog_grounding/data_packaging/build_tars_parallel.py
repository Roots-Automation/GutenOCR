#!/usr/bin/env python3
"""
Parallel version of build_tars.sh using Python multiprocessing.
Processes all directories from 0000-0075 with train/validation/test splits in parallel.
"""

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple
import os


def process_single_tar(args: Tuple[str, str, str]) -> Tuple[bool, str]:
    """
    Process a single tar file creation.
    
    Args:
        args: Tuple of (core_dir, directory_num, split)
    
    Returns:
        Tuple of (success, message)
    """
    core_dir, directory_num, split = args
    
    input_dir = Path(core_dir) / directory_num / split
    output_tar = Path(core_dir) / f"{split}-{directory_num}.tar"
    
    # Skip if tar file already exists
    if output_tar.exists():
        return True, f"Skipping {output_tar.name} (already exists)"
    
    # Check if input directory exists
    if not input_dir.exists():
        return False, f"Skipping {input_dir} (directory does not exist)"
    
    try:
        # Get the directory containing build_tar.py
        script_dir = Path(__file__).parent
        build_tar_script = script_dir / "build_tar.py"
        
        # Run the build_tar.py script
        cmd = ["uv", "run", str(build_tar_script), str(input_dir), "-o", str(output_tar)]
        
        print(f"Processing {input_dir} -> {output_tar.name}")
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=script_dir
        )
        
        if result.returncode == 0:
            return True, f"Done: {output_tar.name}"
        else:
            error_msg = f"Failed {output_tar.name}: {result.stderr.strip()}"
            return False, error_msg
            
    except Exception as e:
        return False, f"Error processing {output_tar.name}: {str(e)}"


def generate_tasks(core_dir: str, start_dir: int = 0, end_dir: int = 75) -> List[Tuple[str, str, str]]:
    """Generate list of tasks (core_dir, directory_num, split)."""
    tasks = []
    splits = ["train", "validation", "test"]
    
    for i in range(start_dir, end_dir + 1):
        directory_num = f"{i:04d}"  # Zero-padded to 4 digits
        for split in splits:
            tasks.append((core_dir, directory_num, split))
    
    return tasks


def main():
    default_dir = os.environ.get("SYNTHDOG_DATA_DIR", "./outputs")
    parser = argparse.ArgumentParser(
        description="Build tar files in parallel for synthdog grounding dataset"
    )
    parser.add_argument(
        "--core-dir", 
        type=str, 
        default=default_dir,
        help=f"Core directory containing numbered subdirectories (default: {default_dir}, override with SYNTHDOG_DATA_DIR env var)"
    )
    parser.add_argument(
        "--start", 
        type=int, 
        default=0,
        help="Start directory number (default: 0)"
    )
    parser.add_argument(
        "--end", 
        type=int, 
        default=75,
        help="End directory number (default: 75)"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=None,
        help="Number of parallel workers (default: number of CPU cores)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be processed without actually running"
    )
    
    args = parser.parse_args()
    
    # Validate core directory
    core_dir = Path(args.core_dir)
    if not core_dir.exists():
        print(f"Error: Core directory does not exist: {core_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Generate tasks
    tasks = generate_tasks(str(core_dir), args.start, args.end)
    
    print(f"Found {len(tasks)} tasks to process")
    
    if args.dry_run:
        print("\nDry run - would process:")
        for core_dir_str, directory_num, split in tasks:
            input_dir = Path(core_dir_str) / directory_num / split
            output_tar = Path(core_dir_str) / f"{split}-{directory_num}.tar"
            
            status = "EXISTS" if output_tar.exists() else "MISSING" if not input_dir.exists() else "TODO"
            print(f"  {input_dir} -> {output_tar.name} [{status}]")
        return
    
    # Determine number of workers
    max_workers = args.workers or os.cpu_count()
    print(f"Using {max_workers} parallel workers")
    
    # Process tasks in parallel
    success_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(process_single_tar, task): task for task in tasks}
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                success, message = future.result()
                if success:
                    success_count += 1
                    print(f"✓ {message}")
                else:
                    error_count += 1
                    print(f"✗ {message}", file=sys.stderr)
            except Exception as e:
                error_count += 1
                _, directory_num, split = task
                print(f"✗ Exception processing {split}-{directory_num}: {e}", file=sys.stderr)
    
    print(f"\n=== Summary ===")
    print(f"Successfully processed: {success_count}")
    print(f"Errors/Skipped: {error_count}")
    print(f"Total tasks: {len(tasks)}")
    
    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
