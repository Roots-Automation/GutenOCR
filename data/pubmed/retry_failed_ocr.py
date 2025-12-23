#!/usr/bin/env python3
"""
PubMed Failed OCR Retry Orchestrator

This script orchestrates the retry of PubMed PDFs that failed during the initial OCR run due to rate limiting.
It analyzes logs from previous runs to identify failures, creates a failures list, and then calls the original
process_pubmed_ocr.py script with the failures list to reprocess only the failed files.

Usage:
    python retry_failed_ocr.py --log-dir /path/to/logs --output-dir /path/to/output --num-workers N
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def analyze_logs_for_failures(log_dir: Path) -> set[str]:
        """Analyze log files to identify failed PDFs by comparing success list against total list"""
        success_stems = set()
        all_pdfs = set()
        
        # First, get all PDFs that should have been processed from parallel.log
        parallel_log = log_dir / "parallel.log"
        if parallel_log.exists():
            print(f"Reading total PDF list from {parallel_log}...")
            with open(parallel_log, 'r') as f:
                for line in f:
                    if "oa_pdf/" in line and ".pdf" in line:
                        # Extract PDF path from parallel log format
                        try:
                            pdf_match = line.split("oa_pdf/")
                            if len(pdf_match) > 1:
                                pdf_path = "oa_pdf/" + pdf_match[1].split()[0]
                                pdf_name = Path(pdf_path).stem
                                all_pdfs.add(pdf_name)
                        except:
                            continue
        
        # Get PDFs that were truly successful using the shard manifest and shard content
        output_dir = log_dir.parent  # Assuming logs are in output_dir/logs
        data_dir = output_dir / "data"
        zero_page_failures = set()  # Track PDFs that were "successful" but had 0 OCR content
        
        # Try to use shard manifest for efficient analysis
        manifest_path = output_dir / "shard_manifest.json"
        
        if manifest_path.exists():
            print(f"Using shard manifest for efficient failure analysis...")
            import json
            import tarfile
            
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                data_manifest = manifest.get('data', {})
                processed_count = 0
                
                # Process each shard to check for successful vs zero-page files
                for shard_id, filenames in data_manifest.items():
                    shard_path = data_dir / f"shard_{shard_id}.tar.gz"
                    
                    if not shard_path.exists():
                        print(f"Warning: Shard {shard_id} referenced in manifest but file not found")
                        continue
                    
                    try:
                        with tarfile.open(shard_path, 'r:gz') as tar:
                            # Get JSON files from this shard
                            json_files = [f for f in filenames if f.endswith('.json')]
                            
                            for json_filename in json_files:
                                try:
                                    # Extract PDF stem from filename like "[stem]_[page].json"
                                    if '_' in json_filename:
                                        pdf_stem = json_filename.split('_')[0]
                                        
                                        # Extract and check JSON content
                                        json_member = tar.getmember(json_filename)
                                        json_file = tar.extractfile(json_member)
                                        
                                        if json_file:
                                            # Check file size first
                                            json_content = json_file.read()
                                            if len(json_content) < 300:  # Empty OCR files are small
                                                zero_page_failures.add(pdf_stem)
                                            else:
                                                try:
                                                    data = json.loads(json_content.decode('utf-8'))
                                                    # Check if OCR content is empty
                                                    text_data = data.get('text', {})
                                                    if (not text_data.get('lines', []) and 
                                                        not text_data.get('words', []) and 
                                                        not text_data.get('paragraphs', [])):
                                                        zero_page_failures.add(pdf_stem)
                                                    else:
                                                        success_stems.add(pdf_stem)
                                                except json.JSONDecodeError:
                                                    # Corrupted JSON is also a failure
                                                    zero_page_failures.add(pdf_stem)
                                        
                                        processed_count += 1
                                        if processed_count % 1000 == 0:
                                            print(f"  Processed {processed_count} files from shards...")
                                            
                                except Exception as e:
                                    continue
                    
                    except tarfile.TarError as e:
                        print(f"Warning: Could not read shard {shard_id}: {e}")
                        continue
                
                print(f"Finished analyzing {processed_count} JSON files from {len(data_manifest)} shards")
                
            except Exception as e:
                print(f"Error reading shard manifest: {e}")
                print("Falling back to success.log approach...")
                # Fallback to success.log approach
                success_log = log_dir / "success.log"
                if success_log.exists():
                    with open(success_log, 'r') as f:
                        for line in f:
                            if line.startswith("OK: oa_pdf/"):
                                pdf_path = line.replace("OK: ", "").strip()
                                pdf_name = Path(pdf_path).stem
                                success_stems.add(pdf_name)
        
        else:
            print("Shard manifest not found, falling back to success.log")
            success_log = log_dir / "success.log"
            if success_log.exists():
                with open(success_log, 'r') as f:
                    for line in f:
                        if line.startswith("OK: oa_pdf/"):
                            pdf_path = line.replace("OK: ", "").strip()
                            pdf_name = Path(pdf_path).stem
                            success_stems.add(pdf_name)
        
        # Also check for any explicit error logs
        failed_explicit = set()
        for log_filename in ["process_log.txt"]:
            log_file = log_dir / log_filename
            if log_file.exists():
                print(f"Checking for explicit failures in {log_file}...")
                with open(log_file, 'r') as f:
                    for line in f:
                        if "Failed to process" in line or "ERROR" in line:
                            if "oa_pdf/" in line:
                                try:
                                    pdf_path = line.split("oa_pdf/")[1].split()[0]
                                    pdf_name = Path(pdf_path).stem
                                    failed_explicit.add(pdf_name)
                                except:
                                    continue
        
        # Calculate failures: all PDFs minus successful ones, plus zero-page failures
        actual_failures = all_pdfs - success_stems
        actual_failures.update(failed_explicit)  # Add any explicit failures
        actual_failures.update(zero_page_failures)  # Add zero-page "successes" as failures
        
        print(f"Total PDFs to process: {len(all_pdfs)}")
        print(f"Successfully processed with content: {len(success_stems)}")
        print(f"Zero-page failures (rate limited): {len(zero_page_failures)}")
        print(f"Explicit failures found: {len(failed_explicit)}")
        print(f"Net failures (to retry): {len(actual_failures)}")
        
        return actual_failures
def create_failures_list(failed_stems: Set[str], output_path: Path) -> None:
    """Create a failures list file from the set of failed stems."""
    logger.info(f"Creating failures list at {output_path}")
    
    with open(output_path, 'w') as f:
        for stem in sorted(failed_stems):
            f.write(f"{stem}\n")
    
    logger.info(f"Created failures list with {len(failed_stems)} entries")


def run_process_pubmed_ocr(input_dir: Path, output_dir: Path, failures_list_path: Path, num_workers: int) -> bool:
    """Run the original process_pubmed_ocr.py script with the failures list."""
    logger.info(f"Running process_pubmed_ocr.py with {num_workers} workers")
    
    # Path to the process_pubmed_ocr.py script
    script_path = Path(__file__).parent / "process_pubmed_ocr.py"
    
    # Build the command
    cmd = [
        sys.executable,
        str(script_path),
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--failures-list", str(failures_list_path),
        "--num-workers", str(num_workers)
    ]
    
    logger.info(f"Executing command: {' '.join(cmd)}")
    
    try:
        # Run the command and stream output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("process_pubmed_ocr.py completed successfully")
            return True
        else:
            logger.error(f"process_pubmed_ocr.py failed with return code {return_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error running process_pubmed_ocr.py: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Orchestrate retry of failed PubMed OCR processing")
    parser.add_argument("--log-dir", type=Path, required=True,
                       help="Directory containing logs from previous OCR runs")
    parser.add_argument("--input-dir", type=Path, required=True,
                       help="Input directory containing PDFs (same as original run)")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for processed files (same as original run)")
    parser.add_argument("--num-workers", type=int, default=10,
                       help="Number of worker processes (default: 10 for rate limiting)")
    parser.add_argument("--failures-list-path", type=Path, default=None,
                       help="Path to save/load the failures list file (default: log_dir/failures_list.txt)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Only analyze failures, don't actually run the retry process")
    
    args = parser.parse_args()
    
    # Set default failures list path if not provided
    if args.failures_list_path is None:
        args.failures_list_path = args.log_dir / "failures_list.txt"
    
    # Step 1: Analyze logs to identify failures
    logger.info("Step 1: Analyzing logs for failures...")
    failed_stems = analyze_logs_for_failures(args.log_dir)
    
    if not failed_stems:
        logger.info("No failed files found in logs. Nothing to retry.")
        return 0
    
    # Step 2: Create failures list file
    logger.info("Step 2: Creating failures list file...")
    create_failures_list(failed_stems, args.failures_list_path)
    
    if args.dry_run:
        logger.info(f"Dry run complete. Failures list saved to {args.failures_list_path}")
        return 0
    
    # Step 3: Run the original process_pubmed_ocr.py script with failures list
    logger.info("Step 3: Running process_pubmed_ocr.py with failures list...")
    success = run_process_pubmed_ocr(args.input_dir, args.output_dir, args.failures_list_path, args.num_workers)
    
    if success:
        logger.info("Retry processing completed successfully!")
        return 0
    else:
        logger.error("Retry processing failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
