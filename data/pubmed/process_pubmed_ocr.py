#!/usr/bin/env python3
"""
PubMed OCR Processing Script

This script processes PubMed PDFs from a local directory structure with Google Vision OCR,
creating:
1. Raw PDF files with OCR JSON  
2. PNG images with OCR JSON (with updated image metadata)

Directory structure:
input_dir/[hex]/[hex]/filename.pdf -> output_dir/
├── raw/
│   └── filename/
│       ├── filename_1.pdf
│       ├── filename_1.json
│       ├── filename_2.pdf
│       ├── filename_2.json
│       └── ...
└── data/
    └── filename/
        ├── filename_1.png
        ├── filename_1.json
        ├── filename_2.png
        ├── filename_2.json
        └── ...

Usage:
    python process_pubmed_ocr.py --input-dir /path/to/input --output-dir /path/to/output [--max-files N] [--start-index N]
"""

import argparse
import copy
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import fitz  # PyMuPDF
    from PIL import Image
    from google.cloud import vision, storage
    from data.google_vision_ocr.google_vision_ocr_extraction import GoogleVisionOCRExtractor
    from data.pubmed.format_ocr_json import create_ocr_formatter
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)

# Configuration - will be updated in main() based on args
LOG_DIR = Path(os.environ.get("PUBMED_OCR_LOG_DIR", "/tmp/pubmed_ocr_logs"))

# ---------- utilities ----------

def _mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _run(cmd: List[str], cwd: Optional[Path] = None, timeout: Optional[int] = None) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate(timeout=timeout)
    return p.returncode, out, err

def _rasterize_pdf(pdf_path: Path, out_dir: Path, pdf_stem: str, dpi: int = 200) -> int:
    """Rasterize PDF to PNG images at specified DPI."""
    _mkdir(out_dir)
    temp_prefix = out_dir / pdf_path.stem
    code, _, _ = _run(["pdftocairo", "-png", "-r", str(dpi), str(pdf_path), str(temp_prefix)])
    if code != 0:
        code2, _, _ = _run(["convert", "-density", str(dpi), str(pdf_path), str(temp_prefix) + "_%d.png"])
        if code2 != 0:
            raise RuntimeError("Rasterization failed")
        # Rename ImageMagick format files to consistent format
        for img_file in out_dir.glob(f"{pdf_path.stem}_*.png"):
            page_num = int(img_file.stem.split('_')[-1])
            new_name = out_dir / f"{pdf_stem}_p{page_num:02d}.png"
            img_file.rename(new_name)
        return len(list(out_dir.glob(f"{pdf_stem}_p*.png")))
    else:
        # Rename pdftocairo format files to consistent format
        for img_file in out_dir.glob(f"{pdf_path.stem}-*.png"):
            page_num = int(img_file.stem.split('-')[-1])
            new_name = out_dir / f"{pdf_stem}_p{page_num:02d}.png"
            img_file.rename(new_name)
        return len(list(out_dir.glob(f"{pdf_stem}_p*.png")))

def _resize_image_to_72dpi(image_path: Path, output_path: Path):
    """Resize image from 200 DPI to 72 DPI for storage."""
    img = Image.open(image_path)
    # Calculate new size: 72/200 = 0.36 scale factor
    scale_factor = 72 / 200
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)
    
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    # Set DPI metadata to 72
    resized_img.save(output_path, dpi=(72, 72))
    return scale_factor

def _scale_bounding_boxes(ocr_data: Dict, scale_factor: float) -> Dict:
    """
    Scale bounding box coordinates in OCR data by the given scale factor.
    
    Args:
        ocr_data: OCR data dictionary containing text with lines, words, paragraphs
        scale_factor: Scale factor to apply (e.g., 0.36 for 200->72 DPI conversion)
    
    Returns:
        New OCR data dictionary with scaled bounding boxes
    """
    scaled_data = copy.deepcopy(ocr_data)
    
    # Scale coordinates in the text structure
    if 'text' in scaled_data:
        text_data = scaled_data['text']
        
        # Scale lines
        if 'lines' in text_data:
            for line in text_data['lines']:
                if 'box' in line:
                    line['box'] = [int(coord * scale_factor) for coord in line['box']]
        
        # Scale words  
        if 'words' in text_data:
            for word in text_data['words']:
                if 'box' in word:
                    word['box'] = [int(coord * scale_factor) for coord in word['box']]
        
        # Scale paragraphs
        if 'paragraphs' in text_data:
            for paragraph in text_data['paragraphs']:
                if 'box' in paragraph:
                    paragraph['box'] = [int(coord * scale_factor) for coord in paragraph['box']]
    
    return scaled_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"pubmed_ocr_{int(time.time())}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TarShardManager:
    """Manager for saving files to compressed tar shards with manifest tracking."""
    
    def __init__(self, output_dir: Path, worker_id: int = 0, shard_size: int = 2048):
        self.output_dir = output_dir
        self.worker_id = worker_id
        self.shard_size = shard_size  # Number of document pairs per shard
        
        # Create directories
        self.raw_dir = output_dir / "raw"
        self.data_dir = output_dir / "data"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Shard tracking
        self.current_raw_shard_count = 0
        self.current_data_shard_count = 0
        self.current_raw_shard_id = None
        self.current_data_shard_id = None
        self.raw_shard_counter = 0
        self.data_shard_counter = 0
        self.raw_tar = None
        self.data_tar = None
        
        # Manifest tracking - maps shard_id -> list of filenames
        self.raw_manifest = {}
        self.data_manifest = {}
        
        # Thread-safe lock for file operations
        self.file_lock = threading.Lock()
        
        # Initialize first shards
        self._init_new_raw_shard()
        self._init_new_data_shard()
    

    
    def _init_new_raw_shard(self):
        """Initialize a new raw shard tar file."""
        if self.raw_tar:
            self.raw_tar.close()
        
        self.current_raw_shard_id = f"w{self.worker_id:03d}_{self.raw_shard_counter:05d}"
        raw_shard_path = self.raw_dir / f"shard_{self.current_raw_shard_id}.tar.gz"
        self.raw_tar = tarfile.open(raw_shard_path, 'w:gz', compresslevel=6)
        self.current_raw_shard_count = 0
        self.raw_manifest[self.current_raw_shard_id] = []
        self.raw_shard_counter += 1
        logger.info(f"Worker {self.worker_id}: Initialized raw shard {self.current_raw_shard_id}")
    
    def _init_new_data_shard(self):
        """Initialize a new data shard tar file."""
        if self.data_tar:
            self.data_tar.close()
        
        self.current_data_shard_id = f"w{self.worker_id:03d}_{self.data_shard_counter:05d}"
        data_shard_path = self.data_dir / f"shard_{self.current_data_shard_id}.tar.gz"
        self.data_tar = tarfile.open(data_shard_path, 'w:gz', compresslevel=6)
        self.current_data_shard_count = 0
        self.data_manifest[self.current_data_shard_id] = []
        self.data_shard_counter += 1
        logger.info(f"Worker {self.worker_id}: Initialized data shard {self.current_data_shard_id}")
    
    def save_raw_files(self, pdf_path: Path, json_path: Path, pdf_stem: str, page_num: int):
        """Save PDF and JSON files to raw shard."""
        with self.file_lock:
            # Check if we need a new shard
            if self.current_raw_shard_count >= self.shard_size:
                logger.info(f"Worker {self.worker_id}: Raw shard {self.current_raw_shard_id} complete with {self.current_raw_shard_count} document pairs")
                self._init_new_raw_shard()
            
            # Add files to tar
            pdf_arcname = f"{pdf_stem}_{page_num}.pdf"
            json_arcname = f"{pdf_stem}_{page_num}.json"
            
            self.raw_tar.add(pdf_path, arcname=pdf_arcname)
            self.raw_tar.add(json_path, arcname=json_arcname)
            
            # Update manifest
            self.raw_manifest[self.current_raw_shard_id].extend([pdf_arcname, json_arcname])
            
            self.current_raw_shard_count += 1
            logger.debug(f"Worker {self.worker_id}: Added {pdf_arcname} and {json_arcname} to raw shard {self.current_raw_shard_id} ({self.current_raw_shard_count}/{self.shard_size})")
    
    def save_data_files(self, png_path: Path, json_path: Path, pdf_stem: str, page_num: int):
        """Save PNG and JSON files to data shard."""
        with self.file_lock:
            # Check if we need a new shard
            if self.current_data_shard_count >= self.shard_size:
                logger.info(f"Worker {self.worker_id}: Data shard {self.current_data_shard_id} complete with {self.current_data_shard_count} document pairs")
                self._init_new_data_shard()
            
            # Add files to tar
            png_arcname = f"{pdf_stem}_{page_num}.png"
            json_arcname = f"{pdf_stem}_{page_num}.json"
            
            self.data_tar.add(png_path, arcname=png_arcname)
            self.data_tar.add(json_path, arcname=json_arcname)
            
            # Update manifest
            self.data_manifest[self.current_data_shard_id].extend([png_arcname, json_arcname])
            
            self.current_data_shard_count += 1
            logger.debug(f"Worker {self.worker_id}: Added {png_arcname} and {json_arcname} to data shard {self.current_data_shard_id} ({self.current_data_shard_count}/{self.shard_size})")
    
    def get_manifest_data(self) -> Dict:
        """Get the current manifest data for this worker."""
        return {
            'raw': dict(self.raw_manifest),
            'data': dict(self.data_manifest)
        }
    
    def close(self):
        """Close all open tar files and clean up lock files."""
        with self.file_lock:
            if self.raw_tar:
                self.raw_tar.close()
                # Clean up lock file
                lock_path = self.raw_dir / f".shard_{self.current_raw_shard_id}.lock"
                lock_path.unlink(missing_ok=True)
                logger.info(f"Worker {self.worker_id}: Closed raw shard {self.current_raw_shard_id} with {self.current_raw_shard_count} document pairs")
            if self.data_tar:
                self.data_tar.close()
                # Clean up lock file
                lock_path = self.data_dir / f".shard_{self.current_data_shard_id}.lock"
                lock_path.unlink(missing_ok=True)
                logger.info(f"Worker {self.worker_id}: Closed data shard {self.current_data_shard_id} with {self.current_data_shard_count} document pairs")


def load_shard_manifest(output_dir: Path) -> Dict:
    """Load the shard manifest JSON file."""
    manifest_path = output_dir / "shard_manifest.json"
    if not manifest_path.exists():
        return {}
    
    try:
        with open(manifest_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def check_pdf_already_processed(pdf_stem: str, output_dir: Path) -> bool:
    """
    Check if a PDF has already been successfully processed by looking in shard manifest
    and checking for actual OCR content (not empty due to rate limiting).
    
    Args:
        pdf_stem: PDF filename stem
        output_dir: Output directory containing data/ subdirectory and shard_manifest.json
    
    Returns:
        True if PDF appears to have been successfully processed with content, False otherwise
    """
    # Load shard manifest
    manifest = load_shard_manifest(output_dir)
    if not manifest or 'data' not in manifest:
        return False
    
    data_dir = output_dir / "data"
    if not data_dir.exists():
        return False
    
    # Look for files matching this PDF stem in the manifest
    found_files = []
    shard_to_check = None
    
    for shard_id, filenames in manifest['data'].items():
        matching_files = [f for f in filenames if f.startswith(f"{pdf_stem}_") and f.endswith('.json')]
        if matching_files:
            found_files.extend(matching_files)
            shard_to_check = shard_id
            break  # PDF should only be in one shard
    
    if not found_files or shard_to_check is None:
        return False
    
    # Check shard file exists (shard_to_check is now a worker-specific string ID)
    shard_path = data_dir / f"shard_{shard_to_check}.tar.gz"
    if not shard_path.exists():
        return False
    
    # Check if at least one page has actual OCR content by examining the tar
    try:
        with tarfile.open(shard_path, 'r:gz') as tar:
            for json_filename in found_files:
                try:
                    # Extract JSON file data
                    json_member = tar.getmember(json_filename)
                    json_file = tar.extractfile(json_member)
                    if json_file:
                        data = json.load(json_file)
                        # Check if this page has actual OCR content
                        text_data = data.get('text', {})
                        if (text_data.get('lines', []) or 
                            text_data.get('words', []) or 
                            text_data.get('paragraphs', [])):
                            # Found at least one page with actual content
                            return True
                except (KeyError, json.JSONDecodeError, tarfile.TarError):
                    continue
    except (tarfile.TarError, FileNotFoundError):
        return False
    
    # All JSON files exist but have no OCR content (likely rate limited)
    return False


def list_all_pdfs_in_directory(input_dir: Path) -> List[Path]:
    """List all PDF files in the input directory with hex structure."""
    logger.info(f"Scanning for PDFs in directory: {input_dir}")
    
    pdf_files = []
    # Walk through hex/hex structure
    for hex1_dir in input_dir.iterdir():
        if hex1_dir.is_dir() and len(hex1_dir.name) == 2:
            for hex2_dir in hex1_dir.iterdir():
                if hex2_dir.is_dir() and len(hex2_dir.name) == 2:
                    for pdf_file in hex2_dir.glob("*.pdf"):
                        pdf_files.append(pdf_file)
    
    logger.info(f"Found {len(pdf_files)} PDF files in directory")
    return pdf_files


def process_pdf_with_document_ocr(pdf_path: Path, output_dir: Path) -> Optional[Dict]:
    """Process PDF using Google Vision OCR in document mode after rasterization."""
    try:
        pdf_stem = pdf_path.stem
        logger.info(f"Processing {pdf_stem} from {pdf_path}")
        
        # Initialize OCR extractor and formatter
        extractor = GoogleVisionOCRExtractor()
        logger.info(f"Using Google Cloud Vision API for OCR processing of {pdf_stem}")
        formatter = create_ocr_formatter()
        
        # Create output directories for this PDF
        pdf_output_dir = output_dir / "temp" / pdf_stem
        pdf_output_dir.mkdir(parents=True, exist_ok=True)
        
        raw_files = []
        data_files = []
        
        # Create temp directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Rasterize PDF to images at 200 DPI for OCR processing
            ocr_images_dir = temp_path / "ocr_images"
            num_pages = _rasterize_pdf(pdf_path, ocr_images_dir, pdf_stem, dpi=200)
            logger.info(f"Rasterized {pdf_stem} to {num_pages} images at 200 DPI")
            
            # Open original PDF for single-page extraction
            doc = fitz.open(str(pdf_path))
            
            # Process each page
            for page_num in range(1, num_pages + 1):
                try:
                    # Find the rasterized image for OCR (200 DPI)
                    ocr_image_path = ocr_images_dir / f"{pdf_stem}_p{page_num:02d}.png"
                    if not ocr_image_path.exists():
                        logger.warning(f"Skipping page {page_num} of {pdf_stem} - image not found")
                        continue
                    
                    # Process with OCR (Google Vision)
                    logger.debug(f"Running OCR on {ocr_image_path}")
                    ocr_result = extractor.extract_ocr(
                        image_path=ocr_image_path,
                        mode='document'
                    )
                    
                    if not ocr_result.get('success', False):
                        logger.warning(f"OCR failed for page {page_num} of {pdf_stem}: {ocr_result.get('error', 'Unknown error')}")
                        continue
                    
                    # Create output files for this page
                    page_pdf_path = pdf_output_dir / f"{pdf_stem}_{page_num}.pdf"
                    raw_json_path = pdf_output_dir / f"{pdf_stem}_{page_num}_raw.json"
                    page_png_path = pdf_output_dir / f"{pdf_stem}_{page_num}.png"
                    data_json_path = pdf_output_dir / f"{pdf_stem}_{page_num}_data.json"
                    
                    # Save single page PDF for raw output
                    if page_num - 1 < len(doc):  # page_num is 1-based
                        single_page_doc = fitz.open()
                        single_page_doc.insert_pdf(doc, from_page=page_num - 1, to_page=page_num - 1)
                        single_page_doc.save(str(page_pdf_path))
                        single_page_doc.close()
                    
                    # Create 72 DPI PNG for data directory storage (smaller file size)
                    scale_factor = _resize_image_to_72dpi(ocr_image_path, page_png_path)
                    
                    # Update image metadata for data directory JSON
                    img = Image.open(page_png_path)
                    width, height = img.size
                    dpi = img.info.get('dpi', (72, 72))  # 72 DPI after resizing
                    
                    # Get original high-resolution image dimensions for raw data
                    orig_img = Image.open(ocr_image_path)
                    orig_width, orig_height = orig_img.size
                    orig_dpi = orig_img.info.get('dpi', (200, 200))
                    
                    # Create raw OCR data with original 200 DPI coordinates and metadata
                    raw_ocr_data = copy.deepcopy(ocr_result)
                    raw_ocr_data['image'] = {
                        "width": orig_width,
                        "height": orig_height,
                        "format": "PNG",
                        "mode": orig_img.mode,
                        "dpi": orig_dpi,
                        "path": f"{pdf_stem}_{page_num}.pdf",
                        "page": page_num
                    }
                    
                    # Create data OCR data with scaled coordinates for 72 DPI images
                    data_ocr_data = copy.deepcopy(ocr_result)
                    data_ocr_data['image'] = {
                        "width": width,
                        "height": height,
                        "format": "PNG",
                        "mode": img.mode,
                        "dpi": dpi,
                        "path": f"{pdf_stem}_{page_num}.png",
                        "page": page_num
                    }
                    
                    # Format raw OCR data with metadata preservation for raw directory
                    formatted_raw_data = formatter.format_ocr_response(
                        raw_ocr_data, 
                        image_path=f"{pdf_stem}_{page_num}.pdf",
                        preserve_metadata=True
                    )
                    
                    # Save formatted JSON with metadata to raw directory
                    with open(raw_json_path, 'w') as f:
                        json.dump(formatted_raw_data, f, indent=2)
                    
                    # Format data OCR with original coordinates first, then scale (no metadata preservation)
                    formatted_data_data = formatter.format_ocr_response(
                        data_ocr_data, 
                        image_path=f"{pdf_stem}_{page_num}.png"
                    )
                    # Now apply scaling to the formatted data
                    formatted_data_data = _scale_bounding_boxes(formatted_data_data, scale_factor)
                    
                    # Save clean formatted JSON to data directory
                    with open(data_json_path, 'w') as f:
                        json.dump(formatted_data_data, f, indent=2)
                    
                    # Track files for tar creation
                    raw_files.append((page_pdf_path, raw_json_path, pdf_stem, page_num))
                    data_files.append((page_png_path, data_json_path, pdf_stem, page_num))
                    
                except Exception as e:
                    logger.error(f"Failed to process page {page_num} of {pdf_stem}: {e}")
                    continue
            
            doc.close()
        
        logger.info(f"Successfully processed {pdf_stem} - {len(raw_files)} pages")
        return {
            "pdf_stem": pdf_stem, 
            "pages_processed": len(raw_files),
            "raw_files": raw_files,
            "data_files": data_files
        }
        
    except Exception as e:
        logger.error(f"Failed to process PDF {pdf_path}: {e}")
        logger.error(traceback.format_exc())
        return None


# Global worker-specific shard managers (process-local)
_worker_shard_managers = {}

def get_worker_shard_manager(output_dir: Path, worker_id: int) -> TarShardManager:
    """Get or create a shard manager for the current worker process."""
    global _worker_shard_managers
    
    if worker_id not in _worker_shard_managers:
        _worker_shard_managers[worker_id] = TarShardManager(output_dir, worker_id)
    
    return _worker_shard_managers[worker_id]

def process_single_pdf_with_shard_manager(args_tuple) -> Tuple[bool, str, Dict]:
    """Process a single PDF and save to worker-specific shard manager."""
    pdf_path, output_dir, worker_id = args_tuple
    pdf_stem = pdf_path.stem
    
    try:
        start_time = time.time()
        
        # Get worker-specific shard manager
        shard_manager = get_worker_shard_manager(output_dir, worker_id)
        
        # Process PDF with OCR
        result = process_pdf_with_document_ocr(pdf_path, output_dir)
        if not result:
            return False, f"Failed to process PDF: {pdf_stem}", {}
        
        raw_files = result.get('raw_files', [])
        data_files = result.get('data_files', [])
        
        # Save files to shard manager
        for pdf_file, json_file, pdf_stem, page_num in raw_files:
            shard_manager.save_raw_files(pdf_file, json_file, pdf_stem, page_num)
        
        for png_file, json_file, pdf_stem, page_num in data_files:
            shard_manager.save_data_files(png_file, json_file, pdf_stem, page_num)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        success_msg = (f"Successfully processed {pdf_stem}: "
                      f"{result['pages_processed']} pages "
                      f"in {processing_time:.1f}s")
        
        logger.info(success_msg)
        return True, success_msg, shard_manager.get_manifest_data()
        
    except Exception as e:
        error_msg = f"Failed to process {pdf_stem}: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return False, error_msg, {}

def cleanup_worker_shard_manager(worker_id: int):
    """Clean up shard manager for a worker."""
    global _worker_shard_managers
    
    if worker_id in _worker_shard_managers:
        _worker_shard_managers[worker_id].close()
        del _worker_shard_managers[worker_id]


def main():
    parser = argparse.ArgumentParser(description="Process PubMed PDFs from local directory with OCR")
    parser.add_argument("--input-dir", type=Path, required=True,
                       help="Input directory containing PDFs in hex/hex structure")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for processed files")
    parser.add_argument("--max-files", type=int, default=None, 
                       help="Maximum number of files to process (for testing)")
    parser.add_argument("--start-index", type=int, default=0,
                       help="Start index for file processing (for resuming)")
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of worker processes (default: CPU count)")
    parser.add_argument("--failures-list", type=Path, default=None,
                       help="Path to file containing list of PDF stems to retry (one per line)")
    parser.add_argument("--log-dir", type=Path, default=None,
                       help="Directory to save log files (default: /tmp/pubmed_ocr_logs)")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip PDFs that have already been successfully processed with OCR content")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
        
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update log directory if specified
    global LOG_DIR
    if args.log_dir:
        LOG_DIR = args.log_dir
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    else:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Will create per-worker shard managers in the processing function
    logger.info(f"Using TarShardManager for compressed output shards")
    
    # Set number of workers
    num_workers = args.num_workers or min(mp.cpu_count(), 10)  # Cap at 10 for OCR processing
    logger.info(f"Using {num_workers} worker processes")
    
    # Find all PDF files in input directory
    all_pdf_files = list_all_pdfs_in_directory(args.input_dir)
    
    # Filter by failures list if provided
    if args.failures_list:
        logger.info(f"Loading failures list from {args.failures_list}")
        with open(args.failures_list, 'r') as f:
            failed_stems = {line.strip() for line in f if line.strip()}
        
        # Filter PDF files to only include those in the failures list
        pdf_files = [pdf for pdf in all_pdf_files if pdf.stem in failed_stems]
        logger.info(f"Filtered to {len(pdf_files)} files from failures list (out of {len(all_pdf_files)} total)")
    else:
        pdf_files = all_pdf_files
    
    # Skip existing files if requested (but not when processing failures list)
    if args.skip_existing and not args.failures_list:
        logger.info("Checking for already processed files...")
        original_count = len(pdf_files)
        pdf_files = [pdf for pdf in pdf_files if not check_pdf_already_processed(pdf.stem, args.output_dir)]
        skipped_count = original_count - len(pdf_files)
        logger.info(f"Skipped {skipped_count} already processed files, {len(pdf_files)} remaining to process")
    elif args.failures_list:
        logger.info("Processing failures list - skip-existing check disabled to allow reprocessing of rate-limited files")
    
    # Apply start index and max files
    if args.start_index > 0:
        pdf_files = pdf_files[args.start_index:]
        logger.info(f"Starting from index {args.start_index}")
    
    if args.max_files:
        pdf_files = pdf_files[:args.max_files]
        logger.info(f"Processing maximum {args.max_files} files")
    
    logger.info(f"Processing {len(pdf_files)} PDF files from {args.input_dir}")
    
    # Check Google Cloud authentication
    try:
        vision.ImageAnnotatorClient()
        logger.info("Google Cloud Vision authentication verified")
    except Exception as e:
        logger.error(f"Google Cloud authentication failed: {e}")
        logger.error("Make sure you have valid Google Cloud credentials set up")
        sys.exit(1)
    
    # Process files in parallel
    successful = 0
    failed = 0
    all_manifests = []
    
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Prepare arguments with worker IDs
            job_args = []
            for i, pdf_path in enumerate(pdf_files):
                worker_id = i % num_workers  # Distribute work across workers
                job_args.append((pdf_path, args.output_dir, worker_id))
            
            # Submit all jobs
            future_to_pdf = {executor.submit(process_single_pdf_with_shard_manager, args): args[0] 
                            for args in job_args}
            
            # Process completed jobs
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    success, message, manifest_data = future.result()
                    if success:
                        successful += 1
                        if manifest_data:
                            all_manifests.append(manifest_data)
                    else:
                        failed += 1
                        logger.warning(message)
                    
                    # Log progress every 10 files
                    if (successful + failed) % 10 == 0:
                        logger.info(f"Progress: {successful + failed}/{len(pdf_files)} "
                                  f"({successful} successful, {failed} failed)")
                        
                except Exception as e:
                    failed += 1
                    logger.error(f"Unexpected error processing {pdf_path}: {e}")
    finally:
        # Clean up all worker shard managers
        for worker_id in range(num_workers):
            cleanup_worker_shard_manager(worker_id)
        
        # Merge and save shard manifest
        merged_manifest = {'raw': {}, 'data': {}}
        for manifest in all_manifests:
            for shard_type in ['raw', 'data']:
                if shard_type in manifest:
                    merged_manifest[shard_type].update(manifest[shard_type])
        
        # Save shard manifest
        manifest_path = args.output_dir / "shard_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(merged_manifest, f, indent=2)
        
        logger.info(f"Saved shard manifest with {len(merged_manifest['raw'])} raw shards and {len(merged_manifest['data'])} data shards")
        logger.info("Finished processing all files")
    
    # Final summary
    logger.info(f"Processing complete: {successful} successful, {failed} failed out of {len(pdf_files)} total")
    
    if failed > 0:
        logger.warning(f"{failed} files failed processing. Check logs for details.")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
