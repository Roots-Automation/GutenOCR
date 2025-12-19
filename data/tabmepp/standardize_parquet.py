"""
Convert parquet dataset to standard, shard-by-pages format with safe parallel processing.

Input: Parquet files with columns [doc_id, pg_id, ocr, img]
- doc_id: str - Document identifier
- pg_id: int - Page number within document  
- ocr: str - JSON string with OCR data (words_data, lines_data)
- img: bytes - Raw image bytes

Output layout (pages_per_shard pages per shard, one input parquet = multiple output shards):
/output/path/
    train-00000-00000.tar  # First shard from first parquet
    train-00000-00001.tar  # Second shard from first parquet
    train-00001-00000.tar  # First shard from second parquet
    ...

Each tar contains:
    {document_id}_{page_id}.jpg
    {document_id}_{page_id}.json

Standard JSON per page:
{
  "text": {
    "words": [{"text": str, "box": [x1, y1, x2, y2]}],
    "lines": [{"text": str, "box": [x1, y1, x2, y2]}]
  },
  "image": {
    "path": "{document_id}_{page_id}.jpg",
    "width": int,
    "height": int,
    "dpi": int | null
  }
}

Key improvements:
- Safe parallel processing: one input parquet = isolated processing
- Shard by pages for predictable, even distribution
- No cross-parquet dependencies or coordination needed
- Each parquet processed independently in parallel
- Memory-efficient streaming within each parquet
- Proper error isolation per input file
"""

import argparse
import io
import json
import logging
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    from transformers import AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data structures for simple parallel processing
# -----------------------------------------------------------------------------
class ParquetTask(NamedTuple):
    """Task for processing a single parquet file."""
    parquet_file: str
    parquet_index: int
    output_dir: Path
    pages_per_shard: int
    add_text_processing: bool
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    local_files_only: bool = True


# -----------------------------------------------------------------------------
# Text processing functions
# -----------------------------------------------------------------------------
def text_1d(detections: List[Dict[str, Any]]) -> str:
    """
    Extract and concatenate text from a list of detection dictionaries.
    Each dictionary is expected to have a 'text' key.
    """
    if not detections:
        return ""
    
    if 'bbox' not in detections[0] and 'box' not in detections[0]:
        raise ValueError("Detections must contain 'bbox' or 'box' key for sorting")

    detections_sorted = sorted(detections, key=lambda x: (x.get('bbox', x.get('box', []))[1], x.get('bbox', x.get('box', []))[0]))
    return ' '.join(d['text'].strip() for d in detections_sorted if 'text' in d)


def text_2d(detections: List[Dict[str, Any]], p: float = 0.8, max_newlines: int = 2) -> str:
    """
    Extract and concatenate text from a list of detection dictionaries in a 2D layout.
    Each dictionary is expected to have a 'text' key and either 'bbox' or 'box' key with coordinates.
    
    This function uses robust line clustering and character density estimation to preserve 
    the 2D spatial layout of text by:
    1. Computing line height and character density using percentile-based estimation
    2. Clustering text blocks into horizontal lines using line height tolerance
    3. Sorting blocks within each line by horizontal position
    4. Adding appropriate spacing between blocks based on absolute position on page
    5. Adding newlines between lines based on vertical gaps
    
    Args:
        detections: List of detection dictionaries, each containing:
            - 'text': The text content
            - 'bbox' or 'box': Bounding box coordinates [x1, y1, x2, y2] or similar format
        p: Percentile for character density estimation (default 0.8)
        max_newlines: Maximum number of newlines to add between blocks (default 2)
    
    Returns:
        String with text arranged in 2D layout preserving spatial relationships
    """
    if not detections:
        return ""
    
    # Normalize bounding box format and extract coordinates
    normalized_blocks = []
    for detection in detections:
        text = detection['text'].strip()
        if not text:  # Skip empty text blocks
            continue
            
        bbox = detection.get('bbox', detection.get('box', []))
        
        if len(bbox) >= 4:
            # Assume format is [x1, y1, x2, y2] or similar
            x1, y1 = float(bbox[0]), float(bbox[1])
            x2, y2 = float(bbox[2]), float(bbox[3])
            
            # Ensure x1 <= x2 and y1 <= y2
            left, right = min(x1, x2), max(x1, x2)
            top, bottom = min(y1, y2), max(y1, y2)
            
            width = right - left
            height = bottom - top
            
            # Skip degenerate boxes
            if width <= 0 or height <= 0:
                continue
            
            normalized_blocks.append({
                'text': text,
                'left': left,
                'right': right,
                'top': top,
                'bottom': bottom,
                'center_y': (top + bottom) / 2,
                'center_x': (left + right) / 2,
                'width': width,
                'height': height,
                'char_ratio': len(text) / width  # characters per unit width
            })
    
    if not normalized_blocks:
        return ""
    
    # Compute line height and character density using robust estimation
    line_height, char_density = _compute_line_height_char_density(normalized_blocks, p)
    
    # Find page bounds for absolute positioning
    page_left = min(block['left'] for block in normalized_blocks)
    page_right = max(block['right'] for block in normalized_blocks)
    page_width = page_right - page_left
    
    # Cluster blocks into lines using line height tolerance
    lines = _cluster_blocks_robust(normalized_blocks, line_height)
    
    # Sort lines by vertical position (top of the line)
    lines = sorted(lines.items(), key=lambda x: x[0])
    
    # Build the output string with absolute positioning
    result_lines = []
    
    for _, line_blocks in lines:
        # Sort blocks in line by horizontal position
        line_blocks.sort(key=lambda b: b['left'])
        
        # Build the line with proper absolute spacing
        line_text = ""
        line_length = 0  # Track current position in characters
        
        for i, block in enumerate(line_blocks):
            # Calculate absolute position on the page as character position
            block_col_pos = (block['left'] - page_left) * char_density
            target_pos = int(round(block_col_pos))
            
            # Add spaces to reach the target position
            spaces_needed = max(0, target_pos - line_length)
            line_text += " " * spaces_needed
            line_length += spaces_needed
            
            # Add the text
            line_text += block['text']
            line_length += len(block['text'])
        
        result_lines.append(line_text.rstrip())
    
    # Join lines with newlines, adding extra newlines for large vertical gaps
    if len(result_lines) <= 1:
        return '\n'.join(result_lines)
    
    final_text = result_lines[0]
    
    for i in range(1, len(result_lines)):
        # Calculate vertical gap between lines
        prev_line_blocks = dict(lines)[list(dict(lines).keys())[i-1]]
        curr_line_blocks = dict(lines)[list(dict(lines).keys())[i]]
        
        prev_line_bottom = max(b['bottom'] for b in prev_line_blocks)
        curr_line_top = min(b['top'] for b in curr_line_blocks)
        gap = curr_line_top - prev_line_bottom
        
        # Add extra newlines for large gaps (more than 1.5x line height)
        if gap > 1.5 * line_height:
            num_newlines = min(max_newlines, max(1, int(round(gap / line_height))))
        else:
            num_newlines = 1
            
        final_text += '\n' * num_newlines + result_lines[i]
    
    return final_text


# -----------------------------------------------------------------------------
# Simple parallel processing functions
# -----------------------------------------------------------------------------
def process_single_parquet(task: ParquetTask) -> Tuple[int, int, int]:
    """Process a single parquet file into multiple tar shards."""
    parquet_file = task.parquet_file
    parquet_index = task.parquet_index
    output_dir = task.output_dir
    pages_per_shard = task.pages_per_shard
    add_text_processing = task.add_text_processing
    model_id = task.model_id
    local_files_only = task.local_files_only
    
    logger.info(f"Processing {parquet_file} (index {parquet_index})")
    
    # Initialize processor for token counting if requested
    processor = None
    if add_text_processing and TRANSFORMERS_AVAILABLE:
        try:
            processor = AutoProcessor.from_pretrained(
                model_id, use_fast=True, local_files_only=local_files_only
            )
        except Exception as e:
            logger.warning(f"Failed to load processor for {parquet_file}: {e}")
    
    try:
        # Load the parquet file
        df = pd.read_parquet(parquet_file)
        total_pages = len(df)
        
        logger.info(f"  {parquet_file}: {total_pages} pages")
        
        # Calculate number of shards needed
        num_shards = ceil(total_pages / pages_per_shard)
        
        total_documents_processed = 0
        total_pages_written = 0
        total_pages_skipped = 0
        
        # Process each shard
        for shard_idx in range(num_shards):
            start_idx = shard_idx * pages_per_shard
            end_idx = min(start_idx + pages_per_shard, total_pages)
            shard_df = df.iloc[start_idx:end_idx]
            
            # Create shard filename: parquet-index_shard-index
            shard_name = f"train-{parquet_index:05d}-{shard_idx:05d}.tar"
            shard_path = output_dir / shard_name
            
            logger.info(f"  Creating {shard_name} with {len(shard_df)} pages")
            
            # Process this shard
            docs_processed, pages_written, pages_skipped = process_shard_data(
                shard_df, shard_path, processor
            )
            
            total_documents_processed += docs_processed
            total_pages_written += pages_written
            total_pages_skipped += pages_skipped
        
        logger.info(f"Completed {parquet_file}: {total_documents_processed} docs, "
                   f"{total_pages_written} pages written, {total_pages_skipped} pages skipped")
        
        return total_documents_processed, total_pages_written, total_pages_skipped
        
    except Exception as e:
        logger.error(f"Failed to process {parquet_file}: {e}")
        return 0, 0, 0


def process_shard_data(df: pd.DataFrame, shard_path: Path, processor=None) -> Tuple[int, int, int]:
    """Process a dataframe slice into a single tar shard."""
    documents_processed = set()
    pages_written = 0
    pages_skipped = 0
    
    with tarfile.open(shard_path, mode="w") as tar:
        for _, row in df.iterrows():
            try:
                doc_id = row['doc_id']
                pg_id = row['pg_id']
                ocr_json = row['ocr']
                img_bytes = row['img']
                
                # Convert page to standard format
                standard_json, image_filename = convert_page_to_standard(
                    doc_id, pg_id, ocr_json, img_bytes, processor
                )
                
                # Add to tar
                json_filename = image_filename.replace(".jpg", ".json")
                add_json_to_tar(tar, json_filename, standard_json)
                add_image_to_tar(tar, image_filename, img_bytes)
                
                documents_processed.add(doc_id)
                pages_written += 1
                
            except Exception as e:
                logger.error(f"Error processing page {row.get('doc_id', '?')}_{row.get('pg_id', '?')}: {e}")
                pages_skipped += 1
    
    return len(documents_processed), pages_written, pages_skipped


def _compute_line_height_char_density(blocks: List[Dict], p: float) -> Tuple[float, float]:
    """
    Compute line height and character density using robust percentile-based estimation.
    
    Args:
        blocks: List of normalized text blocks
        p: Percentile for character density estimation
        
    Returns:
        Tuple of (line_height, char_density)
    """
    # Line height is the minimum height of any text block
    line_height = min(block['height'] for block in blocks)
    
    # Character density is the p-th percentile of character-to-width ratios
    # Higher percentiles correspond to denser text (more chars per unit width)
    char_ratios = [block['char_ratio'] for block in blocks]
    char_density = _percentile(char_ratios, p * 100)
    
    return line_height, char_density


def _cluster_blocks_robust(blocks: List[Dict], line_height: float) -> Dict:
    """
    Cluster blocks into lines using line height tolerance, updating cluster centroids.
    
    Args:
        blocks: List of normalized text blocks
        line_height: Estimated line height for clustering tolerance
        
    Returns:
        Dictionary of lines keyed by average y-position
    """
    lines = {}
    
    for block in blocks:
        block_center_y = block['center_y']
        placed = False
        
        # Search for a nearby line to merge with
        for line_y in list(lines.keys()):  # Use list() to avoid dict mutation during iteration
            if abs(block_center_y - line_y) <= line_height:
                # Add block to existing line
                line_blocks = lines.pop(line_y)
                line_blocks.append(block)
                
                # Recompute line centroid
                new_line_y = sum(b['center_y'] for b in line_blocks) / len(line_blocks)
                lines[new_line_y] = line_blocks
                placed = True
                break
        
        # If no nearby line found, create new line
        if not placed:
            lines[block_center_y] = [block]
    
    return lines


def _percentile(data: List[float], percentile: float) -> float:
    """
    Calculate the percentile of a list of values.
    
    Args:
        data: List of numeric values
        percentile: Percentile to calculate (0-100)
        
    Returns:
        Percentile value
    """
    if not data:
        return 0.0
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    if percentile <= 0:
        return sorted_data[0]
    if percentile >= 100:
        return sorted_data[-1]
    
    # Use linear interpolation for percentiles
    index = (percentile / 100) * (n - 1)
    lower_index = int(index)
    upper_index = min(lower_index + 1, n - 1)
    
    if lower_index == upper_index:
        return sorted_data[lower_index]
    
    # Linear interpolation
    weight = index - lower_index
    return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight


# -----------------------------------------------------------------------------
# Coordinate processing
# -----------------------------------------------------------------------------
def extract_box_coordinates(item: Dict[str, Any], img_width: int, img_height: int) -> List[int]:
    """Convert relative coordinates to absolute pixel coordinates.
    
    Handles rotated boxes by taking min/max of all 4 corners.
    Returns [x1, y1, x2, y2] as integers.
    """
    # Get all 4 corner coordinates (relative 0-1)
    xs = [float(item["X1"]), float(item["X2"]), float(item["X3"]), float(item["X4"])]
    ys = [float(item["Y1"]), float(item["Y2"]), float(item["Y3"]), float(item["Y4"])]
    
    # Find bounding box
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Convert to absolute pixel coordinates
    x1 = int(min_x * img_width)
    y1 = int(min_y * img_height)
    x2 = int(max_x * img_width)
    y2 = int(max_y * img_height)
    
    return [x1, y1, x2, y2]


def get_image_info_from_bytes(img_bytes: bytes) -> Dict[str, Any]:
    """Extract width/height/dpi from image bytes."""
    try:
        with Image.open(io.BytesIO(img_bytes)) as img:
            width, height = img.size
            dpi = img.info.get("dpi")
            dpi_val: Optional[int] = int(dpi[0]) if isinstance(dpi, (tuple, list)) else (
                int(dpi) if isinstance(dpi, (int, float)) else None
            )
            return {
                "width": int(width),
                "height": int(height),
                "dpi": dpi_val,
            }
    except Exception as e:
        logger.error(f"Failed to read image info: {e}")
        return {"width": None, "height": None, "dpi": None}


def convert_page_to_standard(
    doc_id: str, pg_id: int, ocr_json: str, img_bytes: bytes, processor=None
) -> Tuple[Dict[str, Any], str]:
    """Convert one page to the standard schema.
    
    Returns (standard_json, image_filename).
    """
    # Parse OCR data
    ocr_data = json.loads(ocr_json)
    
    # Get image info
    img_info = get_image_info_from_bytes(img_bytes)
    img_width = img_info.get("width", 1)
    img_height = img_info.get("height", 1)
    
    # Process words
    words: List[Dict[str, Any]] = []
    for w in ocr_data.get("words_data", []) or []:
        box = extract_box_coordinates(w, img_width, img_height)
        words.append({"text": w.get("Word", ""), "box": box})

    # Process lines
    lines: List[Dict[str, Any]] = []
    for ln in ocr_data.get("lines_data", []) or []:
        box = extract_box_coordinates(ln, img_width, img_height)
        lines.append({"text": ln.get("Word", ""), "box": box})

    # Create image filename
    image_filename = f"{doc_id}_{pg_id:04d}.jpg"
    
    # Generate text representations
    # Use lines if available, otherwise fall back to words
    detections = lines if lines else words
    
    text_1d_content = ""
    text_2d_content = ""
    
    if detections:
        try:
            text_1d_content = text_1d(detections)
        except Exception as e:
            logger.warning(f"Failed to generate text_1d for {doc_id}_{pg_id}: {e}")
            text_1d_content = ""
        
        try:
            text_2d_content = text_2d(detections)
        except Exception as e:
            logger.warning(f"Failed to generate text_2d for {doc_id}_{pg_id}: {e}")
            text_2d_content = ""
    
    # Build standard format
    text_data = {
        "words": words,
        "lines": lines,
        "text": text_1d_content,
        "text2d": text_2d_content
    }
    
    # Add token counts if processor is available
    metadata = {}
    if processor is not None and TRANSFORMERS_AVAILABLE:
        try:
            qwen_tokens = {
                "text": len(processor.tokenizer(text_1d_content, return_tensors="pt").input_ids[0]),
                "text2d": len(processor.tokenizer(text_2d_content, return_tensors="pt").input_ids[0]),
                "image": (img_width // 28) * (img_height // 28),
            }
            
            if lines:
                qwen_tokens["lines"] = len(processor.tokenizer(json.dumps(lines), return_tensors="pt").input_ids[0])
            if words:
                qwen_tokens["words"] = len(processor.tokenizer(json.dumps(words), return_tensors="pt").input_ids[0])
            
            metadata["qwen_tokens"] = qwen_tokens
        except Exception as e:
            logger.warning(f"Failed to compute token counts for {doc_id}_{pg_id}: {e}")
    
    standard = {
        "text": text_data,
        "image": {"path": image_filename, **img_info}
    }
    
    if metadata:
        standard["metadata"] = metadata

    return standard, image_filename


# -----------------------------------------------------------------------------
# Shard writing
# -----------------------------------------------------------------------------
def add_json_to_tar(tar: tarfile.TarFile, arcname: str, data: Dict[str, Any]) -> None:
    """Add JSON data to tar archive."""
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    info = tarfile.TarInfo(name=arcname)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def add_image_to_tar(tar: tarfile.TarFile, arcname: str, img_bytes: bytes) -> None:
    """Add image bytes to tar archive."""
    info = tarfile.TarInfo(name=arcname)
    info.size = len(img_bytes)
    tar.addfile(info, io.BytesIO(img_bytes))


def open_shard(output_dir: Path, shard_index: int, split: str = "train") -> tarfile.TarFile:
    """Open a new shard tar file."""
    shard_name = f"{split}-{shard_index:05d}.tar"
    shard_path = output_dir / shard_name
    logger.info(f"Opening {shard_name}")
    return tarfile.open(shard_path, mode="w")





def parallel_standardize_parquet(
    input_patterns: List[str],
    output_path: Path,
    pages_per_shard: int = 2048,
    dry_run: bool = False,
    add_text_processing: bool = True,
    max_workers: Optional[int] = None,
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    local_files_only: bool = True,
) -> None:
    """Main processing function with safe parallel processing per parquet file."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all parquet files
    all_files = []
    for pattern in input_patterns:
        files = list(Path().glob(pattern))
        all_files.extend(files)
    
    if not all_files:
        logger.error("No parquet files found!")
        return
    
    logger.info(f"Found {len(all_files)} parquet files to process")
    
    if dry_run:
        logger.info("=" * 80)
        logger.info("DRY RUN ANALYSIS")
        logger.info("=" * 80)
        
        total_pages = 0
        total_size_mb = 0
        total_shards = 0
        
        for i, parquet_file in enumerate(all_files):
            try:
                # Quick analysis without loading full data
                df = pd.read_parquet(parquet_file, columns=['doc_id', 'pg_id'])
                file_pages = len(df)
                file_docs = df['doc_id'].nunique()
                file_shards = ceil(file_pages / pages_per_shard)
                
                # Rough size estimate
                file_size_mb = parquet_file.stat().st_size / 1024 / 1024
                
                total_pages += file_pages
                total_size_mb += file_size_mb
                total_shards += file_shards
                
                logger.info(f"  {parquet_file.name}: {file_docs:4d} docs, {file_pages:6d} pages, "
                           f"{file_shards:3d} shards, {file_size_mb:6.1f} MB")
                           
            except Exception as e:
                logger.error(f"Failed to analyze {parquet_file}: {e}")
        
        cpu_count = os.cpu_count() or 1
        max_workers_actual = max_workers or min(len(all_files), cpu_count)
        
        logger.info("")
        logger.info("Summary:")
        logger.info(f"  Total parquet files: {len(all_files)}")
        logger.info(f"  Total pages: {total_pages:,}")
        logger.info(f"  Total estimated shards: {total_shards}")
        logger.info(f"  Estimated total size: {total_size_mb:.1f} MB")
        logger.info(f"  Pages per shard: {pages_per_shard}")
        logger.info("")
        logger.info("Parallelization:")
        logger.info(f"  Available CPU cores: {cpu_count}")
        logger.info(f"  Parallel parquet workers: {max_workers_actual}")
        logger.info(f"  Each parquet processed independently")
        logger.info(f"  Output format: train-PPPPP-SSSSS.tar")
        logger.info(f"    PPPPP = parquet index (00000-{len(all_files)-1:05d})")
        logger.info(f"    SSSSS = shard index within that parquet")
        
        return
    
    # Create tasks for parallel processing
    tasks = []
    for i, parquet_file in enumerate(all_files):
        task = ParquetTask(
            parquet_file=str(parquet_file),
            parquet_index=i,
            output_dir=output_path,
            pages_per_shard=pages_per_shard,
            add_text_processing=add_text_processing,
            model_id=model_id,
            local_files_only=local_files_only,
        )
        tasks.append(task)
    
    # Process all parquet files in parallel
    if max_workers is None:
        max_workers = min(len(all_files), os.cpu_count() or 1)
    
    logger.info("=" * 80)
    logger.info(f"PROCESSING {len(all_files)} PARQUET FILES")
    logger.info(f"Using {max_workers} parallel workers")
    logger.info("=" * 80)
    
    total_docs = 0
    total_pages_written = 0
    total_pages_skipped = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(process_single_parquet, task): task for task in tasks}
        
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Processing parquet files"):
            task = future_to_task[future]
            try:
                docs_processed, pages_written, pages_skipped = future.result()
                total_docs += docs_processed
                total_pages_written += pages_written
                total_pages_skipped += pages_skipped
                
            except Exception as e:
                logger.error(f"Failed to process {task.parquet_file}: {e}")
    
    logger.info("=" * 80)
    logger.info("PROCESSING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"  Total documents processed: {total_docs}")
    logger.info(f"  Total pages written: {total_pages_written}")
    logger.info(f"  Total pages skipped: {total_pages_skipped}")
    logger.info(f"  Output files in: {output_path}")
    
    # List created shards
    shard_files = sorted(output_path.glob("train-*.tar"))
    logger.info(f"  Created {len(shard_files)} shard files")
    if shard_files:
        total_size_mb = sum(f.stat().st_size for f in shard_files) / 1024 / 1024
        logger.info(f"  Total output size: {total_size_mb:.1f} MB")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Standardize parquet dataset to tar shards with safe parallel processing")
    p.add_argument(
        "input_patterns", 
        nargs="+", 
        help="Glob patterns for input parquet files (e.g., 'data/*.parquet')"
    )
    p.add_argument("output_path", type=Path, help="Path to write output shards")
    p.add_argument(
        "--pages-per-shard",
        type=int,
        default=2048,
        help="Max pages per shard (default: 2048)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show analysis without writing files",
    )
    p.add_argument(
        "--no-text-processing",
        action="store_true",
        help="Skip text processing and token counting (faster)",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        help="Max parallel parquet workers (default: min(files, cores))",
    )
    p.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model ID for tokenization (default: Qwen/Qwen2.5-VL-7B-Instruct)",
    )
    p.add_argument(
        "--no-local-files-only",
        action="store_true",
        help="Allow downloading model from HuggingFace (default: use local cache only)",
    )
    args = p.parse_args()

    parallel_standardize_parquet(
        input_patterns=args.input_patterns,
        output_path=args.output_path,
        pages_per_shard=args.pages_per_shard,
        dry_run=args.dry_run,
        add_text_processing=not args.no_text_processing,
        max_workers=args.max_workers,
        model_id=args.model_id,
        local_files_only=not args.no_local_files_only,
    )


if __name__ == "__main__":
    main()
