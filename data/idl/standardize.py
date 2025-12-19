#!/usr/bin/env python3
"""
IDL ➜ Standard Shard Converter

Converts a single IDL sub-directory (e.g., /mnt/mldata/idl/samples/idl-train-00001)
into a tar shard at /mnt/mldata/datasets/idl/train-00001.tar, containing:
  - {document_id}.pdf
  - {document_id}_{page_id}.png (72 DPI rasterized page images)
  - {document_id}_{page_id}.json (standard per-page JSON)

Original per-document JSON (example):
{
  "pages": [
    {
      "bbox": [[x, y, w, h], ...],
      "poly": [...],
      "score": [...],
      "text": ["PHILIP MORRIS MANAGEMENT CORP.", ...]
    },
    ...
  ]
}

Standard per-page JSON:
{
  "text": {
    "lines": [{"text": str, "box": [x1, y1, x3, y3]}],
    "text": str,      # 1D text representation (reading order)
    "text2d": str   # 2D text representation (spatial layout preserved)
  },
  "image": {
    "path": "{document_id}_{page_id}.<ext>",
    "width": int,
    "height": int,
    "dpi": int | null
  },
  "metadata": {  # Optional, if --text-processing enabled
    "qwen_tokens": {
      "text": int,    # Token count for 1D text
      "text2d": int,  # Token count for 2D text
      "lines": int,   # Token count for JSON lines
      "image": int    # Visual token count (width//28 * height//28)
    }
  }
}

Assumptions & Notes:
- BBoxes in the source JSON are [x, y, w, h] by default. Switch with --bbox-format=x1y1x2y2 if needed.
- Convert to top-left / bottom-right [x1, y1, x3, y3] and round to 3 decimals.
- Optionally scale boxes into pixel space of the rasterized image with --bbox-space=pdf_to_pixel (uses dpi/72).
- Page numbering inside filenames is zero-based by default (e.g., 0000). Switch with --page-base=1 for one-based.
- Images default to PNG at 72 DPI; change with --image-ext and --dpi.
- Dry-run mode validates structure and prints a summary without writing the tar or rasterizing images.

Dependencies:
  pip install pymupdf pillow tqdm
  
  # For text processing and token counting:
  pip install transformers

Usage:
  # Validate only (no outputs)
  python standardize.py \
      --input-dir /mnt/mldata/idl/samples/idl-train-00001 \
      --output /mnt/mldata/datasets/idl/train-00001.tar \
      --dry-run

  # Produce the shard with text processing and token counting
  python standardize.py \
      --input-dir /mnt/mldata/idl/samples/idl-train-00001 \
      --output /mnt/mldata/datasets/idl/train-00001.tar

  # Produce the shard without text processing (faster)
  python standardize.py \
      --input-dir /mnt/mldata/idl/samples/idl-train-00001 \
      --output /mnt/mldata/datasets/idl/train-00001.tar \
      --no-text-processing
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional, Any

import fitz  # PyMuPDF
from tqdm import tqdm

try:
    from transformers import AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class Config:
    input_dir: Path
    output_path: Path
    dpi: int = 72
    image_ext: str = "png"  # png|jpg|tif
    page_base: int = 0       # 0 or 1
    bbox_format: str = "xywh"  # xywh | x1y1x2y2
    bbox_space: str = "as_is"  # as_is | pdf_to_pixel
    dry_run: bool = False
    add_text_processing: bool = True
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    local_files_only: bool = True


# -------------- helpers --------------

def round3(x: float) -> float:
    return round(float(x), 3)


def ensure_dir_exists(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _page_id_str(idx: int, total_pages: int, base: int) -> str:
    # Zero-pad to at least 4 digits; expand if document is extremely long
    max_index = base + max(0, total_pages - 1)
    pad = max(4, len(str(max_index)))
    return f"{idx + base:0{pad}d}"


def _bbox_to_x1y1x3y3(box: List[float], fmt: str, scale: float = 1.0) -> List[float]:
    if fmt == "xywh":
        x, y, w, h = box
        x1, y1, x2, y2 = x, y, x + w, y + h
    elif fmt == "x1y1x2y2":
        x1, y1, x2, y2 = box
    else:
        raise ValueError(f"Unsupported bbox format: {fmt}")

    # scale, then round
    if scale != 1.0:
        x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
    return [round3(x1), round3(y1), round3(x2), round3(y2)]


def load_doc_json(json_path: Path) -> Dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -------------- text processing --------------

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


# -------------- conversion --------------

def validate_structure(doc_dir: Path, bbox_fmt: str) -> Tuple[bool, str]:
    """Light-weight checks used in dry-run mode."""
    doc_id = doc_dir.name
    pdf_path = doc_dir / f"{doc_id}.pdf"
    json_path = doc_dir / f"{doc_id}.json"

    if not pdf_path.exists():
        return False, f"Missing PDF: {pdf_path}"
    if not json_path.exists():
        return False, f"Missing JSON: {json_path}"

    try:
        meta = load_doc_json(json_path)
        pages = meta.get("pages", [])
        if not isinstance(pages, list):
            return False, f"JSON 'pages' is not a list in {json_path}"
        # Try to parse first page's bbox format.
        if pages:
            p0 = pages[0]
            if "bbox" in p0 and p0["bbox"]:
                sample = p0["bbox"][0]
                if bbox_fmt == "xywh" and len(sample) != 4:
                    return False, "bbox format mismatch: expected [x,y,w,h]"
                if bbox_fmt == "x1y1x2y2" and len(sample) != 4:
                    return False, "bbox format mismatch: expected [x1,y1,x2,y2]"
    except Exception as e:
        return False, f"Failed reading JSON {json_path}: {e}"

    try:
        with fitz.open(pdf_path) as doc:
            _ = len(doc)
    except Exception as e:
        return False, f"Failed opening PDF {pdf_path}: {e}"

    return True, "ok"


def convert_one_document(
    doc_dir: Path,
    tar: tarfile.TarFile,
    cfg: Config,
    page_pbar: Optional[tqdm] = None,
    processor=None,
) -> Dict[str, int]:
    """Convert a single {document_id}/ folder and append contents into `tar`.

    Returns brief stats.
    """
    stats = {"pages": 0, "lines": 0}
    doc_id = doc_dir.name
    pdf_path = doc_dir / f"{doc_id}.pdf"
    json_path = doc_dir / f"{doc_id}.json"

    if not pdf_path.exists() or not json_path.exists():
        logging.warning("Skipping %s (missing pdf/json)", doc_id)
        return stats

    meta = load_doc_json(json_path)
    page_ann = meta.get("pages", []) or []

    with fitz.open(pdf_path) as pdf:
        total_pages = len(pdf)

        zoom = cfg.dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for i in range(total_pages):
            page = pdf[i]
            pid = _page_id_str(i, total_pages, cfg.page_base)

            # Render page to image bytes
            pix = page.get_pixmap(matrix=mat, alpha=False)
            coord_scale = (cfg.dpi / 72.0) if cfg.bbox_space == "pdf_to_pixel" else 1.0
            img_bytes = pix.tobytes(cfg.image_ext)
            img_name = f"{doc_id}_{pid}.{cfg.image_ext}"

            # Build per-page JSON
            width, height = pix.width, pix.height
            lines: List[Dict] = []

            if i < len(page_ann):
                p = page_ann[i] or {}
                texts = p.get("text", []) or []
                bboxes = p.get("bbox", []) or []
                n = min(len(texts), len(bboxes))
                if len(texts) != len(bboxes):
                    logging.debug(
                        "Doc %s page %d: mismatched text/bbox lengths (%d vs %d), truncating to %d",
                        doc_id, i, len(texts), len(bboxes), n,
                    )
                for j in range(n):
                    box = _bbox_to_x1y1x3y3(bboxes[j], cfg.bbox_format, scale=coord_scale)
                    lines.append({"text": str(texts[j]), "box": [int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height)]})

            # Generate text representations
            text_1d_content = ""
            text_2d_content = ""
            
            if lines and cfg.add_text_processing:
                try:
                    text_1d_content = text_1d(lines)
                except Exception as e:
                    logging.warning(f"Failed to generate text_1d for {doc_id} page {i}: {e}")
                    text_1d_content = ""
                
                try:
                    text_2d_content = text_2d(lines)
                except Exception as e:
                    logging.warning(f"Failed to generate text_2d for {doc_id} page {i}: {e}")
                    text_2d_content = ""

            # Build text data with enhanced content
            text_data = {
                "lines": lines,
                "text": text_1d_content,
                "text2d": text_2d_content
            }

            # Add token counts if processor is available
            metadata = {}
            if processor is not None and TRANSFORMERS_AVAILABLE and cfg.add_text_processing:
                try:
                    qwen_tokens = {
                        "text": len(processor.tokenizer(text_1d_content, return_tensors="pt").input_ids[0]),
                        "text2d": len(processor.tokenizer(text_2d_content, return_tensors="pt").input_ids[0]),
                        "image": (width // 28) * (height // 28),
                    }
                    
                    if lines:
                        qwen_tokens["lines"] = len(processor.tokenizer(json.dumps(lines), return_tensors="pt").input_ids[0])
                    
                    metadata["qwen_tokens"] = qwen_tokens
                except Exception as e:
                    logging.warning(f"Failed to compute token counts for {doc_id} page {i}: {e}")

            page_json = {
                "text": text_data,
                "image": {
                    "path": img_name,
                    "width": int(width),
                    "height": int(height),
                    "dpi": int(cfg.dpi),
                },
            }
            
            if metadata:
                page_json["metadata"] = metadata
            page_json_bytes = json.dumps(page_json, ensure_ascii=False).encode("utf-8")
            page_json_name = f"{doc_id}_{pid}.json"

            # Add image to tar
            info_img = tarfile.TarInfo(name=img_name)
            info_img.size = len(img_bytes)
            info_img.mtime = int(time.time())
            tar.addfile(info_img, io.BytesIO(img_bytes))

            # Add JSON to tar
            info_js = tarfile.TarInfo(name=page_json_name)
            info_js.size = len(page_json_bytes)
            info_js.mtime = int(time.time())
            tar.addfile(info_js, io.BytesIO(page_json_bytes))

            stats["pages"] += 1
            stats["lines"] += len(lines)

            # Update intra-document progress
            if page_pbar:
                page_pbar.update(1)

    return stats


def process_shard(cfg: Config) -> None:
    # Initialize processor for token counting if requested
    processor = None
    if cfg.add_text_processing and TRANSFORMERS_AVAILABLE:
        try:
            processor = AutoProcessor.from_pretrained(
                cfg.model_id, use_fast=True, local_files_only=cfg.local_files_only
            )
            logging.info(f"Loaded processor: {cfg.model_id}")
        except Exception as e:
            logging.warning(f"Failed to load processor: {e}")

    # Validate & gather document folders
    doc_dirs = sorted([p for p in cfg.input_dir.iterdir() if p.is_dir()])
    if not doc_dirs:
        logging.warning("No document folders found under %s", cfg.input_dir)

    if cfg.dry_run:
        ok = 0
        with tqdm(doc_dirs, desc="Validating documents", unit="doc") as pbar:
            for d in pbar:
                valid, msg = validate_structure(d, cfg.bbox_format)
                if valid:
                    ok += 1
                    pbar.set_postfix(valid=ok, total=len(doc_dirs))
                else:
                    logging.warning("✖ %s: %s", d.name, msg)
        logging.info("Dry-run summary: %d/%d documents valid", ok, len(doc_dirs))
        return

    ensure_dir_exists(cfg.output_path)
    
    # Count total pages for overall progress tracking
    total_pages_to_process = 0
    valid_docs = []
    
    logging.info("Counting pages in documents...")
    for d in tqdm(doc_dirs, desc="Counting pages", unit="doc"):
        valid, msg = validate_structure(d, cfg.bbox_format)
        if not valid:
            logging.warning("Skipping %s due to validation error: %s", d.name, msg)
            continue
        
        # Quick page count
        doc_id = d.name
        pdf_path = d / f"{doc_id}.pdf"
        try:
            with fitz.open(pdf_path) as doc:
                page_count = len(doc)
                total_pages_to_process += page_count
                valid_docs.append((d, page_count))
        except Exception as e:
            logging.warning("Skipping %s due to PDF error: %s", d.name, e)
    
    logging.info(f"Processing {len(valid_docs)} documents with {total_pages_to_process} total pages")

    with tarfile.open(cfg.output_path, mode="w") as tar:
        total_pages = 0
        total_lines = 0
        docs_done = 0

        # Main document progress bar
        doc_pbar = tqdm(valid_docs, desc="Processing documents", unit="doc")
        
        # Page progress bar (shared across all documents)
        page_pbar = tqdm(total=total_pages_to_process, desc="Processing pages", unit="page")

        for doc_dir, page_count in doc_pbar:
            doc_pbar.set_description(f"Processing {doc_dir.name}")
            
            stats = convert_one_document(doc_dir, tar, cfg, page_pbar, processor)
            docs_done += 1
            total_pages += stats["pages"]
            total_lines += stats["lines"]
            
            doc_pbar.set_postfix(
                docs=docs_done,
                pages=total_pages,
                lines=total_lines
            )

        page_pbar.close()
        doc_pbar.close()

    logging.info(
        "Done. Output: %s | documents: %d | pages: %d | lines: %d",
        cfg.output_path, docs_done, total_pages, total_lines
    )


# -------------- CLI --------------

def parse_args(argv: Optional[List[str]] = None) -> Config:
    ap = argparse.ArgumentParser(
        description="Convert an IDL sub-directory into a standard shard tar file.")
    ap.add_argument("--input-dir", required=True, type=Path,
                    help="Path to idl-train-{ix:05d} directory containing {document_id}/ subfolders")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output tar path, e.g., /mnt/mldata/datasets/idl/train-00001.tar")
    ap.add_argument("--dpi", type=int, default=72, help="Rasterization DPI (default: 72)")
    ap.add_argument("--image-ext", choices=["png", "jpg", "tif"], default="png",
                    help="Image extension/format (default: png)")
    ap.add_argument("--page-base", type=int, choices=[0, 1], default=0,
                    help="Filename page index base (0 or 1; default: 0)")
    ap.add_argument("--bbox-format", choices=["xywh", "x1y1x2y2"], default="xywh",
                    help="Source bbox format in JSON (default: xywh)")
    ap.add_argument("--bbox-space", choices=["as_is", "pdf_to_pixel"], default="as_is",
                    help="If 'pdf_to_pixel', scale boxes by dpi/72 to match rasterized image pixels before rounding")
    ap.add_argument("--dry-run", action="store_true", help="Validate only; don't write output tar")
    ap.add_argument("--no-text-processing", action="store_true", 
                    help="Skip text processing and token counting (faster)")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct",
                    help="Model ID for tokenization (default: Qwen/Qwen2.5-VL-7B-Instruct)")
    ap.add_argument("--no-local-files-only", action="store_true",
                    help="Allow downloading model from HuggingFace (default: use local cache only)")
    ap.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    return Config(
        input_dir=args.input_dir,
        output_path=args.output,
        dpi=args.dpi,
        image_ext=args.image_ext,
        page_base=args.page_base,
        bbox_format=args.bbox_format,
        bbox_space=args.bbox_space,
        dry_run=args.dry_run,
        add_text_processing=not args.no_text_processing,
        model_id=args.model_id,
        local_files_only=not args.no_local_files_only,
    )


def main(argv: Optional[List[str]] = None) -> None:
    cfg = parse_args(argv)
    process_shard(cfg)


if __name__ == "__main__":
    main()
