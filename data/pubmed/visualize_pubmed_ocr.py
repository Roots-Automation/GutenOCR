#!/usr/bin/env python3
"""
PubMed OCR Visualization Script

This script takes PubMed OCR JSON outputs and their corresponding page images,
then creates visualization images with bounding boxes overlaid to show
the detected lines, words, and paragraphs.

Usage:
    python visualize_pubmed_ocr.py --input /tmp/pubmed_test --output /tmp/pubmed_viz --pdf-id IJEM-16-117.PMC3354931 --pages 1,2,3
    python visualize_pubmed_ocr.py --input /tmp/pubmed_test --output /tmp/pubmed_viz --pdf-id IJEM-16-117.PMC3354931 --all-pages
    python visualize_pubmed_ocr.py --input /tmp/pubmed_test --output /tmp/pubmed_viz --all-pdfs
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import re

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: PIL (Pillow) is required. Install with: pip install pillow")
    sys.exit(1)

def load_pubmed_ocr_data(json_path: Path) -> Dict:
    """Load PubMed OCR data from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract the OCR data from the structure
        text_data = data.get('text', {})
        return {
            'lines_data': text_data.get('lines', []),
            'words_data': text_data.get('words', []),  
            'paragraphs_data': text_data.get('paragraphs', [])
        }
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return {"lines_data": [], "words_data": [], "paragraphs_data": []}


def draw_bounding_boxes(image: Image.Image, ocr_items: List[Dict], color: str, 
                       box_type: str = "line", show_labels: bool = True) -> Image.Image:
    """
    Draw bounding boxes on image for PubMed OCR data.
    
    Args:
        image: PIL Image to draw on
        ocr_items: List of OCR items with 'text' and 'box' fields
        color: Color for the bounding boxes
        box_type: "line", "word", or "paragraph" for labeling
        show_labels: Whether to show text labels on boxes
    """
    # Create a copy to avoid modifying original
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Try to use a decent font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 12)
            small_font = ImageFont.truetype("arial.ttf", 10)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
    
    for i, item in enumerate(ocr_items):
        try:
            # Extract box coordinates [x1, y1, x2, y2]
            box = item.get('box', [])
            if len(box) != 4:
                print(f"Warning: Invalid box format in {box_type} {i}: {box}")
                continue
                
            x1, y1, x2, y2 = box
            
            # Draw rectangle outline
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            if show_labels:
                # Get text for labeling (truncate if too long)
                text = item.get('text', f'{box_type}_{i}')
                if len(text) > 20:
                    text = text[:17] + "..."
                
                # Position label at top-left of bounding box
                label_x, label_y = x1, y1 - 15
                
                # Ensure label is within image bounds
                if label_y < 0:
                    label_y = y1 + 2
                
                # Draw label background
                bbox = draw.textbbox((label_x, label_y), text, font=small_font)
                draw.rectangle(bbox, fill=color, outline=color)
                
                # Draw label text
                draw.text((label_x, label_y), text, fill="white", font=small_font)
                
        except (KeyError, TypeError, ValueError) as e:
            print(f"Warning: Invalid data in {box_type} {i}: {e}")
            continue
    
    return img_copy

def process_page(image_path: Path, json_path: Path, output_dir: Path, 
                pdf_id: str, page_num: int) -> None:
    """Process a single page and create visualization images."""
    
    # Load image
    try:
        image = Image.open(image_path)
        print(f"Loaded image: {image_path} ({image.size})")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return
    
    # Load OCR data
    ocr_data = load_pubmed_ocr_data(json_path)
    
    lines_data = ocr_data.get('lines_data', [])
    words_data = ocr_data.get('words_data', [])
    paragraphs_data = ocr_data.get('paragraphs_data', [])
    
    print(f"Page {page_num}: {len(lines_data)} lines, {len(words_data)} words, {len(paragraphs_data)} paragraphs")
    
    if not lines_data and not words_data and not paragraphs_data:
        print(f"Warning: No OCR data found for page {page_num}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate lines visualization
    if lines_data:
        lines_image = draw_bounding_boxes(image, lines_data, "red", "line")
        lines_output = output_dir / f"{pdf_id}_p{page_num}_lines.png"
        lines_image.save(lines_output)
        print(f"Saved lines visualization: {lines_output}")
    
    # Generate words visualization
    if words_data:
        words_image = draw_bounding_boxes(image, words_data, "blue", "word", show_labels=False)
        words_output = output_dir / f"{pdf_id}_p{page_num}_words.png"
        words_image.save(words_output)
        print(f"Saved words visualization: {words_output}")
    
    # Generate paragraphs visualization
    if paragraphs_data:
        paragraphs_image = draw_bounding_boxes(image, paragraphs_data, "green", "paragraph")
        paragraphs_output = output_dir / f"{pdf_id}_p{page_num}_paragraphs.png"
        paragraphs_image.save(paragraphs_output)
        print(f"Saved paragraphs visualization: {paragraphs_output}")
    
    # Generate combined visualization
    if lines_data or words_data or paragraphs_data:
        combined_image = image.copy()
        # Draw in order: paragraphs (green), words (blue), lines (red) so lines are on top
        if paragraphs_data:
            combined_image = draw_bounding_boxes(combined_image, paragraphs_data, "green", "paragraph", show_labels=False)
        if words_data:
            combined_image = draw_bounding_boxes(combined_image, words_data, "blue", "word", show_labels=False)
        if lines_data:
            combined_image = draw_bounding_boxes(combined_image, lines_data, "red", "line", show_labels=False)
        combined_output = output_dir / f"{pdf_id}_p{page_num}_combined.png"
        combined_image.save(combined_output)
        print(f"Saved combined visualization: {combined_output}")

def find_pdf_pages(input_dir: Path, pdf_id: str) -> List[int]:
    """Find all available pages for a given PDF ID."""
    pages = []
    
    # Check both raw and data directories
    for subdir in ['raw', 'data']:
        dir_path = input_dir / subdir
        if not dir_path.exists():
            continue
            
        # Look for JSON files with the PDF ID
        json_pattern = f"{pdf_id}_*.json"
        json_files = list(dir_path.glob(json_pattern))
        
        for json_file in json_files:
            # Extract page number from filename like "IJEM-16-117.PMC3354931_1.json"
            match = re.search(rf"{re.escape(pdf_id)}_(\d+)\.json$", json_file.name)
            if match:
                page_num = int(match.group(1))
                pages.append(page_num)
    
    return sorted(list(set(pages)))

def find_all_pdfs(input_dir: Path) -> List[str]:
    """Find all PDF IDs in the input directory."""
    pdf_ids = set()
    
    # Check both raw and data directories
    for subdir in ['raw', 'data']:
        dir_path = input_dir / subdir
        if not dir_path.exists():
            continue
            
        # Look for JSON files
        json_files = list(dir_path.glob("*.json"))
        
        for json_file in json_files:
            # Extract PDF ID from filename like "IJEM-16-117.PMC3354931_1.json"
            match = re.match(r"(.+)_\d+\.json$", json_file.name)
            if match:
                pdf_id = match.group(1)
                pdf_ids.add(pdf_id)
    
    return sorted(list(pdf_ids))

def main():
    print("Starting PubMed OCR visualization...")
    parser = argparse.ArgumentParser(
        description="Visualize PubMed OCR bounding boxes",
        epilog="""
Examples:
  # Visualize specific pages of a PDF
  python visualize_pubmed_ocr.py --input /tmp/pubmed_test --pdf-id IJEM-16-117.PMC3354931 --pages 1,2,3
  
  # Visualize all pages of a PDF
  python visualize_pubmed_ocr.py --input /tmp/pubmed_test --pdf-id IJEM-16-117.PMC3354931 --all-pages
  
  # Visualize all PDFs in directory
  python visualize_pubmed_ocr.py --input /tmp/pubmed_test --all-pdfs --output /tmp/all_viz
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, type=Path,
                       help="Input directory containing raw/ and data/ subdirectories")
    parser.add_argument("--pdf-id", type=str,
                       help="PDF ID (e.g., IJEM-16-117.PMC3354931)")
    parser.add_argument("--pages", type=str,
                       help="Comma-separated list of page numbers (e.g., 1,2,3)")
    parser.add_argument("--all-pages", action="store_true",
                       help="Process all available pages for the specified PDF")
    parser.add_argument("--all-pdfs", action="store_true",
                       help="Process all PDFs in the input directory")
    parser.add_argument("--output", type=Path, default="pubmed_ocr_visualizations",
                       help="Output directory for visualization images")
    parser.add_argument("--use-data-images", action="store_true",
                       help="Use PNG images from data/ directory (default behavior)")
    parser.add_argument("--list-pdfs", action="store_true",
                       help="List all available PDF IDs and exit")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}")
        return
    
    # Handle --list-pdfs option
    if args.list_pdfs:
        pdf_ids = find_all_pdfs(args.input)
        if pdf_ids:
            print(f"Found {len(pdf_ids)} PDFs in {args.input}:")
            for pdf_id in pdf_ids:
                pages = find_pdf_pages(args.input, pdf_id)
                print(f"  {pdf_id} ({len(pages)} pages: {pages})")
        else:
            print(f"No PDFs found in {args.input}")
        return
    
    # Determine which PDFs to process
    if args.all_pdfs:
        pdf_ids = find_all_pdfs(args.input)
        if not pdf_ids:
            print("Error: No PDFs found in input directory")
            return
        print(f"Found {len(pdf_ids)} PDFs: {pdf_ids}")
    elif args.pdf_id:
        pdf_ids = [args.pdf_id]
    else:
        print("Error: Must specify either --pdf-id or --all-pdfs")
        return
    
    # Process each PDF
    total_pages_processed = 0
    for pdf_id in pdf_ids:
        print(f"\nProcessing PDF: {pdf_id}")
        
        # Determine which pages to process
        if args.all_pages or args.all_pdfs:
            pages = find_pdf_pages(args.input, pdf_id)
            if not pages:
                print(f"Warning: No pages found for PDF {pdf_id}")
                continue
        elif args.pages:
            try:
                pages = [int(p.strip()) for p in args.pages.split(',')]
            except ValueError:
                print("Error: Invalid page numbers. Use comma-separated integers.")
                continue
        else:
            print("Error: Must specify --pages, --all-pages, or --all-pdfs")
            continue
        
        print(f"Processing {len(pages)} pages: {pages}")
        
        # Create output directory for this PDF
        pdf_output_dir = args.output / pdf_id
        
        # Process each page
        for page_num in pages:
            print(f"\nProcessing page {page_num} of {pdf_id}...")
            
            # Find image file
            if args.use_data_images:
                # Use PNG from data directory
                image_path = args.input / "data" / f"{pdf_id}_{page_num}.png"
            else:
                # Use PNG from data directory (default behavior)
                image_path = args.input / "data" / f"{pdf_id}_{page_num}.png"
            
            # Find JSON file (prefer data directory, fallback to raw)
            json_path = args.input / "data" / f"{pdf_id}_{page_num}.json"
            if not json_path.exists():
                json_path = args.input / "raw" / f"{pdf_id}_{page_num}.json"
            
            # Check if files exist
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
                
            if not json_path.exists():
                print(f"Warning: JSON not found: {json_path}")
                continue
            
            # Process this page
            process_page(image_path, json_path, pdf_output_dir, pdf_id, page_num)
            total_pages_processed += 1
    
    print(f"\n🎉 Visualization complete!")
    print(f"Processed {total_pages_processed} pages from {len(pdf_ids)} PDFs")
    print(f"Output directory: {args.output}")
    print("\nLegend:")
    print("- Red boxes: Line-level OCR detections")
    print("- Blue boxes: Word-level OCR detections") 
    print("- Green boxes: Paragraph-level OCR detections")
    print("- Combined view shows all three overlaid")

if __name__ == "__main__":
    main()
