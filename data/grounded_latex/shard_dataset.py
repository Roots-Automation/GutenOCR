"""
Usage:
    This script prepares standardized shards from LaTeX PDF + JSON pairs.

    Input:
        - Source directory (SRC_DIR): expected to contain subfolders with files:
            <doc_id>.pdf and <doc_id>.json
          where the JSON has keys "latex" (string) and "bbox" (list of PDF coords).
    Processing:
        - Each PDF is rasterized into a PNG image at DPI=72 using pdf2image.
        - The corresponding JSON annotation is converted to exact pixel coordinates
          with top-left origin, aligned with the rendered PNG dimensions.
        - For each document, three files are written into the shard:
            <doc_id>.pdf
            <doc_id>_00.png
            <doc_id>_00.json (updated schema with image info + pixel bbox coordinates)

    Output:
        - Shards are written to OUT_DIR as tar archives:
            train-00000.tar, train-00001.tar, ...
        - Each shard contains up to SHARD_SIZE documents.
    How to run:
        - Adjust SRC_DIR, OUT_DIR, SHARD_SIZE as needed.
        - Run the script directly:
            python script_name.py
        - The script will iterate through JSON/PDF pairs, process them, and
          create tar shard files in OUT_DIR.
"""

import os
import json
import tarfile
import argparse
from pathlib import Path
from pdf2image import convert_from_path
from pdfrw import PdfReader
from PIL import Image
from multiprocessing import Pool, cpu_count

# Default configuration (can be overridden via CLI arguments)
DEFAULT_SRC_DIR = "./input"
DEFAULT_OUT_DIR = "./output"
DEFAULT_SHARD_SIZE = 2048
DEFAULT_DPI = 72

def pdf_to_pixel_bbox(bbox, pdf_path, dpi):
    """Convert absolute PDF bbox in points to exact pixel coordinates."""
    page = PdfReader(pdf_path).pages[0]
    x0, y0, X, Y = map(float, page.MediaBox)

    scale = dpi / 72.0
    x1, y1, x2, y2 = bbox

    x1_px = (x1 - x0) * scale
    x2_px = (x2 - x0) * scale
    y1_px = (y1 - y0) * scale
    y2_px = (y2 - y0) * scale

    return [
        round(x1_px),
        round(y1_px),
        round(x2_px),
        round(y2_px),
    ]

def process_doc(args):
    """Convert one PDF+JSON doc into image + annotation JSON."""
    doc_id, pdf_path, json_path, dpi = args
    images = convert_from_path(pdf_path, dpi=dpi)
    assert len(images) == 1, f"Expected 1 page, got {len(images)}"
    img = images[0]

    img_path = f"{doc_id}.png"
    img.save(img_path, "PNG")

    with open(json_path, "r") as f:
        orig = json.load(f)

    width, height = img.size

    new_json = {
        "text": {
            "latex": [
                {
                    "text": orig["latex"],
                    "box": pdf_to_pixel_bbox(orig["bbox"], pdf_path, dpi)
                }
            ]
        },
        "image": {
            "path": img_path,
            "width": width,
            "height": height,
            "dpi": dpi,
        },
    }
    return doc_id, img_path, new_json

def make_shard(shard_idx, docs, out_dir, dpi):
    """Write a tar shard with a batch of documents."""
    shard_path = out_dir / f"train-{shard_idx:05d}.tar"
    
    # Prepare args for parallel processing
    process_args = [(doc_id, pdf_path, json_path, dpi) for doc_id, pdf_path, json_path in docs]
    
    # Process documents in parallel
    with Pool(processes=min(cpu_count(), len(docs))) as pool:
        results = pool.map(process_doc, process_args)
    
    # Write to tar file
    with tarfile.open(shard_path, "w") as tar:
        for (doc_id, pdf_path, json_path), (_, img_path, ann) in zip(docs, results):
            tar.add(pdf_path, arcname=f"{doc_id}.pdf")
            tar.add(img_path, arcname=f"{doc_id}.png")

            tmp_json = f"{doc_id}.json"
            with open(tmp_json, "w") as f:
                json.dump(ann, f, indent=2)
            tar.add(tmp_json, arcname=tmp_json)

            os.remove(img_path)
            os.remove(tmp_json)

    print(f"Created shard {shard_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Prepare standardized shards from LaTeX PDF + JSON pairs."
    )
    parser.add_argument(
        "--src-dir", type=Path, default=Path(DEFAULT_SRC_DIR),
        help=f"Source directory containing PDF/JSON pairs (default: {DEFAULT_SRC_DIR})"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR),
        help=f"Output directory for tar shards (default: {DEFAULT_OUT_DIR})"
    )
    parser.add_argument(
        "--shard-size", type=int, default=DEFAULT_SHARD_SIZE,
        help=f"Number of documents per shard (default: {DEFAULT_SHARD_SIZE})"
    )
    parser.add_argument(
        "--dpi", type=int, default=DEFAULT_DPI,
        help=f"DPI for PNG rendering (default: {DEFAULT_DPI})"
    )
    parser.add_argument(
        "--max-docs", type=int, default=None,
        help="Maximum documents to process (default: all)"
    )
    args = parser.parse_args()

    # Validate source directory
    if not args.src_dir.exists():
        print(f"Error: Source directory '{args.src_dir}' does not exist.")
        return 1
    
    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source: {args.src_dir}")
    print(f"Output: {args.out_dir}")
    print(f"Shard size: {args.shard_size}, DPI: {args.dpi}")
    print("Collecting all documents...")
    
    docs = []

    for json_file in args.src_dir.glob("**/*.json"):
        if args.max_docs is not None and len(docs) >= args.max_docs:
            break
            
        pdf_file = json_file.with_suffix(".pdf")
        if not pdf_file.exists():
            continue

        stem = json_file.stem
        parent = json_file.parent.name

        # Keep the full stem (including rotation suffixes) for the doc_id
        doc_id = f"{parent}_{stem}"
        doc_tuple = (doc_id, str(pdf_file), str(json_file))
        docs.append(doc_tuple)

    print(f"Found {len(docs)} documents.")

    if len(docs) == 0:
        print("No documents found. Check your source directory.")
        return 1

    print("Creating shards...")
    for shard_idx in range(0, len(docs), args.shard_size):
        shard_docs = docs[shard_idx: shard_idx + args.shard_size]
        make_shard(shard_idx // args.shard_size, shard_docs, args.out_dir, args.dpi)
    
    print("Done!")
    return 0

if __name__ == "__main__":
    exit(main())
