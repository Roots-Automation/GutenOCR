#!/usr/bin/env python3
"""
SynthDoG Sample Inspector and Visualizer

This script extracts sample images and their annotations from SynthDoG tar archives
and creates visualized versions with bounding box annotations overlaid on the images.
It's useful for quality inspection, debugging, and understanding the generated data.

Features:
- Extracts specific samples or first N samples from tar archives
- Renders bounding box annotations on images
- Supports customizable visualization options (colors, line width, labels)
- Handles multiple image formats (JPG, PNG, TIFF, BMP)
- Saves both original and annotated versions for comparison

Usage:
    python check_sample.py <tar_file> [options]
    
    # Extract first 10 samples from a tar file
    python check_sample.py path/to/train-0011.tar -n 10
    
    # Extract specific sample IDs
    python check_sample.py path/to/train-0011.tar --ids 00087 00042 00013
    
    # Custom output directory and line width
    python check_sample.py path/to/train-0011.tar -o ./my_samples --line-width 5
    
    # Include text labels on annotations
    python check_sample.py path/to/train-0011.tar --label-with-text

Output:
    Creates files in the output directory (default: ./check_sample/):
    - {id}.jpg: Original image files
    - {id}_annotated.jpg: Images with bounding box annotations

Requirements:
    - Pillow (PIL) for image processing
    - Python 3.8+
    - Input tar file with paired image and JSON annotation files
"""

import argparse
from pathlib import Path
import tarfile
import json
import io
from PIL import Image, ImageDraw, ImageFont

# ---------------------------
# Default Config (overridable via CLI)
# ---------------------------
DEFAULT_LINE_WIDTH = 3
DEFAULT_FIRST_N = 25
DEFAULT_OUT_DIR = "./check_sample"


# ---------------------------
# Helpers
# ---------------------------
EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def find_image_member_for_id(tar: tarfile.TarFile, sample_id: str):
    for ext in EXTS:
        name = f"{sample_id}{ext}"
        try:
            return tar.getmember(name)
        except KeyError:
            continue
    raise FileNotFoundError(f"No image found for id {sample_id} (tried extensions {EXTS})")

def find_pairs_in_tar(tar: tarfile.TarFile):
    """
    Yield (id, img_member, json_member) for all id pairs in the tar.
    """
    # index members by name for quick lookup
    names = {m.name: m for m in tar.getmembers() if m.isfile()}
    # discover ids by image members
    for name, m in names.items():
        lower = name.lower()
        if lower.endswith(EXTS):
            sid = Path(name).stem
            jname = f"{sid}.json"
            if jname in names:
                yield sid, m, names[jname]

def parse_bbox(bbox, W, H):
    """
    Accepts [x1, y1, x2, y2] either normalized (0..1) or pixel.
    Returns integer pixel coords clamped to image size.
    """
    if bbox is None or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    # If it looks normalized (all within 0..1), scale up.
    if all(isinstance(v, (int, float)) for v in bbox) and 0 <= min(bbox) and max(bbox) <= 1.0000001:
        x1 *= W; y1 *= H; x2 *= W; y2 *= H
    # Ensure proper ordering
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    # Clamp and convert to ints
    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(0, min(W - 1, round(x2))))
    y2 = int(max(0, min(H - 1, round(y2))))
    # Guard: avoid zero-area boxes
    if x2 == x1: x2 = min(W - 1, x1 + 1)
    if y2 == y1: y2 = min(H - 1, y1 + 1)
    return (x1, y1, x2, y2)


def choose_font(img_w, font_path=None):
    """Choose an appropriate font for annotation labels."""
    # Aim for ~1.6% of width, 10–36px. Try a TTF first, else default bitmap font.
    target_px = max(10, min(36, int(img_w * 0.016)))
    for fp in [font_path, "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/Library/Fonts/Arial.ttf"]:
        if fp:
            try:
                return ImageFont.truetype(fp, target_px)
            except Exception:
                pass
    return ImageFont.load_default()

def text_size(draw, text, font):
    # Robust size measurement across Pillow versions
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return (right - left), (bottom - top)
    except Exception:
        try:
            # Older Pillow
            return draw.textsize(text, font=font)
        except Exception:
            # Last-resort guess
            return (8 * max(1, len(text)), 12)

def draw_label(draw, xy, label, font, pad=2):
    x, y = xy
    tw, th = text_size(draw, label, font)
    draw.rectangle([x, y, x + tw + 2 * pad, y + th + 2 * pad], fill="black")
    draw.text((x + pad, y + pad), label, fill="white", font=font)

def annotate_image(img_bytes: bytes, lines, out_path: Path, line_width: int = DEFAULT_LINE_WIDTH,
                   label_with_text: bool = False, font_path: str = None):
    """
    Annotate an image with bounding boxes for text lines.
    
    Args:
        img_bytes: Raw image bytes
        lines: List of dicts with {"bbox": [x1,y1,x2,y2], "line_id": int, "text": "..."}
        out_path: Path to save annotated image
        line_width: Width of bounding box lines
        label_with_text: Whether to include text content in labels
        font_path: Optional path to TTF font file
    """
    with Image.open(io.BytesIO(img_bytes)) as im:
        im = im.convert("RGB")  # ensure RGB for drawing
        W, H = im.size
        draw = ImageDraw.Draw(im)
        font = choose_font(W, font_path)

        for ln in lines:
            bbox = ln.get("bbox")
            lid = ln.get("line_id", "?")
            text = ln.get("text", "")
            box = parse_bbox(bbox, W, H)
            if not box:
                continue
            x1, y1, x2, y2 = box

            # Rectangle
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=line_width)

            # Label
            label = f"id={lid}" if not label_with_text else f"{lid}: {text}"

            # Measure label to decide placement
            tw, th = text_size(draw, label, font)

            # Prefer above the box if there's room, else inside
            needed = th + 6  # a little padding
            if y1 - needed >= 0:
                ly = y1 - needed
            else:
                ly = y1 + 2

            # Keep label horizontally within the image
            lx = max(0, min(W - (tw + 4), x1 + 2))

            draw_label(draw, (lx, ly), label, font)

        im.save(out_path, quality=95)
    return out_path

# ---------------------------
# Main workflow
# ---------------------------
def process_targets(tar_path: Path, out_dir: Path, target_ids=None, first_n=5,
                    line_width: int = DEFAULT_LINE_WIDTH, label_with_text: bool = False,
                    font_path: str = None):
    """
    Process samples from a tar archive and create annotated images.
    
    Args:
        tar_path: Path to the tar archive
        out_dir: Output directory for images
        target_ids: List of specific sample IDs to extract (or None for first_n)
        first_n: Number of samples to extract if target_ids is None
        line_width: Width of bounding box lines
        label_with_text: Whether to include text content in labels
        font_path: Optional path to TTF font file
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r") as tar:
        if target_ids:
            pairs = []
            for sid in target_ids:
                img_m = find_image_member_for_id(tar, sid)
                try:
                    json_m = tar.getmember(f"{sid}.json")
                except KeyError:
                    raise FileNotFoundError(f"Missing JSON for id {sid}")
                pairs.append((sid, img_m, json_m))
        else:
            # take first_n pairs in archive order
            pairs = []
            for i, (sid, img_m, json_m) in enumerate(find_pairs_in_tar(tar)):
                pairs.append((sid, img_m, json_m))
                if i + 1 >= first_n:
                    break

        for sid, img_m, json_m in pairs:
            # read JSON
            with tar.extractfile(json_m) as jf:
                obj = json.loads(jf.read().decode("utf-8"))
            lines = (obj.get("text") or {}).get("lines") or []
            print(f"[{sid}] {obj.get('image', '')} with {len(lines)} lines")

            # read image bytes
            with tar.extractfile(img_m) as imf:
                img_bytes = imf.read()

            # save original
            img_name = Path(img_m.name).name  # keep original extension
            orig_path = out_dir / img_name
            with open(orig_path, "wb") as f:
                f.write(img_bytes)

            # save annotated
            ann_name = f"{sid}_annotated{Path(img_name).suffix}"
            ann_path = out_dir / ann_name
            annotate_image(img_bytes, lines, ann_path, line_width, label_with_text, font_path)

            print(f"[{sid}] saved:")
            print(f"  - {orig_path}")
            print(f"  - {ann_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and annotate sample images from SynthDoG tar archives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract first 10 samples
    python check_sample.py path/to/train-0011.tar -n 10
    
    # Extract specific sample IDs
    python check_sample.py path/to/train-0011.tar --ids 00087 00042 00013
    
    # Custom output directory with text labels
    python check_sample.py path/to/train-0011.tar -o ./my_samples --label-with-text
        """
    )
    parser.add_argument(
        "tar_file",
        type=Path,
        help="Path to the tar archive to inspect"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path(DEFAULT_OUT_DIR),
        help=f"Output directory for images (default: {DEFAULT_OUT_DIR})"
    )
    parser.add_argument(
        "-n", "--first-n",
        type=int,
        default=DEFAULT_FIRST_N,
        help=f"Number of samples to extract (default: {DEFAULT_FIRST_N})"
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Specific sample IDs to extract (overrides --first-n)"
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=DEFAULT_LINE_WIDTH,
        help=f"Width of bounding box lines (default: {DEFAULT_LINE_WIDTH})"
    )
    parser.add_argument(
        "--label-with-text",
        action="store_true",
        help="Include text content in annotation labels"
    )
    parser.add_argument(
        "--font-path",
        type=str,
        default=None,
        help="Path to TTF font file for labels (optional)"
    )
    
    args = parser.parse_args()
    
    if not args.tar_file.exists():
        print(f"Error: Tar file not found: {args.tar_file}")
        return 1
    
    process_targets(
        tar_path=args.tar_file,
        out_dir=args.output,
        target_ids=args.ids,
        first_n=args.first_n,
        line_width=args.line_width,
        label_with_text=args.label_with_text,
        font_path=args.font_path
    )
    return 0


if __name__ == "__main__":
    exit(main())
