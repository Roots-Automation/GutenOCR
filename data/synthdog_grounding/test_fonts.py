#!/usr/bin/env python3
"""
Per-font isolation test: generates 10 samples per English font with emit_quads=True,
then annotates each image with both AABB rectangles and quad polygons.

Usage:
    cd data/synthdog_grounding
    uv run python test_fonts.py [--output outputs/font_test] [--count 10]
"""

import argparse
import copy
import json
import os
from pathlib import Path

import pillow_compat  # noqa: F401 — patches Pillow 10+ compat methods
import yaml
from PIL import Image, ImageDraw

# ── Colours for annotation ────────────────────────────────────────────────────
LINE_AABB_COLOR = "lime"
WORD_AABB_COLOR = "yellow"
LINE_QUAD_COLOR = (0, 128, 255)  # blue
WORD_QUAD_COLOR = (255, 128, 0)  # orange
LINE_WIDTH = 2


# ── Geometry helpers ──────────────────────────────────────────────────────────


def denorm_bbox(bbox, W, H):
    x1, y1, x2, y2 = bbox
    return (
        int(round(x1 * W)),
        int(round(y1 * H)),
        int(round(x2 * W)),
        int(round(y2 * H)),
    )


def denorm_quad(quad, W, H):
    return [(int(round(x * W)), int(round(y * H))) for x, y in quad]


# ── Annotation ────────────────────────────────────────────────────────────────


def annotate(img_path: Path, metadata: dict, out_path: Path):
    gt = json.loads(metadata["ground_truth"])["gt_parse"]
    lines = gt.get("text_lines", [])
    words = gt.get("text_words", [])

    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    # Words: AABB (yellow) then quad (orange)
    for wd in words:
        if "bbox" in wd:
            b = denorm_bbox(wd["bbox"], W, H)
            draw.rectangle([b[0], b[1], b[2], b[3]], outline=WORD_AABB_COLOR, width=1)
        if "quad" in wd:
            pts = denorm_quad(wd["quad"], W, H)
            draw.polygon(pts, outline=WORD_QUAD_COLOR)

    # Lines: AABB (green) then quad (blue)
    for ln in lines:
        if "bbox" in ln:
            b = denorm_bbox(ln["bbox"], W, H)
            draw.rectangle([b[0], b[1], b[2], b[3]], outline=LINE_AABB_COLOR, width=LINE_WIDTH)
        if "quad" in ln:
            pts = denorm_quad(ln["quad"], W, H)
            draw.polygon(pts, outline=LINE_QUAD_COLOR, width=LINE_WIDTH)

    img.save(out_path, quality=95)


# ── Generation ────────────────────────────────────────────────────────────────


def load_base_config():
    """Load and merge config_en.yaml (which inherits config_base.yaml)."""
    base_dir = Path(__file__).parent

    with open(base_dir / "config/config_base.yaml") as f:
        base = yaml.safe_load(f)
    with open(base_dir / "config/config_en.yaml") as f:
        overlay = yaml.safe_load(f)

    # Remove the _base key that synthtiger normally resolves
    overlay.pop("_base", None)

    def deep_merge(base, override):
        result = copy.deepcopy(base)
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = deep_merge(result[k], v)
            else:
                result[k] = copy.deepcopy(v)
        return result

    return deep_merge(base, overlay)


def generate_for_font(font_path: Path, out_dir: Path, count: int, base_config: dict):
    """Generate `count` samples using only `font_path` and save to `out_dir`."""
    from template import SynthDoG

    config = copy.deepcopy(base_config)
    config["emit_quads"] = True

    # Use this single font file exclusively
    config["document"]["content"]["font"]["paths"] = [str(font_path)]
    config["document"]["content"]["font"]["weights"] = [1]

    t = SynthDoG(config)
    t.init_save(str(out_dir))
    for i in range(count):
        try:
            data = t.generate()
            t.save(str(out_dir), data, i)
        except Exception as e:
            print(f"  [!] sample {i} failed: {e}")


def annotate_output(out_dir: Path, font_stem: str):
    """Read all split metadata.jsonl files and produce annotated images."""
    annotated = 0
    for split_dir in out_dir.iterdir():
        if not split_dir.is_dir():
            continue
        jsonl = split_dir / "metadata.jsonl"
        if not jsonl.exists():
            continue
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                meta = json.loads(line)
                img_path = split_dir / meta["file_name"]
                if not img_path.exists():
                    continue
                ann_path = split_dir / (img_path.stem + "_annotated.jpg")
                try:
                    annotate(img_path, meta, ann_path)
                    annotated += 1
                except Exception as e:
                    print(f"  [!] annotation failed for {img_path.name}: {e}")
    return annotated


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Per-font isolation test with quad+AABB annotation")
    parser.add_argument("--output", default="outputs/font_test", help="Output root directory")
    parser.add_argument("--count", type=int, default=10, help="Samples per font")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    font_dir = script_dir / "resources/font/en"
    font_files = sorted(font_dir.glob("*.ttf")) + sorted(font_dir.glob("*.otf"))

    out_root = script_dir / args.output
    out_root.mkdir(parents=True, exist_ok=True)

    base_config = load_base_config()

    print(f"Testing {len(font_files)} fonts × {args.count} samples each")
    print(f"Output: {out_root}\n")

    for font_path in font_files:
        font_stem = font_path.stem
        out_dir = out_root / font_stem
        print(f"[{font_stem}] generating {args.count} samples …", flush=True)

        try:
            generate_for_font(font_path, out_dir, args.count, base_config)
            n = annotate_output(out_dir, font_stem)
            print(f"[{font_stem}] done — {n} annotated images saved")
        except Exception as e:
            print(f"[{font_stem}] FAILED: {e}")

    print("\nAll fonts tested.")


if __name__ == "__main__":
    main()
