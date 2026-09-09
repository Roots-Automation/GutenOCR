"""Generate 100 samples with the FULL production pipeline and produce annotated visualizations.

Each output PNG is a 3-panel composite:
  left   – clean render
  center – AABB boxes  (block=blue thick, line=green medium, word=red thin)
  right  – quad outlines (line=green, word=red filled; block has no quad so
           blue AABB is repeated for reference)

Effects enabled (mirrors config_base.yaml):
  - Real backgrounds with gaussian blur
  - Paper texture alpha-blending, staining, color tint
  - Elastic distortion, gaussian noise, erode/dilate, coarse dropout, perspective warp
  - Background shadow, document shadow
  - Page skew
  - Final color/grayscale/contrast/brightness/motion blur/gaussian blur/resample/JPEG effects
  - Physical overlays: book spine shadow, fold crease, moiré, low-toner streaks, vignetting, watermark

Run:  uv run python3 viz_samples_full.py [--n 100] [--out outputs/viz_full]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from template import SynthDoG  # noqa: E402

_BASE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Full production config — mirrors config_base.yaml with absolute paths.
# emit_quads forced True for quad visualization.
# Quality filters kept at production values so the visual output is realistic.
# ---------------------------------------------------------------------------
_CFG = {
    "quality": [50, 95],
    "landscape": 0.5,
    "short_size": [720, 1440],
    "aspect_ratio": [1.0, 2.5],
    "emit_quads": True,
    "min_bbox_area": 16,
    "min_contrast_ratio": 1.5,
    "min_word_count": 5,
    "max_textbox_null_frac": 0.95,
    "min_line_height_px": 15.0,
    "min_sharpness": 10.0,
    "max_intra_block_line_overlap": 0.95,
    "max_cross_block_line_overlap": 0.50,
    "background": {
        "image": {"paths": [str(_BASE / "resources/background")], "weights": [1]},
        "effect": {
            "args": [
                {"prob": 1, "args": {"sigma": [0, 10]}},
            ]
        },
    },
    "document": {
        "fullscreen": 0.5,
        "landscape": 0.5,
        "short_size": [480, 1440],
        "aspect_ratio": [1.0, 2.5],
        "paper": {
            "color": {
                "prob": 0.5,
                "rgb": [[0, 255], [0, 255], [0, 255]],
            },
            "image": {
                "paths": [str(_BASE / "resources/paper")],
                "weights": [1],
                "alpha": [0, 0.2],
                "grayscale": 1,
                "crop": 1,
            },
            "stain": {
                "prob": 0.15,
                "args": {
                    "count": [1, 3],
                    "size": [0.03, 0.12],
                    "alpha": [0.05, 0.20],
                    "color_r": [110, 180],
                    "color_g": [80, 140],
                    "color_b": [50, 110],
                },
            },
        },
        "content": {
            "text": {"path": str(_BASE / "resources/corpus/enwiki.txt")},
            "margin": [0, 0.1],
            "font": {"paths": [str(_BASE / "resources/font/en")], "weights": [1], "bold": 0},
            "layout": {
                "text_scale": [0.0334, 0.1],
                "max_row": 10,
                "max_col": 3,
                "fill": [0.5, 1],
                "full": 0.1,
                "align": ["left", "right", "center"],
                "stack_spacing": [0.02, 0.06],
                "stack_fill": [0.5, 1],
                "stack_full": 0.1,
            },
            "textbox": {"fill": [0.5, 1]},
            "textbox_color": {"prob": 1.0, "args": {"colorize": 1}},
            "content_color": {"prob": 0.2, "args": {"colorize": 1}},
            "text_sprinkle": {"prob": 0.3, "args": {"prob": [0.05, 0.15], "offset": [-1, 1]}},
            "page_header": {
                "prob": 0.25,
                "height": [0.06, 0.10],
                "text_scale": [0.040, 0.070],
                "max_col": 3,
            },
            "page_footer": {
                "prob": 0.30,
                "height": [0.06, 0.10],
                "text_scale": [0.040, 0.070],
                "max_col": 3,
                "page_number": {"prob": 0.50},
            },
            "section_heading": {
                "prob": 0.30,
                "text_scale": [0.06, 0.12],
                "height": [0.08, 0.16],
                "max_col": 1,
            },
            "footnote": {
                "prob": 0.20,
                "height": [0.08, 0.15],
                "text_scale": [0.035, 0.060],
                "max_col": 1,
            },
        },
        "effect": {
            "args": [
                {"prob": 1, "args": {"alpha": [0, 1], "sigma": [0, 0.5]}},
                {"prob": 1, "args": {"scale": [0, 8], "per_channel": 0}},
                {"prob": 0.15, "args": {"k": [1, 2]}},
                {"prob": 0.15, "args": {"k": [1, 2]}},
                {"prob": 0.2, "args": {"p": [0.003, 0.015], "size_percent": [0.05, 0.2], "per_channel": 0}},
                {
                    "prob": 1,
                    "args": {
                        "weights": [750, 50, 50, 25, 25, 25, 25, 50],
                        "args": [
                            {"percents": [[0.88, 1], [0.88, 1], [0.88, 1], [0.88, 1]]},
                            {"percents": [[0.88, 1], [1, 1], [0.88, 1], [1, 1]]},
                            {"percents": [[1, 1], [0.88, 1], [1, 1], [0.88, 1]]},
                            {"percents": [[0.88, 1], [1, 1], [1, 1], [1, 1]]},
                            {"percents": [[1, 1], [0.88, 1], [1, 1], [1, 1]]},
                            {"percents": [[1, 1], [1, 1], [0.88, 1], [1, 1]]},
                            {"percents": [[1, 1], [1, 1], [1, 1], [0.88, 1]]},
                            {"percents": [[1, 1], [1, 1], [1, 1], [1, 1]]},
                        ],
                    },
                },
            ]
        },
    },
    "bg_effect": {
        "args": [
            {
                "prob": 1,
                "args": {"intensity": [0, 80], "amount": [0, 0.5], "smoothing": [0.5, 1], "bidirectional": 0},
            }
        ]
    },
    "doc_effect": {
        "args": [
            {
                "prob": 1,
                "args": {"intensity": [0, 80], "amount": [0, 0.5], "smoothing": [0.5, 1], "bidirectional": 0},
            }
        ]
    },
    "effect": {
        "args": [
            {"prob": 0.2, "args": {"rgb": [[0, 255], [0, 255], [0, 255]], "alpha": [0, 0.2]}},
            {"prob": 0.05, "args": {}},
            {"prob": 1, "args": {"alpha": [1, 1.5]}},
            {"prob": 1, "args": {"beta": [-32, 32]}},
            {"prob": 0.5, "args": {"k": [3, 5], "angle": [0, 360]}},
            {"prob": 1, "args": {"sigma": [0, 1.5]}},
            {"prob": 0.2, "args": {"size": [0.4, 0.7]}},
            {"prob": 0.3, "args": {"compression": [10, 40]}},
        ]
    },
    "skew": {"prob": 0.3, "angle": [-3, 3]},
    "book_spine_shadow": {
        "prob": 0.25,
        "args": {"side": "random", "intensity": [30, 100], "width": [0.05, 0.20]},
    },
    "fold_crease": {
        "prob": 0.15,
        "args": {"count": [1, 2], "orientation": "random", "width": [1, 4], "intensity": [10, 45]},
    },
    "moire": {
        "prob": 0.10,
        "args": {"frequency": [0.02, 0.06], "angle": [0, 90], "alpha": [0.05, 0.15]},
    },
    "low_toner_streaks": {
        "prob": 0.15,
        "args": {"orientation": "random", "count": [1, 4], "width": [3, 20], "intensity": [0.05, 0.20]},
    },
    "vignetting": {
        "prob": 0.30,
        "args": {"intensity": [30, 80], "shape": [1.5, 3.0], "mode": "radial"},
    },
    "watermark": {
        "prob": 0.10,
        "args": {
            "words": ["DRAFT", "CONFIDENTIAL", "COPY", "VOID", "SAMPLE", "APPROVED", "RECEIVED"],
            "font_path": str(_BASE / "resources/font/en/regular/NotoSans-Regular.ttf"),
            "font_size_frac": [0.12, 0.22],
            "alpha": [0.08, 0.25],
            "angle": [-45, 45],
            "color": [60, 60, 60],
        },
    },
}

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
_BLOCK_COL = (30, 100, 220)  # blue
_LINE_COL = (20, 170, 60)  # green
_WORD_COL = (210, 40, 40)  # red


def _px(norm: list, W: int, H: int) -> tuple:
    x1, y1, x2, y2 = norm
    return int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)


def _quad_px(quad: list, W: int, H: int) -> list:
    return [(int(p[0] * W), int(p[1] * H)) for p in quad]


def _annotate_aabb(base: Image.Image, data: dict) -> Image.Image:
    img = base.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    W, H = img.size

    for blk in data["blocks"]:
        x1, y1, x2, y2 = _px(blk.bbox, W, H)
        draw.rectangle([x1, y1, x2, y2], outline=(*_BLOCK_COL, 230), width=3)

    for ln in data["lines"]:
        x1, y1, x2, y2 = _px(ln.bbox, W, H)
        draw.rectangle([x1, y1, x2, y2], outline=(*_LINE_COL, 210), width=2)

    for wd in data["words"]:
        x1, y1, x2, y2 = _px(wd.bbox, W, H)
        draw.rectangle([x1, y1, x2, y2], outline=(*_WORD_COL, 190), width=1)

    return Image.alpha_composite(img, overlay).convert("RGB")


def _annotate_quads(base: Image.Image, data: dict) -> Image.Image:
    img = base.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    W, H = img.size

    # Blocks: no quad — draw AABB for reference
    for blk in data["blocks"]:
        x1, y1, x2, y2 = _px(blk.bbox, W, H)
        draw.rectangle([x1, y1, x2, y2], outline=(*_BLOCK_COL, 230), width=3)

    # Lines: quad outline
    for ln in data["lines"]:
        if ln.quad:
            pts = _quad_px(ln.quad, W, H)
            draw.polygon(pts, outline=(*_LINE_COL, 220))
            draw.line(pts + [pts[0]], fill=(*_LINE_COL, 220), width=2)

    # Words: quad with faint fill
    for wd in data["words"]:
        if wd.quad:
            pts = _quad_px(wd.quad, W, H)
            draw.polygon(pts, fill=(*_WORD_COL, 35), outline=(*_WORD_COL, 200))

    return Image.alpha_composite(img, overlay).convert("RGB")


def _separator(H: int, width: int = 4) -> Image.Image:
    return Image.new("RGB", (width, H), (80, 80, 80))


def _composite(base: Image.Image, aabb: Image.Image, quads: Image.Image) -> Image.Image:
    W, H = base.size
    sep = _separator(H)
    out = Image.new("RGB", (W * 3 + 8, H), (60, 60, 60))
    out.paste(base, (0, 0))
    out.paste(sep, (W, 0))
    out.paste(aabb, (W + 4, 0))
    out.paste(sep, (W * 2 + 4, 0))
    out.paste(quads, (W * 2 + 8, 0))
    return out


def _label_strip(width: int, seed: int, data: dict) -> Image.Image:
    h = 22
    strip = Image.new("RGB", (width, h), (30, 30, 30))
    draw = ImageDraw.Draw(strip)
    n_blk = len(data["blocks"])
    n_ln = len(data["lines"])
    n_wd = len(data["words"])
    txt = (
        f"seed={seed:04d}   "
        f"blocks={n_blk}  lines={n_ln}  words={n_wd}   "
        f"sharpness={data['quality_metrics'].get('sharpness', '?')}"
    )
    draw.text((6, 4), txt, fill=(200, 200, 200))
    return strip


def _passes_quality_filters(data: dict, cfg: dict) -> tuple[bool, str]:
    """Return (passes, reason) applying the same gates as SynthDoG.save()."""
    qm = data.get("quality_metrics", {})

    contrast = qm.get("min_line_contrast_ratio")
    threshold = cfg.get("min_contrast_ratio", 1.0)
    if contrast is not None and contrast < threshold:
        return False, f"contrast {contrast:.3f} < {threshold}"

    words = qm.get("word_count", 0)
    min_words = cfg.get("min_word_count", 1)
    if words < min_words:
        return False, f"words {words} < {min_words}"

    null_frac = qm.get("textbox_null_frac") or 0.0
    max_null = cfg.get("max_textbox_null_frac", 1.0)
    if null_frac > max_null:
        return False, f"null_frac {null_frac:.3f} > {max_null}"

    min_h = qm.get("min_line_height_px")
    thresh_h = cfg.get("min_line_height_px", 0.0)
    if min_h is not None and min_h < thresh_h:
        return False, f"min_line_height {min_h:.1f} < {thresh_h}"

    sharp = qm.get("sharpness")
    min_sharp = cfg.get("min_sharpness", 0.0)
    if sharp is not None and sharp < min_sharp:
        return False, f"sharpness {sharp:.1f} < {min_sharp}"

    intra = qm.get("max_intra_block_line_overlap")
    max_intra = cfg.get("max_intra_block_line_overlap", 1.0)
    if intra is not None and intra > max_intra:
        return False, f"intra_overlap {intra:.3f} > {max_intra}"

    cross = qm.get("max_cross_block_line_overlap")
    max_cross = cfg.get("max_cross_block_line_overlap", 1.0)
    if cross is not None and cross > max_cross:
        return False, f"cross_overlap {cross:.3f} > {max_cross}"

    return True, "ok"


def generate_viz(n: int, out_dir: Path, seed_start: int = 0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dog = SynthDoG(_CFG)

    saved = 0
    discarded = 0
    seed = seed_start
    while saved < n:
        try:
            data = dog.generate(seed=seed)
        except Exception as e:
            print(f"  seed={seed}: error — {e}")
            seed += 1
            continue

        if data is None:
            seed += 1
            discarded += 1
            continue

        passes, reason = _passes_quality_filters(data, _CFG)
        if not passes:
            seed += 1
            discarded += 1
            continue

        img_arr = np.clip(data["image"][..., :3], 0, 255).astype(np.uint8)
        base = Image.fromarray(img_arr)

        aabb = _annotate_aabb(base, data)
        quads = _annotate_quads(base, data)

        comp = _composite(base, aabb, quads)
        W_out = comp.width
        label = _label_strip(W_out, seed, data)

        final = Image.new("RGB", (W_out, comp.height + label.height), (30, 30, 30))
        final.paste(comp, (0, 0))
        final.paste(label, (0, comp.height))
        final.save(out_dir / f"sample_{saved:03d}.png")

        saved += 1
        seed += 1

        if saved % 10 == 0:
            total_attempted = seed - seed_start
            print(f"  {saved}/{n}  (seed={seed - 1}, discarded={discarded}/{total_attempted})")

    # HTML gallery
    html_path = out_dir / "index.html"
    imgs = sorted(out_dir.glob("sample_*.png"))
    rows = "\n".join(f'<div class="card"><img src="{p.name}" loading="lazy"><p>{p.stem}</p></div>' for p in imgs)
    html_path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>SynthDoG 100-sample full-pipeline annotation review</title>
<style>
  body {{ background:#1a1a1a; color:#ccc; font-family:monospace; margin:0; padding:12px }}
  h1   {{ font-size:14px; margin-bottom:8px }}
  p.legend {{ font-size:12px; margin-bottom:14px }}
  .grid {{ display:flex; flex-wrap:wrap; gap:8px }}
  .card {{ background:#2a2a2a; padding:4px; border-radius:3px }}
  .card img {{ max-width:100%; display:block }}
  .card p {{ font-size:11px; margin:3px 0 0; text-align:center; color:#888 }}
</style></head>
<body>
<h1>SynthDoG annotation review — full pipeline — {n} samples</h1>
<p class="legend">
  Full production effects: perspective warp, elastic distortion, noise, shadow, skew,
  blur, vignetting, physical overlays.<br>
  <span style="color:#7eb3ff">■ block AABB (blue)</span> &nbsp;
  <span style="color:#4dba5c">■ line AABB / quad (green)</span> &nbsp;
  <span style="color:#e05050">■ word AABB / quad (red)</span>
  &nbsp;|&nbsp; left = clean &nbsp; centre = AABB &nbsp; right = quads
</p>
<div class="grid">
{rows}
</div>
</body></html>
""",
        encoding="utf-8",
    )
    total_attempted = seed - seed_start
    print(f"\nDone: {saved} saved, {discarded} discarded ({discarded / total_attempted * 100:.1f}% discard rate)")
    print(f"Gallery → {html_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=str, default="outputs/viz_full")
    ap.add_argument("--seed-start", type=int, default=0)
    args = ap.parse_args()
    generate_viz(args.n, Path(args.out), args.seed_start)
