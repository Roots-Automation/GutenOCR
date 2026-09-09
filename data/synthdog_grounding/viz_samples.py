"""Generate 100 samples and produce annotated visualizations.

Each output PNG is a 3-panel composite:
  left   – clean render
  center – AABB boxes  (block=blue thick, line=green medium, word=red thin)
  right  – quad outlines (line=green, word=red filled; block has no quad so
           blue AABB is repeated for reference)

Run:  uv run python3 viz_samples.py [--n 100] [--out outputs/viz]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from template import SynthDoG  # noqa: E402

# ---------------------------------------------------------------------------
# Config — effects disabled so annotation alignment is easy to inspect.
# Varied seeds produce varied content / layout.
# ---------------------------------------------------------------------------
_BASE = Path(__file__).resolve().parent

_CFG = {
    "quality": [85, 85],
    "landscape": 0.0,
    "short_size": [480, 480],
    "aspect_ratio": [1.5, 1.5],
    "emit_quads": True,
    "min_bbox_area": 16,
    "min_contrast_ratio": 1.0,
    "min_word_count": 1,
    "max_textbox_null_frac": 1.0,
    "min_line_height_px": 1.0,
    "min_sharpness": 0.0,
    "max_intra_block_line_overlap": 1.0,
    "max_cross_block_line_overlap": 1.0,
    "background": {
        "image": {"paths": [str(_BASE / "resources/background")], "weights": [1]},
        "effect": {"args": [{"prob": 0, "args": {"sigma": [0, 0]}}]},
    },
    "document": {
        "fullscreen": 1.0,
        "landscape": 0.0,
        "short_size": [480, 480],
        "aspect_ratio": [1.5, 1.5],
        "paper": {
            "image": {
                "paths": [str(_BASE / "resources/paper")],
                "weights": [1],
                "alpha": [0, 0],
                "grayscale": 0,
                "crop": 0,
            }
        },
        "content": {
            "text": {"path": str(_BASE / "resources/corpus/enwiki.txt")},
            "font": {"paths": [str(_BASE / "resources/font/en")], "weights": [1]},
            "layout": {
                "text_scale": [0.05, 0.08],
                "max_row": 5,
                "max_col": 2,
                "fill": [0.5, 1],
                "full": 0,
            },
            "textbox": {"fill": [0.5, 1]},
        },
        "effect": {
            "args": [
                {"prob": 0, "args": {"alpha": [0, 0], "sigma": [0, 0]}},
                {"prob": 0, "args": {"scale": [0, 0], "per_channel": 0}},
                {"prob": 0, "args": {"k": [1, 1]}},
                {"prob": 0, "args": {"k": [1, 1]}},
                {"prob": 0, "args": {"p": [0, 0], "size_percent": [0.1, 0.1], "per_channel": 0}},
                {
                    "prob": 0,
                    "args": {"weights": [1], "args": [{"percents": [[1, 1], [1, 1], [1, 1], [1, 1]]}]},
                },
            ]
        },
    },
    "bg_effect": {
        "args": [
            {
                "prob": 0,
                "args": {"intensity": [0, 0], "amount": [0, 0], "smoothing": [1, 1], "bidirectional": 0},
            }
        ]
    },
    "doc_effect": {
        "args": [
            {
                "prob": 0,
                "args": {"intensity": [0, 0], "amount": [0, 0], "smoothing": [1, 1], "bidirectional": 0},
            }
        ]
    },
    "effect": {
        "args": [
            {"prob": 0, "args": {"rgb": [[128, 128], [128, 128], [128, 128]], "alpha": [0, 0]}},
            {"prob": 0, "args": {}},
            {"prob": 1, "args": {"alpha": [1, 1]}},
            {"prob": 1, "args": {"beta": [0, 0]}},
            {"prob": 0, "args": {"k": [3, 3], "angle": [0, 0]}},
            {"prob": 1, "args": {"sigma": [0, 0]}},
            {"prob": 0, "args": {"size": [1, 1]}},
            {"prob": 0, "args": {"compression": [10, 10]}},
        ]
    },
    "skew": {"prob": 0, "angle": [0, 0]},
}

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
_BLOCK_COL = (30, 100, 220)  # blue
_LINE_COL = (20, 170, 60)  # green
_WORD_COL = (210, 40, 40)  # red


def _px(norm: list[float], W: int, H: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = norm
    return int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)


def _quad_px(quad: list, W: int, H: int) -> list[tuple[int, int]]:
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
            # close the polygon explicitly for thin lines
            draw.line(pts + [pts[0]], fill=(*_LINE_COL, 220), width=2)

    # Words: quad with faint fill
    for wd in data["words"]:
        if wd.quad:
            pts = _quad_px(wd.quad, W, H)
            draw.polygon(pts, fill=(*_WORD_COL, 35), outline=(*_WORD_COL, 200))

    return Image.alpha_composite(img, overlay).convert("RGB")


def _separator(H: int, width: int = 4) -> Image.Image:
    sep = Image.new("RGB", (width, H), (80, 80, 80))
    return sep


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
        f"seed={seed:03d}   "
        f"blocks={n_blk}  lines={n_ln}  words={n_wd}   "
        f"sharpness={data['quality_metrics'].get('sharpness', '?')}"
    )
    draw.text((6, 4), txt, fill=(200, 200, 200))
    return strip


def _passes_quality_filters(data: dict, cfg: dict) -> tuple:
    """Return (passes, reason) applying the same gates as SynthDoG.save()."""
    qm = data.get("quality_metrics", {})
    contrast = qm.get("min_line_contrast_ratio")
    threshold = cfg.get("min_contrast_ratio", 1.0)
    if contrast is not None and contrast < threshold:
        return False, f"contrast {contrast:.3f} < {threshold}"
    words = qm.get("word_count", 0)
    if words < cfg.get("min_word_count", 1):
        return False, f"words {words} < {cfg['min_word_count']}"
    null_frac = qm.get("textbox_null_frac") or 0.0
    if null_frac > cfg.get("max_textbox_null_frac", 1.0):
        return False, f"null_frac {null_frac:.3f}"
    min_h = qm.get("min_line_height_px")
    if min_h is not None and min_h < cfg.get("min_line_height_px", 0.0):
        return False, f"min_line_height {min_h:.1f}"
    sharp = qm.get("sharpness")
    if sharp is not None and sharp < cfg.get("min_sharpness", 0.0):
        return False, f"sharpness {sharp:.1f}"
    intra = qm.get("max_intra_block_line_overlap")
    if intra is not None and intra > cfg.get("max_intra_block_line_overlap", 1.0):
        return False, f"intra_overlap {intra:.3f}"
    cross = qm.get("max_cross_block_line_overlap")
    if cross is not None and cross > cfg.get("max_cross_block_line_overlap", 1.0):
        return False, f"cross_overlap {cross:.3f}"
    return True, "ok"


def generate_viz(n: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dog = SynthDoG(_CFG)

    saved = 0
    discarded = 0
    seed = 0
    while saved < n:
        data = dog.generate(seed=seed)
        passes, _ = _passes_quality_filters(data, _CFG)
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
            print(f"  {saved}/{n}  (seed={seed - 1}, discarded={discarded}/{seed})")

    # HTML gallery
    html_path = out_dir / "index.html"
    imgs = sorted(out_dir.glob("sample_*.png"))
    rows = "\n".join(f'<div class="card"><img src="{p.name}" loading="lazy"><p>{p.stem}</p></div>' for p in imgs)
    html_path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>SynthDoG 100-sample annotation review</title>
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
<h1>SynthDoG annotation review — 100 samples (seeds 0–99)</h1>
<p class="legend">
  Each image: &nbsp;
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
    print(f"\nDone: {saved} saved, {discarded} discarded ({discarded / (seed or 1) * 100:.1f}% discard rate)")
    print(f"Gallery → {html_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=str, default="outputs/viz")
    args = ap.parse_args()
    generate_viz(args.n, Path(args.out))
