"""Synthetic table image generator.

CLI entry point for generating synthetic table images with JSON sidecars.

Output layout:
    {output_dir}/
        00000000.jpg
        00000000.json
        00000001.jpg
        00000001.json
        ...

Each JSON sidecar follows the GutenOCR standard schema and includes an
OTSL-encoded table structure field compatible with roots-ocr TSR_ANNOTATION_SCHEMA.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import random
from pathlib import Path

from content_distribution import ContentDistribution
from otsl import Flavor, structure_to_otsl, validate_otsl
from PIL import Image
from table_renderer import augment_image, render_table
from table_structure import generate_table_structure
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _build_content_grid(
    structure,
    dist: ContentDistribution,
    rng: random.Random,
) -> list[list[str]]:
    """Fill a rows×cols grid with sampled cell text."""
    grid: list[list[str]] = []
    for r in range(structure.rows):
        row: list[str] = []
        for c in range(structure.cols):
            if structure.has_header and r == 0:
                row.append(dist.sample_header_content(rng))
            else:
                # ~15% empty cells in body
                if rng.random() < 0.15:
                    row.append("")
                else:
                    row.append(dist.sample_cell_content(rng, max_words=3))
        grid.append(row)
    return grid


def _structure_to_html(structure, content_grid: list[list[str]]) -> str:
    """Build a minimal HTML table string from structure + content."""
    ext: dict[tuple[int, int], object] = {}
    for span in structure.spans:
        for dr in range(span.rowspan):
            for dc in range(span.colspan):
                if dr == 0 and dc == 0:
                    continue
                ext[(span.row + dr, span.col + dc)] = True

    span_map = {(s.row, s.col): s for s in structure.spans}

    parts = ["<table>"]
    for r in range(structure.rows):
        parts.append("  <tr>")
        for c in range(structure.cols):
            if (r, c) in ext:
                continue
            span = span_map.get((r, c))
            attrs = ""
            if span:
                if span.colspan > 1:
                    attrs += f' colspan="{span.colspan}"'
                if span.rowspan > 1:
                    attrs += f' rowspan="{span.rowspan}"'
            text = content_grid[r][c] if r < len(content_grid) and c < len(content_grid[r]) else ""
            tag = "th" if structure.has_header and r == 0 else "td"
            parts.append(f"    <{tag}{attrs}>{text}</{tag}>")
        parts.append("  </tr>")
    parts.append("</table>")
    return "\n".join(parts)


def generate_sample(
    sample_id: int,
    rng: random.Random,
    dist: ContentDistribution,
    *,
    min_rows: int,
    max_rows: int,
    min_cols: int,
    max_cols: int,
    span_prob: float,
    augment: bool,
    augment_count: int,
    output_dir: Path,
    flavor: Flavor = "semantic",
) -> list[Path]:
    """Generate one table sample (and optional augmented variants).

    Args:
        sample_id: Zero-based sample index (used for filenames).
        rng: Seeded random instance.
        dist: Content distribution for cell text.
        min_rows: Minimum table rows.
        max_rows: Maximum table rows.
        min_cols: Minimum table cols.
        max_cols: Maximum table cols.
        span_prob: Probability of a cell starting a span.
        augment: Whether to produce augmented variants.
        augment_count: Number of augmented variants per base sample.
        output_dir: Directory to write files into.
        flavor: OTSL flavor — ``"base"`` (6 tokens) or ``"semantic"`` (9 tokens).

    Returns:
        List of written file paths (jpg + json pairs).
    """
    structure = generate_table_structure(
        rng,
        min_rows=min_rows,
        max_rows=max_rows,
        min_cols=min_cols,
        max_cols=max_cols,
        span_prob=span_prob,
    )
    content_grid = _build_content_grid(structure, dist, rng)
    img, word_boxes, line_boxes = render_table(structure, content_grid)

    otsl_str = structure_to_otsl(structure, content_grid, flavor=flavor)
    validate_otsl(otsl_str, rows=structure.rows, cols=structure.cols, flavor=flavor)
    html_str = _structure_to_html(structure, content_grid)

    def _sidecar(image: Image.Image, img_name: str) -> dict:
        return {
            "image": {
                "path": img_name,
                "width": image.width,
                "height": image.height,
            },
            "text": {
                "words": [{"text": w.text, "box": w.box} for w in word_boxes],
                "lines": [{"text": lb.text, "box": lb.box} for lb in line_boxes],
            },
            "table": {
                "otsl": otsl_str,
                "html": html_str,
                "rows": structure.rows,
                "cols": structure.cols,
            },
        }

    written: list[Path] = []

    def _write(image: Image.Image, idx: int) -> None:
        img_name = f"{idx:08d}.jpg"
        json_name = f"{idx:08d}.json"
        img_path = output_dir / img_name
        json_path = output_dir / json_name

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=92)
        img_path.write_bytes(buf.getvalue())

        json_path.write_text(json.dumps(_sidecar(image, img_name), ensure_ascii=False))
        written.extend([img_path, json_path])

    base_id = sample_id * (augment_count + 1) if augment else sample_id
    _write(img, base_id)

    if augment:
        for i in range(augment_count):
            aug_img = augment_image(img, rng)
            _write(aug_img, base_id + i + 1)

    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate synthetic table images with OTSL sidecars.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of base tables to generate.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=Path, default=Path("./out/raw"), help="Output directory.")
    parser.add_argument("--min-rows", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=12)
    parser.add_argument("--min-cols", type=int, default=2)
    parser.add_argument("--max-cols", type=int, default=8)
    parser.add_argument("--span-prob", type=float, default=0.2, help="Per-cell span probability.")
    parser.add_argument("--augment", action="store_true", help="Produce augmented variants.")
    parser.add_argument(
        "--augment-count",
        type=int,
        default=2,
        help="Augmented variants per base sample.",
    )
    parser.add_argument(
        "--distribution",
        type=Path,
        default=None,
        help="Path to UNLV distribution pickle. Defaults to bundled sample_words.json.",
    )
    parser.add_argument(
        "--flavor",
        choices=["base", "semantic"],
        default="semantic",
        help="OTSL flavor: 'base' (6 tokens) or 'semantic' (9 tokens, includes ched/rhed/srow).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    dist = ContentDistribution(args.distribution)

    logger.info("Generating %d samples → %s (flavor=%s)", args.num_samples, args.output_dir, args.flavor)

    for i in tqdm(range(args.num_samples), desc="Generating"):
        generate_sample(
            i,
            rng,
            dist,
            min_rows=args.min_rows,
            max_rows=args.max_rows,
            min_cols=args.min_cols,
            max_cols=args.max_cols,
            span_prob=args.span_prob,
            augment=args.augment,
            augment_count=args.augment_count,
            output_dir=args.output_dir,
            flavor=args.flavor,
        )

    total = sum(1 for _ in args.output_dir.glob("*.jpg"))
    logger.info("Done. %d images written to %s", total, args.output_dir)


if __name__ == "__main__":
    main()
