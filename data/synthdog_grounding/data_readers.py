"""Unified sample iterator for SynthDoG data sources.

Supports both raw generation output (metadata.jsonl directories) and
packaged tar archives, normalizing both to a common dict shape.
"""

from __future__ import annotations

import json
import sys
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from serialization import (
    KEY_QUALITY_METRICS,
    KEY_TEXT_BLOCKS,
    KEY_TEXT_LINES,
    KEY_TEXT_WORDS,
    decode_metadata,
)

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment,misc]


def extract_image_metadata(img_path: Path) -> tuple[int, int, float | None]:
    """Extract (width, height, dpi) from an image file.

    Reused by package.py — this is the canonical implementation.
    """
    if Image is None:
        raise ImportError("Pillow is required for image metadata extraction")
    with Image.open(img_path) as im:
        width, height = im.size
        info = getattr(im, "info", {}) or {}
        dpi = None

        if "dpi" in info:
            v = info["dpi"]
            if isinstance(v, tuple) and len(v) > 0:
                dpi = float(v[0])
            elif isinstance(v, (int, float)):
                dpi = float(v)
        elif "jfif_density" in info:
            density = info.get("jfif_density")
            unit = info.get("jfif_unit", 1)
            if isinstance(density, tuple) and len(density) > 0:
                x_density = float(density[0])
            elif isinstance(density, (int, float)):
                x_density = float(density)
            else:
                x_density = None
            if x_density is not None:
                if unit == 1:
                    dpi = x_density
                elif unit == 2:
                    dpi = x_density * 2.54

        return width, height, (round(dpi, 2) if dpi is not None else None)


def _normalize_sample(
    text_lines: list[Any],
    text_words: list[Any],
    text_blocks: list[Any],
    quality_metrics: dict[str, Any],
    image_path: str,
    image_width: int | None,
    image_height: int | None,
    image_dpi: float | None,
) -> dict[str, Any]:
    """Build the canonical normalized dict shape."""
    return {
        "text": {"lines": text_lines, "words": text_words, "blocks": text_blocks},
        "image": {
            "path": image_path,
            "width": image_width,
            "height": image_height,
            "dpi": image_dpi,
        },
        "quality_metrics": quality_metrics,
    }


def _iter_directory(directory: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Iterate samples from a raw generation output directory."""
    meta_path = directory / "metadata.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.jsonl found in {directory}")

    with meta_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warning] JSON decode error on line {line_no}: {e}", file=sys.stderr)
                continue

            file_name = rec.get("file_name")
            if not file_name:
                print(f"[warning] No 'file_name' in line {line_no}; skipping.", file=sys.stderr)
                continue

            gt_parse = decode_metadata(rec)
            text_lines = gt_parse.get(KEY_TEXT_LINES, [])
            text_words = gt_parse.get(KEY_TEXT_WORDS, [])
            text_blocks = gt_parse.get(KEY_TEXT_BLOCKS, [])
            quality_metrics = gt_parse.get(KEY_QUALITY_METRICS, {})

            # Extract image metadata if possible
            img_path = directory / file_name
            width, height, dpi = None, None, None
            if img_path.exists() and Image is not None:
                try:
                    width, height, dpi = extract_image_metadata(img_path)
                except Exception as e:
                    print(f"[warning] Failed reading image metadata for {img_path}: {e}", file=sys.stderr)

            sample_id = Path(file_name).stem
            yield (
                sample_id,
                _normalize_sample(text_lines, text_words, text_blocks, quality_metrics, file_name, width, height, dpi),
            )


def _iter_tar(tar_path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Iterate samples from a packaged tar archive."""
    with tarfile.open(tar_path, "r") as tar:
        json_members = sorted(
            (m for m in tar.getmembers() if m.name.endswith(".json") and m.isfile()),
            key=lambda m: m.name,
        )

        for member in json_members:
            f = tar.extractfile(member)
            if f is None:
                continue
            try:
                data = json.loads(f.read().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"[warning] Error reading {member.name}: {e}", file=sys.stderr)
                continue

            sample_id = member.name.replace(".json", "")

            # Handle old tar format gracefully
            text_data = data.get("text", {})
            normalized = _normalize_sample(
                text_lines=text_data.get("lines", []),
                text_words=text_data.get("words", []),
                text_blocks=text_data.get("blocks", []),
                quality_metrics=data.get("quality_metrics", {}),
                image_path=data.get("image", {}).get("path", ""),
                image_width=data.get("image", {}).get("width"),
                image_height=data.get("image", {}).get("height"),
                image_dpi=data.get("image", {}).get("dpi"),
            )
            yield sample_id, normalized


def iter_samples(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Auto-detect input format and yield (sample_id, normalized_dict) pairs.

    Supports:
    - Directory containing metadata.jsonl (raw generation output)
    - Tar file (.tar) (packaged archive)
    """
    path = Path(path)
    if path.is_dir():
        yield from _iter_directory(path)
    elif path.is_file() and path.suffix == ".tar":
        yield from _iter_tar(path)
    else:
        raise ValueError(f"Unsupported input: {path} (expected directory or .tar file)")
