"""Shared schema, constants, and annotations for SynthDoG metadata.

Centralizes the contract between producer (template.py) and consumer
(package.py) so that key names, encoding/decoding logic, annotation
dataclasses, and shared constants live in one place.
"""

import json
from dataclasses import dataclass
from typing import Any

# ── Dataset splits ───────────────────────────────────────────────────
SPLITS = ["train", "validation", "test"]


# ── Annotation dataclasses ───────────────────────────────────────────
@dataclass
class WordAnnotation:
    text: str
    bbox: list[float]
    line_id: int
    word_id: int
    quad: list[list[float]] | None = None


@dataclass
class LineAnnotation:
    text: str
    bbox: list[float]
    block_id: int
    line_id: int
    quad: list[list[float]] | None = None


@dataclass
class BlockAnnotation:
    block_id: int
    bbox: list[float]
    line_ids: list[int]


# ── Canonical key names ──────────────────────────────────────────────
KEY_FILE_NAME = "file_name"
KEY_GROUND_TRUTH = "ground_truth"
KEY_GT_PARSE = "gt_parse"
KEY_TEXT_LINES = "text_lines"
KEY_TEXT_BLOCKS = "text_blocks"
KEY_TEXT_WORDS = "text_words"
KEY_QUALITY_METRICS = "quality_metrics"


# ── Annotation → dict helpers ─────────────────────────────────────────


def line_annotation_to_dict(ln: LineAnnotation) -> dict[str, Any]:
    """Convert a LineAnnotation to a JSON-serializable dict."""
    entry: dict[str, Any] = {"text": ln.text, "bbox": ln.bbox, "line_id": ln.line_id, "block_id": ln.block_id}
    if ln.quad is not None:
        entry["quad"] = ln.quad
    return entry


def word_annotation_to_dict(wd: WordAnnotation) -> dict[str, Any]:
    """Convert a WordAnnotation to a JSON-serializable dict."""
    entry: dict[str, Any] = {"text": wd.text, "bbox": wd.bbox, "word_id": wd.word_id, "line_id": wd.line_id}
    if wd.quad is not None:
        entry["quad"] = wd.quad
    return entry


def block_annotation_to_dict(blk: BlockAnnotation) -> dict[str, Any]:
    """Convert a BlockAnnotation to a JSON-serializable dict."""
    return {"block_id": blk.block_id, "bbox": blk.bbox, "line_ids": blk.line_ids}


def encode_metadata(
    image_filename: str,
    keys: list[str],
    values: list[Any],
) -> dict[str, str]:
    """Encode metadata into the HuggingFace-compatible JSONL format.

    This is the single authoritative encoder – every producer must use it
    so that consumers can rely on a stable schema.

    Args:
        image_filename: The image file name (e.g. ``"image_42.jpg"``).
        keys: Task names (e.g. ``["text_lines", "text_blocks", ...]``).
        values: Ground-truth data corresponding to each task name.

    Returns:
        A dict with ``"file_name"`` and ``"ground_truth"`` (JSON string).

    Raises:
        ValueError: If *keys* and *values* have different lengths.
    """
    if len(keys) != len(values):
        raise ValueError(f"Length does not match: keys({len(keys)}), values({len(values)})")

    gt_parse = {KEY_GT_PARSE: dict(zip(keys, values))}
    gt_parse_str = json.dumps(gt_parse, ensure_ascii=False)
    return {KEY_FILE_NAME: image_filename, KEY_GROUND_TRUTH: gt_parse_str}


def decode_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Decode a metadata JSONL record back into a structured dict.

    Handles the ``ground_truth`` field being either a raw dict or a
    JSON-encoded string (the format written by :func:`encode_metadata`).

    Args:
        record: A single parsed JSONL record.

    Returns:
        The ``gt_parse`` dict (e.g. ``{"text_lines": [...], ...}``),
        or an empty dict if the record is malformed.
    """
    gt = record.get(KEY_GROUND_TRUTH, {})
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except (json.JSONDecodeError, TypeError):
            return {}
    if not isinstance(gt, dict):
        return {}
    return gt.get(KEY_GT_PARSE, {})
