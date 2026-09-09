"""Shared schema, constants, and annotations for latex_academic metadata.

Schema is intentionally aligned with synthdog_grounding/serialization.py so
that downstream HuggingFace dataset loaders work without changes.
"""

import json
from dataclasses import dataclass
from typing import Any

SPLITS = ["train", "validation", "test"]

KEY_FILE_NAME = "file_name"
KEY_GROUND_TRUTH = "ground_truth"
KEY_GT_PARSE = "gt_parse"
KEY_TEXT_LINES = "text_lines"
KEY_TEXT_BLOCKS = "text_blocks"
KEY_TEXT_WORDS = "text_words"
KEY_QUALITY_METRICS = "quality_metrics"
KEY_GENERATION_PARAMS = "generation_params"


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
    font_family: str | None = None
    font_size_px: int | None = None
    text_color_rgb: list[int] | None = None


@dataclass
class BlockAnnotation:
    text: str
    block_id: int
    bbox: list[float]
    line_ids: list[int]
    # Extends synthdog region_type with "abstract", "caption", "reference"
    region_type: str = "body"
    quad: list[list[float]] | None = None


def line_annotation_to_dict(ln: LineAnnotation) -> dict[str, Any]:
    entry: dict[str, Any] = {"text": ln.text, "bbox": ln.bbox, "line_id": ln.line_id, "block_id": ln.block_id}
    if ln.quad is not None:
        entry["quad"] = ln.quad
    if ln.font_family is not None:
        entry["font_family"] = ln.font_family
    if ln.font_size_px is not None:
        entry["font_size_px"] = ln.font_size_px
    if ln.text_color_rgb is not None:
        entry["text_color_rgb"] = ln.text_color_rgb
    return entry


def word_annotation_to_dict(wd: WordAnnotation) -> dict[str, Any]:
    entry: dict[str, Any] = {"text": wd.text, "bbox": wd.bbox, "word_id": wd.word_id, "line_id": wd.line_id}
    if wd.quad is not None:
        entry["quad"] = wd.quad
    return entry


def block_annotation_to_dict(blk: BlockAnnotation) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "text": blk.text,
        "block_id": blk.block_id,
        "bbox": blk.bbox,
        "line_ids": blk.line_ids,
        "region_type": blk.region_type,
    }
    if blk.quad is not None:
        entry["quad"] = blk.quad
    return entry


def encode_metadata(
    image_filename: str,
    keys: list[str],
    values: list[Any],
) -> dict[str, str]:
    """Encode metadata into the HuggingFace-compatible JSONL format.

    Identical contract to synthdog_grounding/serialization.py:encode_metadata.
    """
    if len(keys) != len(values):
        raise ValueError(f"Length does not match: keys({len(keys)}), values({len(values)})")
    gt_parse = {KEY_GT_PARSE: dict(zip(keys, values))}
    gt_parse_str = json.dumps(gt_parse, ensure_ascii=False)
    return {KEY_FILE_NAME: image_filename, KEY_GROUND_TRUTH: gt_parse_str}


def decode_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Decode a metadata JSONL record back into a structured dict."""
    gt = record.get(KEY_GROUND_TRUTH, {})
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except (json.JSONDecodeError, TypeError):
            return {}
    if not isinstance(gt, dict):
        return {}
    return gt.get(KEY_GT_PARSE, {})
