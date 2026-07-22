"""Tests for serialization.py — annotation dataclasses, encode/decode, to_dict helpers."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serialization import (
    BlockAnnotation,
    LineAnnotation,
    WordAnnotation,
    block_annotation_to_dict,
    decode_metadata,
    encode_metadata,
    line_annotation_to_dict,
    word_annotation_to_dict,
)

# ---------------------------------------------------------------------------
# encode_metadata
# ---------------------------------------------------------------------------


def test_encode_metadata_mismatched_lengths_raises():
    """encode_metadata must raise ValueError when keys and values have different lengths."""
    with pytest.raises(ValueError, match="[Ll]ength"):
        encode_metadata("img.jpg", ["k1", "k2"], ["v1"])


def test_encode_metadata_returns_file_name_and_ground_truth_keys():
    result = encode_metadata("image_0.jpg", ["text_lines"], [[{"text": "hello"}]])
    assert "file_name" in result
    assert "ground_truth" in result


def test_encode_metadata_file_name_preserved():
    result = encode_metadata("image_42.jpg", [], [])
    assert result["file_name"] == "image_42.jpg"


def test_encode_metadata_ground_truth_is_json_string():
    result = encode_metadata("img.jpg", ["k"], ["v"])
    assert isinstance(result["ground_truth"], str)
    # Must be valid JSON.
    parsed = json.loads(result["ground_truth"])
    assert isinstance(parsed, dict)


def test_encode_metadata_unicode_preserved():
    """Non-ASCII text must survive the JSON round-trip (ensure_ascii=False)."""
    result = encode_metadata("img.jpg", ["label"], ["héllo wörld"])
    parsed = json.loads(result["ground_truth"])
    assert parsed["gt_parse"]["label"] == "héllo wörld"


# ---------------------------------------------------------------------------
# decode_metadata
# ---------------------------------------------------------------------------


def test_decode_metadata_roundtrip():
    """encode → decode must reconstruct the original gt_parse content."""
    keys = ["text_lines", "quality_metrics"]
    values = [[{"text": "foo", "bbox": [0, 0, 1, 1]}], {"sharpness": 42.0}]
    record = encode_metadata("img.jpg", keys, values)
    parsed = decode_metadata(record)
    assert parsed["text_lines"] == values[0]
    assert parsed["quality_metrics"] == values[1]


def test_decode_metadata_handles_malformed_json_gracefully():
    """Malformed ground_truth JSON must return an empty dict, not raise."""
    record = {"ground_truth": "not valid json{{{{"}
    result = decode_metadata(record)
    assert result == {}


def test_decode_metadata_missing_gt_parse_key_returns_empty():
    """If ground_truth is valid JSON but lacks 'gt_parse', return empty dict."""
    record = {"ground_truth": json.dumps({"other_key": "value"})}
    result = decode_metadata(record)
    assert result == {}


def test_decode_metadata_handles_dict_ground_truth():
    """ground_truth may already be a dict (not a JSON string); decode must handle both."""
    record = {"ground_truth": {"gt_parse": {"label": "hi"}}}
    result = decode_metadata(record)
    assert result["label"] == "hi"


def test_decode_metadata_missing_ground_truth_key_returns_empty():
    result = decode_metadata({})
    assert result == {}


# ---------------------------------------------------------------------------
# line_annotation_to_dict
# ---------------------------------------------------------------------------


def test_line_annotation_to_dict_has_required_keys():
    ln = LineAnnotation(text="hello", bbox=[0.0, 0.1, 0.5, 0.2], block_id=0, line_id=0)
    d = line_annotation_to_dict(ln)
    assert "text" in d
    assert "bbox" in d
    assert "line_id" in d
    assert "block_id" in d


def test_line_annotation_to_dict_no_quad_key_when_none():
    ln = LineAnnotation(text="hello", bbox=[0.0, 0.1, 0.5, 0.2], block_id=0, line_id=0, quad=None)
    d = line_annotation_to_dict(ln)
    assert "quad" not in d


def test_line_annotation_to_dict_includes_quad_when_set():
    quad = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.1], [0.0, 0.1]]
    ln = LineAnnotation(text="hi", bbox=[0.0, 0.0, 0.5, 0.1], block_id=0, line_id=0, quad=quad)
    d = line_annotation_to_dict(ln)
    assert d["quad"] == quad


# ---------------------------------------------------------------------------
# word_annotation_to_dict
# ---------------------------------------------------------------------------


def test_word_annotation_to_dict_has_required_keys():
    wd = WordAnnotation(text="foo", bbox=[0.0, 0.1, 0.3, 0.2], line_id=0, word_id=0)
    d = word_annotation_to_dict(wd)
    assert "text" in d
    assert "bbox" in d
    assert "word_id" in d
    assert "line_id" in d


def test_word_annotation_to_dict_no_quad_when_none():
    wd = WordAnnotation(text="foo", bbox=[0.0, 0.1, 0.3, 0.2], line_id=0, word_id=0, quad=None)
    assert "quad" not in word_annotation_to_dict(wd)


# ---------------------------------------------------------------------------
# block_annotation_to_dict
# ---------------------------------------------------------------------------


def test_block_annotation_to_dict_has_required_keys():
    blk = BlockAnnotation(block_id=0, bbox=[0.0, 0.0, 1.0, 1.0], line_ids=[0, 1])
    d = block_annotation_to_dict(blk)
    assert "block_id" in d
    assert "bbox" in d
    assert "line_ids" in d
    assert "region_type" in d


def test_block_annotation_to_dict_default_region_type():
    blk = BlockAnnotation(block_id=0, bbox=[0.0, 0.0, 1.0, 1.0], line_ids=[0])
    assert block_annotation_to_dict(blk)["region_type"] == "body"
