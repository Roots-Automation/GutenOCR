"""Tests for extraction.py — no lualatex required (unit tests only)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extraction import (
    _assign_column,
    _build_page_annotations,
    _looks_like_section_heading,
    _norm,
)


def _mock_word(x0: float, top: float, x1: float, bottom: float, text: str) -> dict:
    return {"x0": x0, "top": top, "x1": x1, "bottom": bottom, "text": text}


class TestNorm:
    def test_midpoint(self):
        assert _norm(50.0, 100) == 0.5

    def test_clamp_high(self):
        assert _norm(200.0, 100) == 1.0

    def test_clamp_low(self):
        assert _norm(-5.0, 100) == 0.0

    def test_rounding(self):
        # Should be rounded to 3 decimal places
        result = _norm(1.0, 3)
        assert result == round(1 / 3, 3)


class TestAssignColumn:
    def test_single_col_always_zero(self):
        assert _assign_column(x_center=300.0, page_w_pts=600.0, n_cols=1) == 0
        assert _assign_column(x_center=10.0, page_w_pts=600.0, n_cols=1) == 0

    def test_two_col_left(self):
        assert _assign_column(x_center=100.0, page_w_pts=600.0, n_cols=2) == 0

    def test_two_col_right(self):
        assert _assign_column(x_center=400.0, page_w_pts=600.0, n_cols=2) == 1

    def test_two_col_boundary(self):
        # Exactly at midpoint → left column (< not <=)
        assert _assign_column(x_center=300.0, page_w_pts=600.0, n_cols=2) == 1


class TestLineGroupingSingleCol:
    def test_two_words_same_line(self):
        words = [
            _mock_word(10, 100, 50, 115, "Hello"),
            _mock_word(55, 100, 100, 115, "World"),
        ]
        result = _build_page_annotations(
            raw_words=words,
            img_w=612,
            img_h=792,
            scale=1.0,
            page_w_pts=612.0,
            page_h_pts=792.0,
            n_cols=1,
        )
        assert result.line_count == 1
        assert result.lines[0].text == "Hello World"

    def test_two_words_different_lines(self):
        words = [
            _mock_word(10, 100, 50, 115, "First"),
            _mock_word(10, 140, 60, 155, "Second"),
        ]
        result = _build_page_annotations(
            raw_words=words,
            img_w=612,
            img_h=792,
            scale=1.0,
            page_w_pts=612.0,
            page_h_pts=792.0,
            n_cols=1,
        )
        assert result.line_count == 2

    def test_word_count(self):
        words = [_mock_word(i * 20, 100, i * 20 + 15, 115, f"w{i}") for i in range(5)]
        result = _build_page_annotations(
            raw_words=words,
            img_w=612,
            img_h=792,
            scale=1.0,
            page_w_pts=612.0,
            page_h_pts=792.0,
            n_cols=1,
        )
        assert result.word_count == 5

    def test_bbox_normalized(self):
        words = [_mock_word(0, 0, 612, 792, "Full")]
        result = _build_page_annotations(
            raw_words=words,
            img_w=612,
            img_h=792,
            scale=1.0,
            page_w_pts=612.0,
            page_h_pts=792.0,
            n_cols=1,
        )
        assert result.words[0].bbox == [0.0, 0.0, 1.0, 1.0]


class TestLineGroupingTwoCol:
    def test_same_y_different_columns(self):
        # Left column word and right column word at same y-level must be separate lines
        words = [
            _mock_word(10, 200, 100, 215, "LeftWord"),
            _mock_word(350, 200, 440, 215, "RightWord"),
        ]
        result = _build_page_annotations(
            raw_words=words,
            img_w=612,
            img_h=792,
            scale=1.0,
            page_w_pts=612.0,
            page_h_pts=792.0,
            n_cols=2,
        )
        assert result.line_count == 2
        texts = {ln.text for ln in result.lines}
        assert "LeftWord" in texts
        assert "RightWord" in texts


class TestRegionTypeFromHeadings:
    def test_abstract_region_type(self):
        # "Abstract" heading followed by body text → abstract region_type on body
        words = [
            _mock_word(50, 80, 150, 100, "Abstract"),
            _mock_word(50, 110, 300, 125, "Lorem"),
            _mock_word(305, 110, 500, 125, "ipsum"),
            _mock_word(50, 200, 300, 215, "Introduction"),
            _mock_word(50, 230, 400, 245, "Body"),
            _mock_word(405, 230, 600, 245, "text"),
        ]
        result = _build_page_annotations(
            raw_words=words,
            img_w=612,
            img_h=792,
            scale=1.0,
            page_w_pts=612.0,
            page_h_pts=792.0,
            n_cols=1,
        )
        region_types = {blk.region_type for blk in result.blocks}
        assert "abstract" in region_types

    def test_reference_region_type(self):
        words = [
            _mock_word(50, 600, 200, 620, "References"),
            _mock_word(50, 635, 400, 650, "Smith"),
            _mock_word(405, 635, 600, 650, "et"),
        ]
        result = _build_page_annotations(
            raw_words=words,
            img_w=612,
            img_h=792,
            scale=1.0,
            page_w_pts=612.0,
            page_h_pts=792.0,
            n_cols=1,
        )
        region_types = {blk.region_type for blk in result.blocks}
        assert "reference" in region_types


class TestLooksLikeSectionHeading:
    def test_title_case_short(self):
        assert _looks_like_section_heading("Introduction")
        assert _looks_like_section_heading("Related Work")

    def test_lowercase_not_heading(self):
        assert not _looks_like_section_heading("this is a long sentence with many words that are lowercase")

    def test_too_long_not_heading(self):
        long_text = "Word " * 10
        assert not _looks_like_section_heading(long_text.strip())
