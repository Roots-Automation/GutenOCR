"""Unit tests for table_structure.py."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from table_structure import BorderStyle, generate_table_structure


def _rng(seed: int = 0) -> random.Random:
    return random.Random(seed)


class TestGenerateTableStructure:
    def test_dimensions_within_bounds(self):
        rng = _rng(1)
        for _ in range(50):
            s = generate_table_structure(rng, min_rows=2, max_rows=6, min_cols=2, max_cols=5)
            assert 2 <= s.rows <= 6
            assert 2 <= s.cols <= 5

    def test_no_span_overlap(self):
        rng = _rng(42)
        for _ in range(100):
            s = generate_table_structure(rng, span_prob=0.5)
            occupied: set[tuple[int, int]] = set()
            for span in s.spans:
                for dr in range(span.rowspan):
                    for dc in range(span.colspan):
                        if dr == 0 and dc == 0:
                            continue
                        pos = (span.row + dr, span.col + dc)
                        assert pos not in occupied, f"Overlap at {pos}"
                        occupied.add(pos)

    def test_spans_within_bounds(self):
        rng = _rng(7)
        for _ in range(50):
            s = generate_table_structure(rng, span_prob=0.4)
            for span in s.spans:
                assert span.row + span.rowspan <= s.rows
                assert span.col + span.colspan <= s.cols

    def test_no_span_prob_zero(self):
        rng = _rng(0)
        s = generate_table_structure(rng, span_prob=0.0)
        assert s.spans == []

    def test_header_flag(self):
        rng = _rng(3)
        results = [generate_table_structure(rng, header_prob=1.0).has_header for _ in range(10)]
        assert all(results)

        rng = _rng(3)
        results = [generate_table_structure(rng, header_prob=0.0).has_header for _ in range(10)]
        assert not any(results)

    def test_border_style_is_valid(self):
        rng = _rng(5)
        for _ in range(20):
            s = generate_table_structure(rng)
            assert s.border_style in list(BorderStyle)

    def test_reproducible_with_same_seed(self):
        s1 = generate_table_structure(_rng(99))
        s2 = generate_table_structure(_rng(99))
        assert s1.rows == s2.rows
        assert s1.cols == s2.cols
        assert s1.spans == s2.spans
        assert s1.has_header == s2.has_header
