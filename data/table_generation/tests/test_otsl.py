"""Unit tests for otsl.py."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from otsl import NL, OTSL_VOCAB, XCEL, YCEL, structure_to_otsl, validate_otsl
from table_structure import Span, TableStructure, generate_table_structure


def _simple_grid(rows: int, cols: int, fill: str = "text") -> list[list[str]]:
    return [[fill] * cols for _ in range(rows)]


class TestStructureToOtsl:
    def test_simple_2x2_all_filled(self):
        s = TableStructure(rows=2, cols=2, spans=[])
        grid = _simple_grid(2, 2)
        otsl = structure_to_otsl(s, grid)
        assert otsl == "FCEL FCEL NL FCEL FCEL NL"

    def test_simple_2x2_all_empty(self):
        s = TableStructure(rows=2, cols=2, spans=[])
        grid = _simple_grid(2, 2, fill="")
        otsl = structure_to_otsl(s, grid)
        assert otsl == "ECEL ECEL NL ECEL ECEL NL"

    def test_colspan_produces_xcel(self):
        # 1×2 table with a single colspan=2 cell
        s = TableStructure(rows=1, cols=2, spans=[Span(row=0, col=0, rowspan=1, colspan=2)])
        grid = [["Header", ""]]
        otsl = structure_to_otsl(s, grid)
        assert otsl == "FCEL XCEL NL"

    def test_rowspan_produces_ycel(self):
        # 2×1 table with a single rowspan=2 cell
        s = TableStructure(rows=2, cols=1, spans=[Span(row=0, col=0, rowspan=2, colspan=1)])
        grid = [["A"], [""]]
        otsl = structure_to_otsl(s, grid)
        assert otsl == "FCEL NL YCEL NL"

    def test_2d_span_produces_xcel_and_ycel(self):
        # 2×2 with single 2×2 span
        s = TableStructure(rows=2, cols=2, spans=[Span(row=0, col=0, rowspan=2, colspan=2)])
        grid = [["A", ""], ["", ""]]
        otsl = structure_to_otsl(s, grid)
        tokens = otsl.split()
        assert tokens.count(XCEL) == 1
        assert tokens.count(YCEL) == 2  # (1,0) and (1,1) — wait, (1,1) is also YCEL
        assert tokens.count(NL) == 2

    def test_all_tokens_in_vocab(self):
        rng = random.Random(0)
        for _ in range(20):
            s = generate_table_structure(rng, span_prob=0.3)
            grid = _simple_grid(s.rows, s.cols)
            otsl = structure_to_otsl(s, grid)
            for t in otsl.split():
                assert t in OTSL_VOCAB

    def test_ends_with_nl(self):
        rng = random.Random(1)
        for _ in range(20):
            s = generate_table_structure(rng)
            otsl = structure_to_otsl(s, _simple_grid(s.rows, s.cols))
            assert otsl.endswith(NL)

    def test_nl_count_equals_rows(self):
        rng = random.Random(2)
        for _ in range(20):
            s = generate_table_structure(rng)
            otsl = structure_to_otsl(s, _simple_grid(s.rows, s.cols))
            assert otsl.split().count(NL) == s.rows


class TestValidateOtsl:
    def test_valid_sequence_passes(self):
        validate_otsl("FCEL FCEL NL ECEL FCEL NL", rows=2, cols=2)

    def test_wrong_row_count_raises(self):
        with pytest.raises(ValueError, match="rows"):
            validate_otsl("FCEL NL", rows=2, cols=1)

    def test_wrong_col_count_raises(self):
        with pytest.raises(ValueError, match="tokens"):
            validate_otsl("FCEL FCEL NL FCEL NL", rows=2, cols=2)

    def test_unknown_token_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            validate_otsl("FCEL BADTOKEN NL", rows=1, cols=2)

    def test_missing_trailing_nl_raises(self):
        with pytest.raises(ValueError, match="NL"):
            validate_otsl("FCEL FCEL", rows=1, cols=2)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            validate_otsl("", rows=0, cols=0)
