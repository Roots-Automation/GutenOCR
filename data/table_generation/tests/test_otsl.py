"""Unit tests for otsl.py."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from otsl import (
    CHED,
    FCEL,
    LCEL,
    NL,
    OTSL_VOCAB_BASE,
    OTSL_VOCAB_SEMANTIC,
    RHED,
    SROW,
    UCEL,
    XCEL,
    structure_to_otsl,
    validate_otsl,
)
from table_structure import Span, TableStructure, generate_table_structure


def _simple_grid(rows: int, cols: int, fill: str = "text") -> list[list[str]]:
    return [[fill] * cols for _ in range(rows)]


class TestStructureToOtslBase:
    """Tests for flavor='base' (6-token vocab)."""

    def test_simple_2x2_all_filled(self):
        s = TableStructure(rows=2, cols=2, spans=[])
        otsl = structure_to_otsl(s, _simple_grid(2, 2), flavor="base")
        assert otsl == "fcel fcel nl fcel fcel nl"

    def test_simple_2x2_all_empty(self):
        s = TableStructure(rows=2, cols=2, spans=[])
        otsl = structure_to_otsl(s, _simple_grid(2, 2, fill=""), flavor="base")
        assert otsl == "ecel ecel nl ecel ecel nl"

    def test_colspan_produces_lcel(self):
        s = TableStructure(rows=1, cols=2, spans=[Span(row=0, col=0, rowspan=1, colspan=2)])
        otsl = structure_to_otsl(s, [["Header", ""]], flavor="base")
        assert otsl == "fcel lcel nl"

    def test_rowspan_produces_ucel(self):
        s = TableStructure(rows=2, cols=1, spans=[Span(row=0, col=0, rowspan=2, colspan=1)])
        otsl = structure_to_otsl(s, [["A"], [""]], flavor="base")
        assert otsl == "fcel nl ucel nl"

    def test_2d_span_produces_lcel_ucel_xcel(self):
        s = TableStructure(rows=2, cols=2, spans=[Span(row=0, col=0, rowspan=2, colspan=2)])
        otsl = structure_to_otsl(s, [["A", ""], ["", ""]], flavor="base")
        tokens = otsl.split()
        assert tokens.count(LCEL) == 1
        assert tokens.count(UCEL) == 1
        assert tokens.count(XCEL) == 1
        assert tokens.count(NL) == 2

    def test_header_row_emits_fcel_not_ched(self):
        s = TableStructure(rows=2, cols=2, has_header=True)
        otsl = structure_to_otsl(s, _simple_grid(2, 2), flavor="base")
        assert CHED not in otsl.split()

    def test_all_tokens_in_base_vocab(self):
        rng = random.Random(0)
        for _ in range(20):
            s = generate_table_structure(rng, span_prob=0.3)
            otsl = structure_to_otsl(s, _simple_grid(s.rows, s.cols), flavor="base")
            for t in otsl.split():
                assert t in OTSL_VOCAB_BASE

    def test_ends_with_nl(self):
        rng = random.Random(1)
        for _ in range(20):
            s = generate_table_structure(rng)
            assert structure_to_otsl(s, _simple_grid(s.rows, s.cols), flavor="base").endswith(NL)

    def test_nl_count_equals_rows(self):
        rng = random.Random(2)
        for _ in range(20):
            s = generate_table_structure(rng)
            otsl = structure_to_otsl(s, _simple_grid(s.rows, s.cols), flavor="base")
            assert otsl.split().count(NL) == s.rows


class TestStructureToOtslSemantic:
    """Tests for flavor='semantic' (9-token vocab)."""

    def test_header_row_emits_ched(self):
        s = TableStructure(rows=2, cols=3, has_header=True)
        otsl = structure_to_otsl(s, _simple_grid(2, 3), flavor="semantic")
        tokens = otsl.split()
        # First row: all ched
        assert tokens[:3] == [CHED, CHED, CHED]
        # Second row: fcel (non-header body)
        assert FCEL in tokens[4:]

    def test_row_header_col_emits_rhed(self):
        s = TableStructure(rows=3, cols=2, has_header=False, has_row_header=True)
        otsl = structure_to_otsl(s, _simple_grid(3, 2), flavor="semantic")
        tokens = otsl.split()
        # col 0 of each row should be rhed
        assert tokens[0] == RHED  # row 0, col 0
        assert tokens[3] == RHED  # row 1, col 0
        assert tokens[6] == RHED  # row 2, col 0

    def test_section_row_emits_srow(self):
        s = TableStructure(rows=3, cols=2, has_header=True, section_rows=[2])
        otsl = structure_to_otsl(s, _simple_grid(3, 2), flavor="semantic")
        tokens = otsl.split()
        # row 2 (last row): srow srow nl
        assert tokens[-3:] == [SROW, SROW, NL]

    def test_srow_takes_priority_over_rhed(self):
        s = TableStructure(rows=3, cols=2, has_header=False, has_row_header=True, section_rows=[1])
        otsl = structure_to_otsl(s, _simple_grid(3, 2), flavor="semantic")
        tokens = otsl.split()
        # row 1 is a section row → all srow, not rhed
        assert tokens[3] == SROW
        assert tokens[4] == SROW

    def test_ched_takes_priority_over_rhed(self):
        # header row + row header col: header row wins → ched
        s = TableStructure(rows=2, cols=2, has_header=True, has_row_header=True)
        otsl = structure_to_otsl(s, _simple_grid(2, 2), flavor="semantic")
        tokens = otsl.split()
        assert tokens[0] == CHED  # (0,0): header row wins over row-header col

    def test_all_tokens_in_semantic_vocab(self):
        rng = random.Random(3)
        for _ in range(30):
            s = generate_table_structure(rng, span_prob=0.3)
            otsl = structure_to_otsl(s, _simple_grid(s.rows, s.cols), flavor="semantic")
            for t in otsl.split():
                assert t in OTSL_VOCAB_SEMANTIC

    def test_nl_count_equals_rows(self):
        rng = random.Random(4)
        for _ in range(20):
            s = generate_table_structure(rng)
            otsl = structure_to_otsl(s, _simple_grid(s.rows, s.cols), flavor="semantic")
            assert otsl.split().count(NL) == s.rows


class TestValidateOtsl:
    def test_valid_base_sequence_passes(self):
        validate_otsl("fcel fcel nl ecel fcel nl", rows=2, cols=2, flavor="base")

    def test_valid_semantic_sequence_passes(self):
        validate_otsl("ched ched nl rhed fcel nl srow srow nl", rows=3, cols=2, flavor="semantic")

    def test_semantic_token_rejected_by_base_flavor(self):
        with pytest.raises(ValueError, match="Unknown"):
            validate_otsl("ched fcel nl", rows=1, cols=2, flavor="base")

    def test_wrong_row_count_raises(self):
        with pytest.raises(ValueError, match="rows"):
            validate_otsl("fcel nl", rows=2, cols=1)

    def test_wrong_col_count_raises(self):
        with pytest.raises(ValueError, match="tokens"):
            validate_otsl("fcel fcel nl fcel nl", rows=2, cols=2)

    def test_unknown_token_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            validate_otsl("fcel BADTOKEN nl", rows=1, cols=2)

    def test_missing_trailing_nl_raises(self):
        with pytest.raises(ValueError, match="NL"):
            validate_otsl("fcel fcel", rows=1, cols=2)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            validate_otsl("", rows=0, cols=0)
