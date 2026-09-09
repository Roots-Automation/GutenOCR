"""Tests for content.py — no lualatex or pdfplumber required."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from content import (
    _EQUATIONS_DISPLAY,
    _EQUATIONS_INLINE,
    AlgorithmContent,
    FigureContent,
    ListContent,
    SyntheticContent,
    TheoremContent,
    _make_algorithm,
    _make_figure,
    _make_list,
    _make_theorem,
    escape_latex,
)
from layout import LayoutConfig


def _rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


def _layout(rng: random.Random | None = None) -> LayoutConfig:
    return LayoutConfig.sample(rng or _rng())


class TestEscapeLatex:
    def test_backslash_becomes_textbackslash(self):
        assert escape_latex("\\") == r"\textbackslash{}"

    def test_backslash_does_not_double_escape(self):
        result = escape_latex("\\")
        assert result == r"\textbackslash{}"
        assert r"\textbackslash\{\}" not in result

    def test_mixed_backslash_and_ampersand(self):
        result = escape_latex("a\\b&c")
        assert r"\textbackslash{}" in result
        assert r"\&" in result

    def test_all_specials_produce_escape_sequences(self):
        mapping = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "^": r"\^{}",
            "~": r"\textasciitilde{}",
        }
        for char, expected in mapping.items():
            assert escape_latex(char) == expected, f"Wrong escape for {char!r}"

    def test_ampersand(self):
        assert escape_latex("a & b") == r"a \& b"

    def test_percent(self):
        assert escape_latex("100%") == r"100\%"

    def test_dollar(self):
        assert escape_latex("$5") == r"\$5"

    def test_underscore(self):
        assert escape_latex("some_var") == r"some\_var"

    def test_plain_text_unchanged(self):
        text = "Hello World 123"
        assert escape_latex(text) == text


class TestSyntheticContent:
    def test_generates_without_error(self):
        layout = _layout()
        content = SyntheticContent.generate(layout, _rng())
        assert content.title
        assert content.authors
        assert content.sections

    def test_section_count_matches_layout(self):
        rng = _rng(7)
        layout = LayoutConfig.sample(rng)
        content = SyntheticContent.generate(layout, _rng(99))
        assert len(content.sections) == layout.n_sections

    def test_full_text_is_non_empty(self):
        content = SyntheticContent.generate(_layout(), _rng())
        assert len(content.full_text) > 50

    def test_template_context_has_required_keys(self):
        content = SyntheticContent.generate(_layout(), _rng())
        ctx = content.to_template_context()
        for key in ("title", "authors", "sections", "n_cols", "font_package", "font_size_pt"):
            assert key in ctx, f"Missing key: {key}"

    def test_section_context_has_new_keys(self):
        content = SyntheticContent.generate(_layout(), _rng())
        ctx = content.to_template_context()
        sec = ctx["sections"][0]
        for key in ("align_block", "table_caption", "list_content", "theorem", "algorithm", "figure"):
            assert key in sec, f"Missing section key: {key}"

    def test_equations_come_from_wordlist(self):
        all_eq = set(_EQUATIONS_INLINE) | set(_EQUATIONS_DISPLAY)
        rng = _rng(5)
        layout = LayoutConfig.sample(rng, equations_per_section_range=(1, 3))
        content = SyntheticContent.generate(layout, _rng(5))
        for sec in content.sections:
            if sec.display_equation is not None:
                assert sec.display_equation in all_eq, f"Unknown equation: {sec.display_equation}"

    def test_references_only_when_requested(self):
        rng = _rng(1)
        layout = LayoutConfig.sample(rng, references_prob=0.0)
        content = SyntheticContent.generate(layout, _rng(1))
        assert content.references == []

    def test_abstract_only_when_requested(self):
        rng = _rng(2)
        layout = LayoutConfig.sample(rng, abstract_prob=0.0)
        content = SyntheticContent.generate(layout, _rng(2))
        assert content.abstract == ""

    def test_at_most_one_math_block_per_section(self):
        rng = _rng(3)
        layout = LayoutConfig.sample(rng)
        content = SyntheticContent.generate(layout, rng)
        for sec in content.sections:
            assert not (sec.display_equation and sec.align_block), "Section has both display_equation and align_block"


class TestListContent:
    def test_make_list_returns_correct_type(self):
        rng = _rng()
        result = _make_list(rng, ordered=False, ref_keys=[])
        assert isinstance(result, ListContent)
        assert isinstance(result.items, list)
        assert 3 <= len(result.items) <= 5
        assert result.ordered is False

    def test_make_list_ordered(self):
        rng = _rng(10)
        result = _make_list(rng, ordered=True, ref_keys=["ref0", "ref1"])
        assert result.ordered is True

    def test_list_items_are_strings(self):
        rng = _rng()
        result = _make_list(rng, ordered=False, ref_keys=[])
        assert all(isinstance(item, str) and len(item) > 0 for item in result.items)


class TestTheoremContent:
    def test_make_theorem_returns_correct_type(self):
        rng = _rng()
        result = _make_theorem(rng)
        assert isinstance(result, TheoremContent)
        assert result.env_type in ("theorem", "lemma", "definition", "corollary", "proposition", "remark")
        assert len(result.body) > 0

    def test_theorem_body_is_non_empty(self):
        rng = _rng(7)
        result = _make_theorem(rng)
        assert result.body.strip()


class TestAlgorithmContent:
    def test_make_algorithm_returns_correct_type(self):
        rng = _rng()
        result = _make_algorithm(rng, idx=0)
        assert isinstance(result, AlgorithmContent)
        assert result.idx == 0
        assert len(result.caption) > 0
        assert len(result.lines) >= 4

    def test_algorithm_lines_are_strings(self):
        rng = _rng(5)
        result = _make_algorithm(rng, idx=2)
        assert all(isinstance(ln, str) for ln in result.lines)

    def test_algorithm_idx_stored(self):
        rng = _rng()
        for idx in [0, 1, 5]:
            result = _make_algorithm(rng, idx=idx)
            assert result.idx == idx


class TestFigureContent:
    def test_make_figure_returns_correct_type(self):
        rng = _rng()
        result = _make_figure(rng, idx=0)
        assert isinstance(result, FigureContent)
        assert result.idx == 0
        assert 2.0 <= result.height_cm <= 6.0
        assert len(result.caption) > 0

    def test_figure_caption_contains_figure_number(self):
        rng = _rng()
        result = _make_figure(rng, idx=3)
        assert "Figure 4" in result.caption
