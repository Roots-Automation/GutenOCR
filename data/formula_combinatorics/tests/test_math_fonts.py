"""Tests for the math_fonts domain."""

from __future__ import annotations

import random

import pytest
from formula_combinatorics._template_dsl import sample
from formula_combinatorics.domains import GENERATORS, TEMPLATES

_GEN = GENERATORS["math_fonts"]
_TMPLS = TEMPLATES["math_fonts"]
_TMPL_BY_NAME = {t.name: t for t in _TMPLS}


def _draw(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return [_GEN(rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_smoke_generates_nonempty():
    results = _draw(500)
    assert all(isinstance(s, str) and len(s) > 0 for s in results)


# ---------------------------------------------------------------------------
# Font command coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "font_cmd",
    [r"\mathsf", r"\mathtt", r"\mathit", r"\mathnormal"],
)
def test_font_command_appears(font_cmd: str):
    results = _draw(2000)
    assert any(font_cmd in s for s in results), f"{font_cmd} never appeared in 2000 samples"


# ---------------------------------------------------------------------------
# Full-expression wrap structural patterns
# ---------------------------------------------------------------------------


def test_fontbf_linear_relation_is_multitoken():
    r"""The \mathbf wrap must contain a space (multi-token content)."""
    results = _draw(2000)
    assert any(r"\mathbf{" in s and " " in s[s.index(r"\mathbf{") :] for s in results)


def test_fontsf_category_chain_has_xrightarrow():
    results = _draw(2000)
    assert any(r"\mathsf{" in s and r"\xrightarrow" in s for s in results)


def test_fontit_polynomial_has_plus():
    results = _draw(2000)
    assert any(r"\mathit{" in s and "+" in s for s in results)


# ---------------------------------------------------------------------------
# Per-template sampling sanity
# ---------------------------------------------------------------------------


def test_mathtt_xor_slots_are_distinct():
    """All three variable slots in mathtt_xor must be pairwise distinct."""
    tmpl = _TMPL_BY_NAME["mathtt_xor"]
    rng = random.Random(42)
    for _ in range(200):
        rendered = sample(tmpl, rng)
        # Extract the three \mathtt{...} values
        import re

        vals = re.findall(r"\\mathtt\{([^}]+)\}", rendered)
        assert len(vals) == 3
        assert len(set(vals)) == 3, f"Non-distinct mathtt slots in: {rendered}"


def test_mathsf_functor_arrow_distinct_categories():
    """Source and target categories in mathsf_functor_arrow must differ."""
    tmpl = _TMPL_BY_NAME["mathsf_functor_arrow"]
    rng = random.Random(7)
    for _ in range(200):
        rendered = sample(tmpl, rng)
        import re

        cats = re.findall(r"\\mathsf\{([^}]+)\}", rendered)
        assert len(cats) == 2
        assert cats[0] != cats[1], f"Identical categories in: {rendered}"


def test_mathit_composed_map_distinct_functions():
    """The two function letters in mathit_composed_map must differ."""
    tmpl = _TMPL_BY_NAME["mathit_composed_map"]
    rng = random.Random(13)
    for _ in range(200):
        rendered = sample(tmpl, rng)
        import re

        fns = re.findall(r"\\mathit\{([^}]+)\}", rendered)
        assert len(fns) == 2
        assert fns[0] != fns[1], f"Identical function letters in: {rendered}"


# ---------------------------------------------------------------------------
# No exceptions over many draws (validates DSL invariants)
# ---------------------------------------------------------------------------


def test_no_exceptions_over_1000_draws():
    rng = random.Random(99)
    for _ in range(1000):
        _GEN(rng)  # must not raise
