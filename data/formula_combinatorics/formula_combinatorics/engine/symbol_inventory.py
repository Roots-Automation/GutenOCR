"""Symbol inventory: declared symbol sets for coverage testing.

MUST_COVER
----------
Multi-character LaTeX tokens drawn from the shared _vocab.py pools.
Every token here must appear in a fixed-seed corpus of COVERAGE_N formulas.
Sourced descriptively from the shared pools so that adding a symbol to
_vocab.py automatically propagates it into the coverage check.

Single-letter variables (x, y, a, b, …) are excluded from MUST_COVER because
substring matching them in LaTeX strings produces false positives (e.g. "x"
inside \\exp or \\max).  They live in SHOULD_COVER instead.

SHOULD_COVER
------------
Broader set: auto-generated union of every Slot/ExcludeSlot pool across all
registered templates, minus empty strings.  Used for soft coverage reports
and future stratum checks (WU5).  Computed lazily via collect_should_cover().
"""

from __future__ import annotations

from ._vocab import (
    _BBOLD,
    _BOLD_GREEK,
    _BOLD_VECS,
    _CALLIGRAPHIC,
    _FUNCS,
    _GREEK,
    _GREEK_UPPER,
)

# \mathbb{H} (quaternions) appears only in a handful of algebra templates and
# is absent from typical 5k-sample corpora; exclude it from the hard gate.
_BBOLD_MUST: tuple[str, ...] = tuple(s for s in _BBOLD if s != r"\mathbb{H}")

# ---------------------------------------------------------------------------
# MUST_COVER: multi-character pool tokens that every 5k-formula corpus must contain
# ---------------------------------------------------------------------------
#
# Fraktur symbols (\mathfrak{g}, \mathfrak{h}, …) appear only in group-theory
# and ring/field templates and are absent from typical 5k corpora; they live
# in SHOULD_COVER instead.

MUST_COVER: frozenset[str] = frozenset(
    sym
    for sym in (
        *_GREEK,  # \alpha … \vartheta  (23 symbols)
        *_GREEK_UPPER,  # \Gamma … \Theta     (10 symbols)
        *_CALLIGRAPHIC,  # \mathcal{A} … \mathcal{V}  (18 symbols)
        *_BBOLD_MUST,  # \mathbb{C} … \mathbb{Z} minus \mathbb{H}  (8 symbols)
        *_FUNCS,  # \sin, \cos, … \operatorname{sgn}  (17 symbols)
        *_BOLD_VECS,  # \mathbf{a} … \mathbf{z}    (12 symbols)
        *_BOLD_GREEK,  # \boldsymbol{\alpha} … (12 symbols)
    )
    if sym  # exclude empty strings
)

# ---------------------------------------------------------------------------
# SYMBOL_STRATA: named partitions of MUST_COVER for stratum-level reporting
# ---------------------------------------------------------------------------

SYMBOL_STRATA: dict[str, frozenset[str]] = {
    "greek": frozenset((*_GREEK, *_GREEK_UPPER)),
    "calligraphic": frozenset(_CALLIGRAPHIC),
    "blackboard_bold": frozenset(_BBOLD_MUST),
    "functions": frozenset(_FUNCS),
    "bold_vectors": frozenset(_BOLD_VECS),
    "bold_greek": frozenset(_BOLD_GREEK),
}

# ---------------------------------------------------------------------------
# SHOULD_COVER: auto-generated from all Slot pools (lazy)
# ---------------------------------------------------------------------------


def collect_should_cover() -> frozenset[str]:
    """Return the union of every Slot/ExcludeSlot pool across all registered templates.

    Computed on demand to avoid importing the full domain registry at module load.
    Includes single-letter variables excluded from MUST_COVER.
    """
    from formula_combinatorics.domains import TEMPLATES
    from formula_combinatorics.engine._template_dsl import ExcludeSlot, Slot, Template

    symbols: set[str] = set()

    def _visit(t: Template) -> None:
        for s in t.slots.values():
            if isinstance(s, (Slot, ExcludeSlot)):
                symbols.update(sym for sym in s.pool if sym)
        for v in t.variants:
            _visit(v)

    for domain_templates in TEMPLATES.values():
        for t in domain_templates:
            _visit(t)

    return frozenset(symbols)


# Constants for coverage tests
COVERAGE_N: int = 5_000
COVERAGE_SEED: int = 0
