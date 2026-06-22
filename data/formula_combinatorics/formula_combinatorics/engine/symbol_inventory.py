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

# ---------------------------------------------------------------------------
# Frequency-tier helpers (private)
# ---------------------------------------------------------------------------

# Most universal greek lower — appear in essentially every math domain.
_GREEK_HEAD: tuple[str, ...] = (
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\delta",
    r"\lambda",
    r"\mu",
    r"\sigma",
    r"\theta",
    r"\phi",
    r"\omega",
)
_GREEK_BODY: tuple[str, ...] = tuple(s for s in _GREEK if s not in _GREEK_HEAD)

# Most universal greek upper — standard summation/product/set notation.
_GREEK_UPPER_HEAD: tuple[str, ...] = (r"\Gamma", r"\Delta", r"\Sigma", r"\Pi", r"\Omega")
_GREEK_UPPER_BODY: tuple[str, ...] = tuple(s for s in _GREEK_UPPER if s not in _GREEK_UPPER_HEAD)

# Core blackboard bold — appear in virtually all analysis/algebra.
_BBOLD_HEAD: tuple[str, ...] = (r"\mathbb{R}", r"\mathbb{N}", r"\mathbb{C}", r"\mathbb{Z}")
_BBOLD_BODY: tuple[str, ...] = tuple(s for s in _BBOLD if s not in _BBOLD_HEAD and s != r"\mathbb{H}")

# Most common functions — taught in every calculus course.
_FUNCS_HEAD: tuple[str, ...] = (r"\sin", r"\cos", r"\tan", r"\exp", r"\ln", r"\log")
_FUNCS_TAIL: tuple[str, ...] = tuple(s for s in _FUNCS if s not in _FUNCS_HEAD)

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
# SYMBOL_FREQUENCY_TIERS: principled-proxy frequency partition of MUST_COVER
# ---------------------------------------------------------------------------
# Membership reflects structural role and ubiquity in standard math notation:
#   head  — appear in essentially all math domains (basic greek, core functions, core ℝℕℂℤ)
#   body  — common but domain-specific (remaining greek, calligraphic, remaining blackboard bold,
#            bold vectors)
#   tail  — specialist / rare in general corpora (bold greek, less-common trig/special functions)
# The three tiers are a disjoint partition of MUST_COVER.

SYMBOL_FREQUENCY_TIERS: dict[str, frozenset[str]] = {
    "head": frozenset((*_GREEK_HEAD, *_GREEK_UPPER_HEAD, *_BBOLD_HEAD, *_FUNCS_HEAD)),
    "body": frozenset((*_GREEK_BODY, *_GREEK_UPPER_BODY, *_BBOLD_BODY, *_CALLIGRAPHIC, *_BOLD_VECS)),
    "tail": frozenset((*_FUNCS_TAIL, *_BOLD_GREEK)),
}

# Reverse lookup: symbol → tier name (for per-sample labeling)
SYMBOL_TIER_OF: dict[str, str] = {sym: tier for tier, syms in SYMBOL_FREQUENCY_TIERS.items() for sym in syms}

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
COVERAGE_N: int = 8_000
COVERAGE_SEED: int = 0
