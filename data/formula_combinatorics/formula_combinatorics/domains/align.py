"""Multi-line LaTeX align/multline/gather/split/cases domain generators.

Each of the 27 original _align() styles is expressed as a named Template with
typed Slot declarations.  n_eff estimates are conservative product-of-pool counts.
"""

from __future__ import annotations

import random

from .._template_dsl import _EXPR_SLOT, _SCALAR_SLOT, _VAR_SLOT, E, S, Template
from .._templates import _poly
from .._vocab import (
    _VARS,
    _expr,
    _s,
    _v,
)
from ._config import register_domain

# ---------------------------------------------------------------------------
# Shared pools
# ---------------------------------------------------------------------------

_LHS_SIMPLE = ("f(x)", "g(t)", "y", "I", "S")
_LHS_VAR = ("y", "z", "w")
_LHS_CALC = ("f(x)", "g(t)", "y", "S", "I")
_REL_INE = (r"\leq", r"\geq")
_REL_INE3 = (r"\leq", r"\geq", r"\ll")
_N_CHOICES = ("n", "N")
_LABELS = ("eq1", "eq2", "main", "result", "key", "def", "prop")
_DELIM_PAIRS = (("(", ")"), ("[", "]"))


# ---------------------------------------------------------------------------
# Helper Sub-generators (for styles that can't be expressed as format strings)
# ---------------------------------------------------------------------------


def _diff_op(rng: random.Random) -> str:
    v = rng.choice(_VARS)
    return rng.choice([rf"\frac{{d}}{{d{v}}}", rf"\frac{{d^2}}{{d{v}^2}}"])


def _piecewise_2(rng: random.Random) -> str:
    v = _v(rng)
    e1, e2 = _expr(rng, 1), _expr(rng, 1)
    cond = rng.choice(["0", _s(rng), r"\pi"])
    body = (
        rf"\begin{{cases}} "
        rf"{e1} & \text{{if }} {v} \geq {cond} \\"
        rf" {e2} & \text{{if }} {v} < {cond} "
        rf"\end{{cases}}"
    )
    return rf"\begin{{align*}}f({v}) &= {body}\end{{align*}}"


def _piecewise_3(rng: random.Random) -> str:
    v = _v(rng)
    e1, e2, e3 = _expr(rng, 1), _expr(rng, 1), _expr(rng, 1)
    body = (
        rf"\begin{{cases}} "
        rf"{e1} & \text{{if }} {v} > 0 \\"
        rf" {e2} & \text{{if }} {v} = 0 \\"
        rf" {e3} & \text{{if }} {v} < 0 "
        rf"\end{{cases}}"
    )
    return rf"\begin{{align*}}g({v}) &= {body}\end{{align*}}"


def _invis_bracket_split(rng: random.Random) -> str:
    lhs = rng.choice(("f(x)", "g(t)", "y", "S"))
    op, cl = rng.choice(_DELIM_PAIRS)
    e1, e2, e3, e4 = _expr(rng, 2), _expr(rng, 2), _expr(rng, 1), _expr(rng, 1)
    lines = [
        rf"{lhs} &= \left{op} {e1} + {e2} + \cdots \right.",
        rf"&\left. \quad + {e3} + {e4} \right{cl}",
    ]
    sep = r" \\"
    return rf"\begin{{align*}}{sep.join(lines)}\end{{align*}}"


def _integral_bounds_split(rng: random.Random) -> str:
    v = _v(rng)
    fn = rng.choice([rf"\frac{{d}}{{d{v}}}", rf"\frac{{d^2}}{{d{v}^2}}"])
    a, b = _s(rng), _s(rng)
    e1, e2, e3, e4 = _expr(rng, 2), _expr(rng, 1), _expr(rng, 1), _expr(rng, 1)
    lines = [
        rf"I &= \left[ {fn}\left( {e1} \right) + {e2} \right.",
        rf"&\left. \quad - {e3} \cdot {e4} \right]_{{{a}}}^{{{b}}}",
    ]
    sep = r" \\"
    return rf"\begin{{align*}}{sep.join(lines)}\end{{align*}}"


def _grouped_coeff(rng: random.Random) -> str:
    lhs = rng.choice(_LHS_VAR)
    c = _s(rng)
    e1, e2, e3, e4, e5, e6 = (
        _expr(rng, 2),
        _expr(rng, 2),
        _expr(rng, 1),
        _expr(rng, 1),
        _expr(rng, 1),
        _expr(rng, 1),
    )
    lines = [
        rf"{lhs} &= {c} \left( \frac{{{e1}}}{{{e2}}} + {e3} \right.",
        rf"&\left. \qquad + \frac{{{e4}}}{{{e5}}} \right) + {e6}",
    ]
    sep = r" \\"
    return rf"\begin{{align*}}{sep.join(lines)}\end{{align*}}"


def _multline_poly_body(rng: random.Random, env: str) -> str:
    v = _v(rng)
    poly = _poly(rng, v, max_degree=rng.randint(4, 6))
    terms = poly.split("+")
    mid = max(1, len(terms) // 2)
    first = "+".join(terms[:mid]).rstrip()
    rest = "+".join(terms[mid:]).lstrip()
    return rf"\begin{{{env}}}{first} \\ \quad + {rest}\end{{{env}}}"


def _multline_poly_starred(rng: random.Random) -> str:
    return _multline_poly_body(rng, "multline*")


def _multline_poly_numbered(rng: random.Random) -> str:
    return _multline_poly_body(rng, "multline")


def _gather_body(rng: random.Random, env: str) -> str:
    n_eqns = rng.randint(2, 3)
    eqn_lines = []
    for _ in range(n_eqns):
        lhs = rng.choice((_v(rng), "f(x)", "g(t)", "y"))
        eqn_lines.append(rf"{lhs} = {_expr(rng, 2)}")
    sep = r" \\"
    return rf"\begin{{{env}}}{sep.join(eqn_lines)}\end{{{env}}}"


def _gather_starred(rng: random.Random) -> str:
    return _gather_body(rng, "gather*")


def _gather_numbered(rng: random.Random) -> str:
    return _gather_body(rng, "gather")


def _multicolumn_scalars(rng: random.Random) -> str:
    rows = []
    for _ in range(3):
        v1, v2, v3 = _v(rng), _v(rng), _v(rng)
        c1, c2, c3 = _s(rng), _s(rng), _s(rng)
        rows.append(rf"{v1} &= {c1} && {v2} &= {c2} && {v3} &= {c3}")
    sep = r" \\"
    return rf"\begin{{align*}}{sep.join(rows)}\end{{align*}}"


def _multicolumn_mixed(rng: random.Random) -> str:
    rows = []
    for _ in range(2):
        v1, v2, v3 = _v(rng), _v(rng), _v(rng)
        e1, e2, e3 = _expr(rng, 1), _expr(rng, 1), _expr(rng, 1)
        rows.append(rf"{v1} &= {e1} && {v2} &= {e2} && {v3} &= {e3}")
    sep = r" \\"
    return rf"\begin{{align*}}{sep.join(rows)}\end{{align*}}"


def _multicolumn_inequalities(rng: random.Random) -> str:
    rel = rng.choice((r"\leq", r"\geq"))
    rows = []
    for _ in range(3):
        v1, v2 = _v(rng), _v(rng)
        e1, e2 = _expr(rng, 1), _expr(rng, 1)
        rows.append(rf"{v1} &{rel} {e1} && {v2} &{rel} {e2}")
    sep = r" \\"
    return rf"\begin{{align*}}{sep.join(rows)}\end{{align*}}"


# ---------------------------------------------------------------------------
# Templates — one per style (or variants for 2/3-line choices)
# ---------------------------------------------------------------------------

_TEMPLATES: list[Template] = [
    # style 0 — chained equalities (2 or 3 lines)
    Template(
        name="align_chained_equality",
        latex="",
        slots={},
        variants=[
            Template(
                name="align_chained_2lines",
                latex=r"\begin{{align*}}{lhs} &= {e1} \\ &= {e2}\end{{align*}}",
                slots={"lhs": S(_LHS_SIMPLE), "e1": _EXPR_SLOT, "e2": _EXPR_SLOT},
            ),
            Template(
                name="align_chained_3lines",
                latex=r"\begin{{align*}}{lhs} &= {e1} \\ &= {e2} \\ &= {e3}\end{{align*}}",
                slots={"lhs": S(_LHS_SIMPLE), "e1": _EXPR_SLOT, "e2": _EXPR_SLOT, "e3": _EXPR_SLOT},
            ),
        ],
    ),
    # style 1 — 2×2 linear system
    Template(
        name="align_linear_system_2x2",
        latex=(
            r"\begin{{align*}}"
            r"{a} x + {b} y &= {r1} \\ "
            r"{c} x + {d} y &= {r2}"
            r"\end{{align*}}"
        ),
        slots={
            "a": _SCALAR_SLOT,
            "b": _SCALAR_SLOT,
            "c": _SCALAR_SLOT,
            "d": _SCALAR_SLOT,
            "r1": _SCALAR_SLOT,
            "r2": _SCALAR_SLOT,
        },
    ),
    # style 2 — algebraic simplification
    Template(
        name="align_simplification",
        latex=(
            r"\begin{{align*}}"
            r"{lhs} &= {e1} + {e2} \\ "
            r"&= {e3}"
            r"\end{{align*}}"
        ),
        slots={"lhs": S((*_VARS, "f(x)")), "e1": _EXPR_SLOT, "e2": _EXPR_SLOT, "e3": _EXPR_SLOT},
    ),
    # style 3 — differential operator applied to expression
    Template(
        name="align_derivative",
        latex=(
            r"\begin{{align*}}"
            r"{op}\left[{e1}\right] &= {e2} \\ "
            r"&= {e3}"
            r"\end{{align*}}"
        ),
        slots={"op": E(_diff_op, n=10), "e1": _EXPR_SLOT, "e2": _EXPR_SLOT, "e3": _EXPR_SLOT},
    ),
    # style 4 — inequality chain
    Template(
        name="align_inequality_chain",
        latex="",
        slots={},
        variants=[
            Template(
                name="align_leq_chain",
                latex=r"\begin{{align*}}{e1} &\leq {e2} \\ &\leq {e3}\end{{align*}}",
                slots={"e1": _EXPR_SLOT, "e2": _EXPR_SLOT, "e3": _EXPR_SLOT},
            ),
            Template(
                name="align_geq_chain",
                latex=r"\begin{{align*}}{e1} &\geq {e2} \\ &\geq {e3}\end{{align*}}",
                slots={"e1": _EXPR_SLOT, "e2": _EXPR_SLOT, "e3": _EXPR_SLOT},
            ),
        ],
    ),
    # style 5 — summation expansion
    Template(
        name="align_series_expansion",
        latex="",
        slots={},
        variants=[
            Template(
                name="align_series_expansion_n",
                latex=(
                    r"\begin{{align*}}"
                    r"\sum_{{k=1}}^{{n}} a_k &= a_1 + a_2 + \cdots + a_{{n}} \\ "
                    r"&= {e}"
                    r"\end{{align*}}"
                ),
                slots={"e": _EXPR_SLOT},
            ),
            Template(
                name="align_series_expansion_N",
                latex=(
                    r"\begin{{align*}}"
                    r"\sum_{{k=1}}^{{N}} a_k &= a_1 + a_2 + \cdots + a_{{N}} \\ "
                    r"&= {e}"
                    r"\end{{align*}}"
                ),
                slots={"e": _EXPR_SLOT},
            ),
        ],
    ),
    # style 6 — 2-case piecewise
    Template(
        name="align_piecewise_2case",
        latex="{body}",
        slots={"body": E(_piecewise_2, n=2_000)},
    ),
    # style 7 — 3-case piecewise
    Template(
        name="align_piecewise_3case",
        latex="{body}",
        slots={"body": E(_piecewise_3, n=3_000)},
    ),
    # style 8 — series sum with underbrace
    Template(
        name="align_series_underbrace",
        latex="",
        slots={},
        variants=[
            Template(
                name="align_series_underbrace_n",
                latex=(
                    r"\begin{{align*}}"
                    r"S_{{n}} &= \underbrace{{a_1 + a_2 + \cdots + a_{{n}}}}_{{n \text{{ terms}}}} \\ "
                    r"&= {e1} + {e2}"
                    r"\end{{align*}}"
                ),
                slots={"e1": _EXPR_SLOT, "e2": _EXPR_SLOT},
            ),
            Template(
                name="align_series_underbrace_N",
                latex=(
                    r"\begin{{align*}}"
                    r"S_{{N}} &= \underbrace{{a_1 + a_2 + \cdots + a_{{N}}}}_{{N \text{{ terms}}}} \\ "
                    r"&= {e1} + {e2}"
                    r"\end{{align*}}"
                ),
                slots={"e1": _EXPR_SLOT, "e2": _EXPR_SLOT},
            ),
        ],
    ),
    # style 9 — 3×3 linear system
    Template(
        name="align_linear_system_3x3",
        latex=(
            r"\begin{{align*}}"
            r"{a} x + {b} y + {c} z &= {r1} \\ "
            r"{d} x + {e} y &= {r2} \\ "
            r"{f} z &= {r3}"
            r"\end{{align*}}"
        ),
        slots={
            "a": _SCALAR_SLOT,
            "b": _SCALAR_SLOT,
            "c": _SCALAR_SLOT,
            "d": _SCALAR_SLOT,
            "e": _SCALAR_SLOT,
            "f": _SCALAR_SLOT,
            "r1": _SCALAR_SLOT,
            "r2": _SCALAR_SLOT,
            "r3": _SCALAR_SLOT,
        },
    ),
    # style 10 — binomial expansion with overbrace
    Template(
        name="align_binomial_overbrace",
        latex="",
        slots={},
        variants=[
            Template(
                name="align_binomial_overbrace_n",
                latex=(
                    r"\begin{{align*}}"
                    r"(a+b)^{{n}} &= \overbrace{{a^2 + 2ab + b^2}}^{{(a+b)^2}} \cdot (a+b)^{{n-2}} \\ "
                    r"&= {e}"
                    r"\end{{align*}}"
                ),
                slots={"e": _EXPR_SLOT},
            ),
            Template(
                name="align_binomial_overbrace_3",
                latex=(
                    r"\begin{{align*}}"
                    r"(a+b)^{{3}} &= \overbrace{{a^2 + 2ab + b^2}}^{{(a+b)^2}} \cdot (a+b) \\ "
                    r"&= {e}"
                    r"\end{{align*}}"
                ),
                slots={"e": _EXPR_SLOT},
            ),
        ],
    ),
    # style 11 — transitive inequality chain
    Template(
        name="align_transitive_inequality",
        latex="",
        slots={},
        variants=[
            Template(
                name="align_transitive_leq",
                latex=r"\begin{{align*}}{e1} &\leq {e2} \\ &\leq {e3}\end{{align*}}",
                slots={"e1": _EXPR_SLOT, "e2": _EXPR_SLOT, "e3": _EXPR_SLOT},
            ),
            Template(
                name="align_transitive_geq",
                latex=r"\begin{{align*}}{e1} &\geq {e2} \\ &\geq {e3}\end{{align*}}",
                slots={"e1": _EXPR_SLOT, "e2": _EXPR_SLOT, "e3": _EXPR_SLOT},
            ),
            Template(
                name="align_transitive_ll",
                latex=r"\begin{{align*}}{e1} &\ll {e2} \\ &\ll {e3}\end{{align*}}",
                slots={"e1": _EXPR_SLOT, "e2": _EXPR_SLOT, "e3": _EXPR_SLOT},
            ),
        ],
    ),
    # style 12 — invisible-bracket split
    Template(
        name="align_invisible_bracket_split",
        latex="{body}",
        slots={"body": E(_invis_bracket_split, n=3_000)},
    ),
    # style 13 — derivative bracket spanning two rows
    Template(
        name="align_integral_bounds_split",
        latex="{body}",
        slots={"body": E(_integral_bounds_split, n=5_000)},
    ),
    # style 14 — grouped coefficient spanning two rows
    Template(
        name="align_grouped_coefficient",
        latex="{body}",
        slots={"body": E(_grouped_coeff, n=4_000)},
    ),
    # style 15 — multline*: long polynomial split
    Template(
        name="multline_poly_starred",
        latex="{body}",
        slots={"body": E(_multline_poly_starred, n=5_000)},
    ),
    # style 16 — multline*: summation expansion
    Template(
        name="multline_sum_expansion",
        latex="",
        slots={},
        variants=[
            Template(
                name="multline_sum_expansion_n",
                latex=(
                    r"\begin{{multline*}}"
                    r"{lhs} = {e1} + {e2} + \cdots \\ "
                    r"\quad + {e3} + {e4}"
                    r"\end{{multline*}}"
                ),
                slots={
                    "lhs": S(("f(x)", "S", "T", "I")),
                    "e1": _EXPR_SLOT,
                    "e2": _EXPR_SLOT,
                    "e3": _EXPR_SLOT,
                    "e4": _EXPR_SLOT,
                },
            ),
        ],
    ),
    # style 17 — multline (numbered): polynomial split
    Template(
        name="multline_poly_numbered",
        latex="{body}",
        slots={"body": E(_multline_poly_numbered, n=5_000)},
    ),
    # style 18 — gather* (unnumbered)
    Template(
        name="gather_starred",
        latex="{body}",
        slots={"body": E(_gather_starred, n=10_000)},
    ),
    # style 19 — gather (numbered)
    Template(
        name="gather_numbered",
        latex="{body}",
        slots={"body": E(_gather_numbered, n=10_000)},
    ),
    # style 20 — equation + split (algebraic derivation)
    Template(
        name="equation_split_algebraic",
        latex=(
            r"\begin{{equation}}\begin{{split}}"
            r"{lhs} &= {e1} + {e2} \\ "
            r"&= {e3} \\ "
            r"&= {e4}"
            r"\end{{split}}\end{{equation}}"
        ),
        slots={"lhs": S(_LHS_CALC), "e1": _EXPR_SLOT, "e2": _EXPR_SLOT, "e3": _EXPR_SLOT, "e4": _EXPR_SLOT},
    ),
    # style 21 — equation + split (calculus)
    Template(
        name="equation_split_calculus",
        latex=(
            r"\begin{{equation}}\begin{{split}}"
            r"\int_{{{a}}}^{{{b}}} {e1} \, d{v} &= \left[ {e2} \right]_{{{a}}}^{{{b}}} \\ "
            r"&= {e3}"
            r"\end{{split}}\end{{equation}}"
        ),
        slots={
            "v": _VAR_SLOT,
            "a": _SCALAR_SLOT,
            "b": _SCALAR_SLOT,
            "e1": _EXPR_SLOT,
            "e2": _EXPR_SLOT,
            "e3": _EXPR_SLOT,
        },
    ),
    # style 22 — multi-column: 3 cols × 3 rows of scalar equalities
    Template(
        name="align_multicolumn_scalars",
        latex="{body}",
        slots={"body": E(_multicolumn_scalars, n=1_000)},
    ),
    # style 23 — multi-column: 3 cols × 2 rows of mixed expressions
    Template(
        name="align_multicolumn_mixed",
        latex="{body}",
        slots={"body": E(_multicolumn_mixed, n=5_000)},
    ),
    # style 24 — multi-column: 2 cols × 3 rows of inequality chains
    Template(
        name="align_multicolumn_inequalities",
        latex="{body}",
        slots={"body": E(_multicolumn_inequalities, n=3_000)},
    ),
    # style 25 — labeled equation
    Template(
        name="equation_labeled",
        latex=r"\begin{{equation}}\label{{{lbl}}} {e} \end{{equation}}",
        slots={"lbl": S(_LABELS), "e": _EXPR_SLOT},
    ),
    # style 26 — numbered align with label on first row
    Template(
        name="align_labeled_numbered",
        latex="",
        slots={},
        variants=[
            Template(
                name="align_labeled_2lines",
                latex=(
                    r"\begin{{align}}"
                    r"\label{{{lbl}}} {lhs} &= {e1} \\ "
                    r"&= {e2}"
                    r"\end{{align}}"
                ),
                slots={"lbl": S(_LABELS), "lhs": S(("f(x)", "g(t)", "y", "S")), "e1": _EXPR_SLOT, "e2": _EXPR_SLOT},
            ),
        ],
    ),
]

GENERATORS, WEIGHTS, TEMPLATES = register_domain("align", _TEMPLATES)
