"""Algebra domain generators."""

from __future__ import annotations

from .._template_dsl import _ATOM_SLOT, _EXPR_SLOT, _FN_RICH_SLOT, _FN_SLOT, _LIM_MOD, E, P, S, Template, X
from .._templates import _poly, _poly_mid_factory, _substack_prod, _substack_sum
from .._vocab import (
    _COEFF_POOL,
    _GEO_N,
    _GREEK,
    _LOG_BASES,
    _SCALARS,
    _VARS,
    _VARS_SCALARS,
    _VEC_POOL,
    _atom,
    _eps_sub,
    _expr,
    _idx_atom,
    _tol_sub,
)
from ._config import register_domain

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

# _VARS_SCALARS, _GRP_NAMES, _RING_NAMES, _ELT_POOL, _LOG_BASES imported from _vocab
_EXP_POOL: tuple[str, ...] = ("2", "3", "4", "m", "n", "p", "q")
_MAT_POOL: tuple[str, ...] = ("A", "B", "C", "M", "T", "U")

# Style-modifier pool
_STYLE_CMDS: tuple[str, ...] = (r"\displaystyle", r"\textstyle", r"\scriptstyle", r"\scriptscriptstyle")

# \genfrac pools
_GENFRAC_LEFT: tuple[str, ...] = ("", r"\langle", r"\lfloor", r"\lceil", r"\|", "(", "[", r"\{")
_GENFRAC_RIGHT: tuple[str, ...] = ("", r"\rangle", r"\rfloor", r"\rceil", r"\|", ")", "]", r"\}")
_GENFRAC_THICK: tuple[str, ...] = ("", "0pt", "0.4pt", "0.8pt")
_GENFRAC_STYLE: tuple[str, ...] = ("", "0", "1", "2", "3")

# Text-fraction pools
_TEXT_NUMER: tuple[str, ...] = (
    r"\text{rise}",
    r"\text{distance}",
    r"\text{rate}",
    r"\text{observed}",
    r"\text{output}",
    r"\text{signal}",
    r"\text{profit}",
    r"\text{numerator}",
    r"\text{change in } y",
    r"\text{work}",
    r"\text{input}",
    r"\text{cost}",
)
_TEXT_DENOM: tuple[str, ...] = (
    r"\text{run}",
    r"\text{time}",
    r"\text{rate}",
    r"\text{expected}",
    r"\text{input}",
    r"\text{noise}",
    r"\text{cost}",
    r"\text{denominator}",
    r"\text{change in } x",
    r"\text{work}",
    r"\text{output}",
    r"\text{profit}",
)

# Polynomial degree pools
_DEG_POLY_POOL: tuple[str, ...] = ("p", "q", "f", "g", "h", "r", "s")
_DEG_N_POOL: tuple[str, ...] = ("n", "m", "d", "k", "r", "N")

# ---------------------------------------------------------------------------
# Inline sub-generators (ParamSub only — simple pools are inlined as S slots)
# ---------------------------------------------------------------------------


_poly_mid_3 = _poly_mid_factory(2)  # one middle term:   c v^2
_poly_mid_4 = _poly_mid_factory(3)  # two middle terms:  c v^3 + c v^2
_poly_mid_5 = _poly_mid_factory(4)  # three middle terms: c v^4 + c v^3 + c v^2


# ---------------------------------------------------------------------------
# Algebra templates
# ---------------------------------------------------------------------------

_QUAD_SLOTS: dict = {k: S(_VARS_SCALARS, idx=0.35) for k in ["v0", "p", "q", "r"]}
_QUAD_DISTINCT: list[list[str]] = [["v0", "p", "q", "r"]]

_ALGEBRA_TEMPLATES: list[Template] = [
    # ── Quadratic family (c=0..7) ────────────────────────────────────────────
    Template(
        name="quadratic_formula_pm",
        latex=r"{v0} = \frac{{-{q} \pm \sqrt{{{q}^2 - 4 {p} {r}}}}}{{2{p}}}",
        slots=_QUAD_SLOTS,
        distinct=_QUAD_DISTINCT,
    ),
    Template(
        name="quadratic_formula_pos",
        latex=r"{v0} = \frac{{-{q} + \sqrt{{{q}^2 - 4 {p} {r}}}}}{{2{p}}}",
        slots=_QUAD_SLOTS,
        distinct=_QUAD_DISTINCT,
    ),
    Template(
        name="quadratic_formula_neg",
        latex=r"{v0} = \frac{{-{q} - \sqrt{{{q}^2 - 4 {p} {r}}}}}{{2{p}}}",
        slots=_QUAD_SLOTS,
        distinct=_QUAD_DISTINCT,
    ),
    Template(
        name="discriminant_def",
        latex=r"\Delta = {q}^2 - 4 {p} {r}",
        slots=_QUAD_SLOTS,
        distinct=_QUAD_DISTINCT,
    ),
    Template(
        name="discriminant_condition",
        latex=r"{q}^2 - 4 {p} {r} {rel} 0",
        slots={**_QUAD_SLOTS, "rel": S((">", "=", "<", r"\geq", r"\leq", r"\neq"))},
        distinct=_QUAD_DISTINCT,
    ),
    Template(
        name="quadratic_factored",
        latex=r"{p}\left({v0} - {q}\right)\left({v0} - {r}\right) = 0",
        slots=_QUAD_SLOTS,
        distinct=_QUAD_DISTINCT,
    ),
    Template(
        name="quadratic_completed_square",
        latex=r"{p}\left({v0} + \frac{{{q}}}{{2{p}}}\right)^2 = \frac{{{q}^2 - 4 {p} {r}}}{{4{p}}}",
        slots=_QUAD_SLOTS,
        distinct=_QUAD_DISTINCT,
    ),
    Template(
        name="quadratic_monic",
        latex=r"{v0}^2 + \frac{{{q}}}{{{p}}}{v0} + \frac{{{r}}}{{{p}}} = 0",
        slots=_QUAD_SLOTS,
        distinct=_QUAD_DISTINCT,
    ),
    # ── Absolute value equations / inequalities ──────────────────────────────
    Template(
        name="abs_value_equations",
        latex="",
        slots={},
        variants=[
            Template(
                name="abs_val_eq",
                latex=r"\left|{a} {v} + {b}\right| = {c}",
                slots={
                    "v": S(_VARS, 0.35),
                    "a": S(_SCALARS, 0.35),
                    "b": S(_SCALARS, 0.35),
                    "c": S(_SCALARS, 0.35),
                },
                distinct=[["a", "b", "c"]],
            ),
            Template(
                name="abs_val_lt",
                latex=r"\left|{a} {v} + {b}\right| < {c}",
                slots={
                    "v": S(_VARS, 0.35),
                    "a": S(_SCALARS, 0.35),
                    "b": S(_SCALARS, 0.35),
                    "c": S(_SCALARS, 0.35),
                },
                distinct=[["a", "b", "c"]],
            ),
            Template(
                name="abs_val_gt",
                latex=r"\left|{a} {v} + {b}\right| > {c}",
                slots={
                    "v": S(_VARS, 0.35),
                    "a": S(_SCALARS, 0.35),
                    "b": S(_SCALARS, 0.35),
                    "c": S(_SCALARS, 0.35),
                },
                distinct=[["a", "b", "c"]],
            ),
            Template(
                name="abs_val_leq",
                latex=r"\left|{a} {v} + {b}\right| \leq {c}",
                slots={
                    "v": S(_VARS, 0.35),
                    "a": S(_SCALARS, 0.35),
                    "b": S(_SCALARS, 0.35),
                    "c": S(_SCALARS, 0.35),
                },
                distinct=[["a", "b", "c"]],
            ),
            Template(
                name="abs_val_two_sided",
                latex=r"{c1} \leq \left|{v} - {pt}\right| \leq {c2}",
                slots={
                    "v": S(_VARS, 0.35),
                    "pt": E(_idx_atom, n=200),
                    "c1": S(_SCALARS, 0.35),
                    "c2": X(_SCALARS, ("c1",), 0.35),
                },
            ),
            Template(
                name="abs_val_expr",
                latex=r"\left|{expr}\right| = {c}",
                slots={
                    "expr": _EXPR_SLOT,
                    "c": S(_SCALARS, 0.35),
                },
            ),
        ],
    ),
    # ── Polynomial family (c=8..16) ──────────────────────────────────────────
    Template(
        name="expanded_polynomial",
        latex="",
        slots={},
        variants=[
            Template(
                name="expanded_polynomial_deg3",
                latex=r"{a} {v}^{{3}} + {mid} + {s_lin} {v} + {b}",
                slots={
                    "v": S(_VARS, 0.35),
                    "a": S(_SCALARS, 0.35),
                    "b": X(_SCALARS, ("a",)),
                    "mid": P(_poly_mid_3, "v", n=9),
                    "s_lin": S(_SCALARS),
                },
            ),
            Template(
                name="expanded_polynomial_deg4",
                latex=r"{a} {v}^{{4}} + {mid} + {s_lin} {v} + {b}",
                slots={
                    "v": S(_VARS, 0.35),
                    "a": S(_SCALARS, 0.35),
                    "b": X(_SCALARS, ("a",)),
                    "mid": P(_poly_mid_4, "v", n=81),
                    "s_lin": S(_SCALARS),
                },
            ),
            Template(
                name="expanded_polynomial_deg5",
                latex=r"{a} {v}^{{5}} + {mid} + {s_lin} {v} + {b}",
                slots={
                    "v": S(_VARS, 0.35),
                    "a": S(_SCALARS, 0.35),
                    "b": X(_SCALARS, ("a",)),
                    "mid": P(_poly_mid_5, "v", n=729),
                    "s_lin": S(_SCALARS),
                },
            ),
        ],
    ),
    Template(
        name="rational_fraction",
        latex=r"\frac{{{num}}}{{{den}}}",
        slots={
            "v": S(_VARS, 0.35),
            "num": P(_poly, "v", n=500),
            "den": P(_poly, "v", n=500),
        },
    ),
    Template(
        name="difference_of_squares",
        latex=r"\left({v} - {u}\right)\left({v} + {u}\right) = {v}^2 - \left({u}\right)^2",
        slots={
            "v": S(_VARS, 0.35),
            "u": _EXPR_SLOT,
        },
    ),
    Template(
        name="polynomial_nth_root",
        latex=r"\sqrt[{n}]{{{poly}}}",
        slots={
            "v": S(_VARS, 0.35),
            "n": E(lambda rng: rng.choice(["2", "3", "4", "5", "6", "n", "m", "k", "p"]), n=9),
            "poly": P(_poly, "v", n=5000),
        },
    ),
    Template(
        name="sum_of_cubes",
        latex=r"\left({u}\right)^3 + \left({w}\right)^3 = \left({u}+{w}\right)\left(\left({u}\right)^2 - {u} {w} + \left({w}\right)^2\right)",
        slots={
            "u": _EXPR_SLOT,
            "w": _EXPR_SLOT,
        },
    ),
    Template(
        name="difference_of_cubes",
        latex=r"\left({u}\right)^3 - \left({w}\right)^3 = \left({u}-{w}\right)\left(\left({u}\right)^2 + {u} {w} + \left({w}\right)^2\right)",
        slots={
            "u": _EXPR_SLOT,
            "w": _EXPR_SLOT,
        },
    ),
    Template(
        name="perfect_square",
        latex="",
        slots={},
        variants=[
            Template(
                name="perfect_square_plus",
                latex=r"\left({u} + {w}\right)^2 = \left({u}\right)^2 + 2 {u} {w} + \left({w}\right)^2",
                slots={"u": _EXPR_SLOT, "w": _EXPR_SLOT},
            ),
            Template(
                name="perfect_square_minus",
                latex=r"\left({u} - {w}\right)^2 = \left({u}\right)^2 - 2 {u} {w} + \left({w}\right)^2",
                slots={"u": _EXPR_SLOT, "w": _EXPR_SLOT},
            ),
        ],
    ),
    Template(
        name="general_factored",
        latex="",
        slots={},
        variants=[
            Template(
                name="general_factored_2roots",
                latex=r"{a}\left({v} - {r0}\right)\left({v} - {r1}\right) = 0",
                slots={
                    "v": S(_VARS, 0.35),
                    "a": S(_SCALARS, idx=0.35),
                    "r0": X(_SCALARS, ("a",), idx=0.35),
                    "r1": X(_SCALARS, ("a", "r0"), idx=0.35),
                },
            ),
            Template(
                name="general_factored_3roots",
                latex=r"{a}\left({v} - {r0}\right)\left({v} - {r1}\right)\left({v} - {r2}\right) = 0",
                slots={
                    "v": S(_VARS, 0.35),
                    "a": S(_SCALARS, idx=0.35),
                    "r0": X(_SCALARS, ("a",), idx=0.35),
                    "r1": X(_SCALARS, ("a", "r0"), idx=0.35),
                    "r2": X(_SCALARS, ("a", "r0", "r1"), idx=0.35),
                },
            ),
            Template(
                name="general_factored_4roots",
                latex=(
                    r"{a}\left({v} - {r0}\right)\left({v} - {r1}\right)"
                    r"\left({v} - {r2}\right)\left({v} - {r3}\right) = 0"
                ),
                slots={
                    "v": S(_VARS, 0.35),
                    "a": S(_SCALARS, idx=0.35),
                    "r0": X(_SCALARS, ("a",), idx=0.35),
                    "r1": X(_SCALARS, ("a", "r0"), idx=0.35),
                    "r2": X(_SCALARS, ("a", "r0", "r1"), idx=0.35),
                    "r3": X(_SCALARS, ("a", "r0", "r1", "r2"), idx=0.35),
                },
            ),
        ],
    ),
    Template(
        name="remainder_factor_theorem",
        latex="",
        slots={},
        variants=[
            Template(
                name="remainder_theorem",
                latex=r"{fn}({v}) = ({v} - {pt}) \cdot {fn}_1({v}) + {fn}({pt})",
                slots={
                    "v": S(_VARS, 0.35),
                    "fn": _FN_SLOT,
                    "pt": _EXPR_SLOT,
                },
            ),
            Template(
                name="factor_theorem",
                latex=r"{fn}({pt}) = 0 \implies ({v} - {pt}) \mid {fn}({v})",
                slots={
                    "v": S(_VARS, 0.35),
                    "fn": _FN_RICH_SLOT,
                    "pt": _EXPR_SLOT,
                },
            ),
        ],
    ),
    # ── Function composition and inverse identities ──────────────────────────
    Template(
        name="function_composition",
        latex="",
        slots={},
        variants=[
            Template(
                name="composition_two",
                latex=r"\left({f} \circ {g}\right)({v}) = {f}\!\left({g}({v})\right)",
                slots={
                    "v": S(_VARS, 0.35),
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                },
            ),
            Template(
                name="composition_three",
                latex=r"\left({f} \circ {g} \circ {h}\right)({v}) = {f}\!\left({g}\!\left({h}({v})\right)\right)",
                slots={
                    "v": S(_VARS, 0.35),
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "h": _FN_SLOT,
                },
            ),
            Template(
                name="inverse_cancel_right",
                latex=r"{f}\!\left({f}^{{-1}}({v})\right) = {v}",
                slots={
                    "v": S(_VARS, 0.35),
                    "f": _FN_SLOT,
                },
            ),
            Template(
                name="inverse_cancel_left",
                latex=r"{f}^{{-1}}\!\left({f}({v})\right) = {v}",
                slots={
                    "v": S(_VARS, 0.35),
                    "f": _FN_SLOT,
                },
            ),
            Template(
                name="inverse_composition",
                latex=r"\left({f} \circ {g}\right)^{{-1}} = {g}^{{-1}} \circ {f}^{{-1}}",
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                },
            ),
        ],
    ),
    # ── General algebra (c=17..29) ───────────────────────────────────────────
    Template(
        name="binomial_theorem",
        latex=(
            r"\left({u} + {w}\right)^{{{exp}}} = "
            r"\sum{lim_mod}_{{k=0}}^{{{exp}}} \binom{{{exp}}}{{k}} \left({u}\right)^k \left({w}\right)^{{{exp}-k}}"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "exp": S(("n", "m", "k", "p", "r", "2", "3", "4", "5", "6")),
            "u": _EXPR_SLOT,
            "w": _EXPR_SLOT,
        },
    ),
    Template(
        name="log_identities",
        latex="",
        slots={},
        variants=[
            Template(
                name="log_quotient_rule",
                latex=(
                    r"\log_{{{base}}}\!\left(\frac{{{a} {v}}}{{{b}}}\right) = "
                    r"\log_{{{base}}} {a} + \log_{{{base}}} {v} - \log_{{{base}}} {b}"
                ),
                slots={
                    "base": S(_LOG_BASES),
                    "v": S(_VARS, 0.35),
                    "a": S(_SCALARS, 0.35),
                    "b": X(_SCALARS, ("a",)),
                },
            ),
            Template(
                name="log_power_rule",
                latex=r"\log_{{{base}}}\!\left(\left({arg}\right)^{{{n}}}\right) = {n} \log_{{{base}}} {arg}",
                slots={
                    "base": S(_LOG_BASES),
                    "n": S(("2", "3", "n", "k", "a", "b", "c", "d", "m", "p", "q")),
                    "arg": _EXPR_SLOT,
                },
            ),
            Template(
                name="log_change_of_base",
                latex=r"\log_{{{base}}} {arg} = \frac{{\log_{{{base2}}} {arg}}}{{\log_{{{base2}}} {base}}}",
                slots={
                    "base": S(_LOG_BASES),
                    "base2": X(_LOG_BASES, ("base",)),
                    "arg": _EXPR_SLOT,
                },
            ),
            Template(
                name="log_product_rule",
                latex=(
                    r"\log_{{{base}}}\!\left({u} \cdot {w}\right) = "
                    r"\log_{{{base}}} {u} + \log_{{{base}}} {w}"
                ),
                slots={
                    "base": S(_LOG_BASES),
                    "u": _EXPR_SLOT,
                    "w": _EXPR_SLOT,
                },
            ),
            Template(
                name="log_base_one",
                latex=r"\log_{{{base}}} 1 = 0",
                slots={"base": S(_LOG_BASES, idx=0.35)},
            ),
            Template(
                name="log_base_self",
                latex=r"\log_{{{base}}} {base} = 1",
                slots={"base": S(_LOG_BASES, idx=0.35)},
            ),
            Template(
                name="log_exp_cancel",
                latex=r"\log_{{{base}}} {base}^{{{n}}} = {n}",
                slots={
                    "base": S(_LOG_BASES, idx=0.35),
                    "n": S(("2", "3", "n", "k", "m", "p")),
                },
            ),
            Template(
                name="log_base_power_cancel",
                latex=r"{base}^{{\log_{{{base}}} {arg}}} = {arg}",
                slots={
                    "base": S(_LOG_BASES, idx=0.35),
                    "arg": _EXPR_SLOT,
                },
            ),
        ],
    ),
    Template(
        name="epsilon_delta",
        latex="",
        slots={},
        variants=[
            Template(
                name="epsilon_delta_nearness",
                latex=r"\left|{v} - {pt}\right| < {tol}",
                slots={
                    "v": S(_VARS, 0.35),
                    "pt": E(_idx_atom, n=200),
                    "tol": E(_tol_sub, n=3),
                },
            ),
            Template(
                name="epsilon_delta_full",
                latex=(
                    r"0 < \left|{v} - {pt}\right| < \delta "
                    r"\implies \left|{fn}({v}) - {b}\right| < {eps}"
                ),
                slots={
                    "v": S(_VARS, 0.35),
                    "pt": E(_idx_atom, n=200),
                    "fn": _FN_RICH_SLOT,
                    "b": S(_SCALARS, 0.35),
                    "eps": E(_eps_sub, n=2),
                },
            ),
            Template(
                name="epsilon_delta_fn_nearness",
                latex=r"\left|{fn}({v}) - {b}\right| < {eps}",
                slots={
                    "v": S(_VARS, 0.35),
                    "fn": _FN_RICH_SLOT,
                    "b": S(_SCALARS, 0.35),
                    "eps": E(_eps_sub, n=2),
                },
            ),
            Template(
                name="epsilon_delta_interval",
                latex=r"{pt} - {eps} < {v} < {pt} + {eps}",
                slots={
                    "v": S(_VARS, 0.35),
                    "pt": E(_idx_atom, n=200),
                    "eps": E(_eps_sub, n=2),
                },
            ),
        ],
    ),
    Template(
        name="completing_the_square",
        latex="",
        slots={},
        variants=[
            Template(
                name="completing_square_monic",
                latex=(
                    r"{v}^2 + {a} {v} + {c_const} = "
                    r"\left({v} + \frac{{{a}}}{{2}}\right)^2 + {c_const} - \frac{{{a}^2}}{{4}}"
                ),
                slots={
                    "v": S(_VARS, 0.35),
                    "a": S(_SCALARS, 0.35),
                    "c_const": E(_idx_atom, n=200),
                },
            ),
            Template(
                name="completing_square_nonmonic",
                latex=(
                    r"{p} {v}^2 + {a} {v} + {b} = "
                    r"{p}\!\left({v} + \frac{{{a}}}{{2{p}}}\right)^2 + {b} - \frac{{{a}^2}}{{4{p}}}"
                ),
                slots={
                    "v": S(_VARS, 0.35),
                    "a": S(_SCALARS, 0.35),
                    "b": X(_SCALARS, ("a",)),
                    "p": X(_SCALARS, ("a", "b")),
                },
            ),
            Template(
                name="completing_square_partial",
                latex=(
                    r"{v}^2 + {coeff} {v} = "
                    r"\left({v} + \frac{{{coeff}}}{{2}}\right)^2 - \frac{{{coeff}^2}}{{4}}"
                ),
                slots={
                    "v": S(_VARS, 0.35),
                    "coeff": E(_idx_atom, n=200),
                },
            ),
            Template(
                name="completing_square_vertex",
                latex=r"f({v}) = {a}\!\left({v} - {b}\right)^2 + {c3}",
                slots={
                    "v": S(_VARS, 0.35),
                    "a": S(_SCALARS, 0.35),
                    "b": X(_SCALARS, ("a",)),
                    "c3": X(_SCALARS, ("a", "b")),
                },
            ),
        ],
    ),
    Template(
        name="floor_ceil",
        latex="",
        slots={},
        variants=[
            Template(
                name="floor_fraction",
                latex=r"\left\lfloor \frac{{{num}}}{{{den}}} \right\rfloor",
                slots={
                    "num": _EXPR_SLOT,
                    "den": _ATOM_SLOT,
                },
            ),
            Template(
                name="ceil_fraction",
                latex=r"\left\lceil \frac{{{num}}}{{{den}}} \right\rceil",
                slots={
                    "num": _EXPR_SLOT,
                    "den": _ATOM_SLOT,
                },
            ),
            Template(
                name="floor_add_int",
                latex=r"\left\lfloor {v} + {n} \right\rfloor = \left\lfloor {v} \right\rfloor + {n}",
                slots={
                    "v": _ATOM_SLOT,
                    "n": S(_SCALARS),
                },
            ),
            Template(
                name="ceil_neg_floor",
                latex=r"\left\lceil {v} \right\rceil = -\left\lfloor -{v} \right\rfloor",
                slots={"v": _ATOM_SLOT},
            ),
            Template(
                name="floor_sum_bound",
                latex=r"\left\lfloor {u} \right\rfloor + \left\lfloor {w} \right\rfloor \leq \left\lfloor {u} + {w} \right\rfloor",
                slots={
                    "u": _EXPR_SLOT,
                    "w": _EXPR_SLOT,
                },
            ),
        ],
    ),
    Template(
        name="exponential_growth",
        latex="",
        slots={},
        variants=[
            Template(
                name="exponential_growth_pos",
                latex=r"{v} = {a} e^{{{g} t}}",
                slots={
                    "v": S(_VARS, 0.35),
                    "a": S(_SCALARS, 0.35),
                    "g": S(_GREEK, idx=0.35),
                },
            ),
            Template(
                name="exponential_decay",
                latex=r"{v} = {a} e^{{-{g} t}}",
                slots={
                    "v": S(_VARS, 0.35),
                    "a": S(_SCALARS, 0.35),
                    "g": S(_GREEK, idx=0.35),
                },
            ),
            Template(
                name="exponential_general_base",
                latex=r"{v} = {a} \cdot {b}^t",
                slots={
                    "v": S(_VARS, 0.35),
                    "a": S(_SCALARS, 0.35),
                    "b": X(_SCALARS, ("a",)),
                },
            ),
            Template(
                name="logistic_growth",
                latex=r"{v}(t) = \frac{{{a}}}{{1 + {b} e^{{-{g} t}}}}",
                slots={
                    "v": S(_VARS, 0.35),
                    "a": S(_SCALARS, 0.35),
                    "b": X(_SCALARS, ("a",)),
                    "g": S(_GREEK, idx=0.35),
                },
            ),
        ],
    ),
    Template(
        name="product_formula",
        latex=r"\prod{lim_mod}_{{k={start}}}^{{{n}}} \left(1 + \frac{{{a11}}}{{k + {v}}}\right)",
        slots={
            "lim_mod": _LIM_MOD,
            "v": S(_VARS, 0.35),
            "n": S(("n", "m", "N", "M", "r"), idx=0.35),
            "a11": S(("a", "b", "c", "d", "m", "n", "p", "q"), idx=0.35),
            "start": S(("1", "0", "2")),
        },
    ),
    Template(
        name="proportion_identity",
        latex=r"\frac{{{e1}}}{{{e2}}} = \frac{{{e3}}}{{{e4}}}",
        slots={
            "e1": _EXPR_SLOT,
            "e2": _EXPR_SLOT,
            "e3": _EXPR_SLOT,
            "e4": _EXPR_SLOT,
        },
    ),
    Template(
        name="sum_of_squares",
        latex=r"\left({e1}\right)^2 + \left({e2}\right)^2 = {at}^2",
        slots={
            "e1": _EXPR_SLOT,
            "e2": _EXPR_SLOT,
            "at": _ATOM_SLOT,
        },
    ),
    Template(
        name="partial_fraction",
        latex=r"\frac{{{a}}}{{{v}({v} - {b})}} = \frac{{{s1}}}{{{v}}} + \frac{{{s2}}}{{{v} - {b}}}",
        slots={
            "v": S(_VARS, 0.35),
            "a": S(_SCALARS, 0.35),
            "b": X(_SCALARS, ("a",)),
            "s1": S(_SCALARS),
            "s2": S(_SCALARS),
        },
    ),
    Template(
        name="vieta_formulas",
        latex="",
        slots={},
        variants=[
            Template(
                name="vieta_quadratic",
                latex=(
                    r"{vv}_1 + {vv}_2 = -\frac{{{b}}}{{{a}}}, "
                    r"\quad {vv}_1 {vv}_2 = \frac{{{c_coef}}}{{{a}}}"
                ),
                slots={
                    "vv": S(_VARS),
                    "a": S(_SCALARS, 0.35),
                    "b": X(_SCALARS, ("a",)),
                    "c_coef": X(_SCALARS, ("a", "b")),
                },
            ),
            Template(
                name="vieta_cubic",
                latex=(
                    r"{vv}_1 + {vv}_2 + {vv}_3 = -\frac{{{b}}}{{{a}}}, \quad "
                    r"{vv}_1 {vv}_2 + {vv}_1 {vv}_3 + {vv}_2 {vv}_3 = \frac{{{c_coef}}}{{{a}}}, \quad "
                    r"{vv}_1 {vv}_2 {vv}_3 = -\frac{{{d_coef}}}{{{a}}}"
                ),
                slots={
                    "vv": S(_VARS),
                    "a": S(_SCALARS, 0.35),
                    "b": X(_SCALARS, ("a",)),
                    "c_coef": X(_SCALARS, ("a", "b")),
                    "d_coef": X(_SCALARS, ("a", "b", "c_coef")),
                },
            ),
        ],
    ),
    Template(
        name="am_gm",
        latex="",
        slots={},
        variants=[
            Template(
                name="am_gm_two",
                latex=r"\frac{{{u} + {w}}}{{2}} \geq \sqrt{{{u} \, {w}}}",
                slots={
                    "u": S(_COEFF_POOL, idx=0.35),
                    "w": S(_COEFF_POOL, idx=0.35),
                },
                distinct=[["u", "w"]],
            ),
            Template(
                name="am_gm_three",
                latex=r"\frac{{{u} + {w} + {x}}}{{3}} \geq \sqrt[3]{{{u} \, {w} \, {x}}}",
                slots={
                    "u": S(_COEFF_POOL, idx=0.35),
                    "w": S(_COEFF_POOL, idx=0.35),
                    "x": S(_COEFF_POOL, idx=0.35),
                },
                distinct=[["u", "w", "x"]],
            ),
            Template(
                name="am_gm_power_mean",
                latex=(
                    r"\frac{{{u}^{{1/{v}}} + {w}^{{1/{v}}}}}{{2}} "
                    r"\geq \left(\frac{{{u} + {w}}}{{2}}\right)^{{1/{v}}}"
                ),
                slots={
                    "v": S(_VARS, 0.35),
                    "u": S(_COEFF_POOL, idx=0.35),
                    "w": S(_COEFF_POOL, idx=0.35),
                },
                distinct=[["u", "w"]],
            ),
            Template(
                name="am_gm_nvar",
                latex=(
                    r"\frac{{{a}_1 + {a}_2 + \cdots + {a}_{{{n}}}}}{{{n}}} "
                    r"\geq \sqrt[{{{n}}}]{{{a}_1 \cdot {a}_2 \cdots {a}_{{{n}}}}}"
                ),
                slots={
                    "a": S(_COEFF_POOL),
                    "n": S(_GEO_N),
                },
            ),
            Template(
                name="hm_gm_two",
                latex=r"\frac{{2{u}\,{w}}}{{{u}+{w}}} \leq \sqrt{{{u}\,{w}}}",
                slots={
                    "u": S(_COEFF_POOL, idx=0.35),
                    "w": S(_COEFF_POOL, idx=0.35),
                },
                distinct=[["u", "w"]],
            ),
            Template(
                name="hm_gm_am_chain",
                latex=(r"\frac{{2{u}\,{w}}}{{{u}+{w}}} \leq \sqrt{{{u}\,{w}}} \leq \frac{{{u}+{w}}}{{2}}"),
                slots={
                    "u": S(_COEFF_POOL, idx=0.35),
                    "w": S(_COEFF_POOL, idx=0.35),
                },
                distinct=[["u", "w"]],
            ),
            Template(
                name="hm_def",
                latex=r"H_{{{n}}} = \frac{{{n}}}{{\frac{{1}}{{{a}_1}} + \frac{{1}}{{{a}_2}} + \cdots + \frac{{1}}{{{a}_{{{n}}}}}}}",
                slots={
                    "a": S(_COEFF_POOL),
                    "n": S(_GEO_N),
                },
            ),
            Template(
                name="am_gm_hm_inequality",
                latex=(
                    r"\frac{{{n}}}{{\sum{lim_mod}_{{k=1}}^{{{n}}} \frac{{1}}{{{a}_k}}}} "
                    r"\leq \left(\prod{lim_mod}_{{k=1}}^{{{n}}} {a}_k\right)^{{1/{n}}} "
                    r"\leq \frac{{1}}{{{n}}} \sum{lim_mod}_{{k=1}}^{{{n}}} {a}_k"
                ),
                slots={
                    "lim_mod": _LIM_MOD,
                    "a": S(_COEFF_POOL),
                    "n": S(_GEO_N),
                },
            ),
        ],
    ),
    Template(
        name="cauchy_schwarz",
        latex="",
        slots={},
        variants=[
            Template(
                name="cauchy_schwarz_sum",
                latex=(
                    r"\left(\sum{lim_mod}_{{{idx}=1}}^{{{ub}}} {p1}_{{{idx}}} \, {p2}_{{{idx}}}\right)^2 "
                    r"\leq \sum{lim_mod}_{{{idx}=1}}^{{{ub}}} {p1}_{{{idx}}}^2 "
                    r"\cdot \sum{lim_mod}_{{{idx}=1}}^{{{ub}}} {p2}_{{{idx}}}^2"
                ),
                slots={
                    "lim_mod": _LIM_MOD,
                    "idx": S(("i", "j", "k", "l", "m", "r")),
                    "ub": S(("n", "m", "N", "M", "K", "L", "P")),
                    "p1": S(_SCALARS),
                    "p2": X(_SCALARS, ("p1",)),
                },
            ),
            Template(
                name="cauchy_schwarz_integral",
                latex=(
                    r"\left(\int{lim_mod}_{{{lo}}}^{{{hi}}} {f1}({v}) \, {f2}({v}) \, d{v}\right)^2 "
                    r"\leq \int{lim_mod}_{{{lo}}}^{{{hi}}} {f1}({v})^2 \, d{v} "
                    r"\cdot \int{lim_mod}_{{{lo}}}^{{{hi}}} {f2}({v})^2 \, d{v}"
                ),
                slots={
                    "lim_mod": _LIM_MOD,
                    "v": S(_VARS, 0.35),
                    "lo": _ATOM_SLOT,
                    "hi": _ATOM_SLOT,
                    "f1": _FN_RICH_SLOT,
                    "f2": _FN_RICH_SLOT,
                },
            ),
            Template(
                name="cauchy_schwarz_vector",
                latex=(
                    r"\left|\langle \mathbf{{{u1}}}, \, \mathbf{{{u2}}} \rangle\right|^2 "
                    r"\leq \left\|\mathbf{{{u1}}}\right\|^2 \left\|\mathbf{{{u2}}}\right\|^2"
                ),
                slots={
                    "u1": S(tuple(_VEC_POOL), idx=0.35),
                    "u2": X(tuple(_VEC_POOL), ("u1",), idx=0.35),
                },
            ),
        ],
    ),
    Template(
        name="polynomial_division",
        latex="",
        slots={},
        variants=[
            Template(
                name="polynomial_division_abstract",
                latex=(
                    r"{f}({v}) = {q}({v}) \cdot {g}({v}) + {r}({v}), "
                    r"\quad \deg {r} < \deg {g}"
                ),
                slots={
                    "v": S(_VARS, 0.35),
                    "f": _FN_SLOT,
                    "q": _FN_SLOT,
                    "g": _FN_SLOT,
                    "r": _FN_SLOT,
                },
            ),
            Template(
                name="polynomial_division_remainder_form",
                latex=(r"{f}({v}) = {g}({v}) \cdot {q}({v}) + {r}({v})"),
                slots={
                    "v": S(_VARS, 0.35),
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "q": _FN_SLOT,
                    "r": _FN_SLOT,
                },
            ),
            Template(
                name="polynomial_division_linear_divisor",
                latex=(r"{f}({v}) = ({v} - {a}) \cdot {q}({v}) + {f}({a})"),
                slots={
                    "v": S(_VARS, 0.35),
                    "a": S(_COEFF_POOL, idx=0.35),
                    "f": _FN_SLOT,
                    "q": _FN_SLOT,
                },
            ),
            Template(
                name="polynomial_division_uniqueness",
                latex=(
                    r"\exists!\, {q}, {r} : {f}({v}) = {g}({v}) \cdot {q}({v}) + {r}({v}), "
                    r"\quad \deg {r} < \deg {g}"
                ),
                slots={
                    "v": S(_VARS, 0.35),
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "q": _FN_SLOT,
                    "r": _FN_SLOT,
                },
            ),
        ],
    ),
    Template(
        name="triangle_inequality",
        latex="",
        slots={},
        variants=[
            Template(
                name="triangle_inequality_basic",
                latex=r"\left|{a} + {b}\right| \leq \left|{a}\right| + \left|{b}\right|",
                slots={
                    "a": S(_COEFF_POOL, idx=0.35),
                    "b": S(_COEFF_POOL, idx=0.35),
                },
                distinct=[["a", "b"]],
            ),
            Template(
                name="triangle_inequality_difference",
                latex=r"\left|{a} - {b}\right| \leq \left|{a}\right| + \left|{b}\right|",
                slots={
                    "a": S(_COEFF_POOL, idx=0.35),
                    "b": S(_COEFF_POOL, idx=0.35),
                },
                distinct=[["a", "b"]],
            ),
            Template(
                name="triangle_inequality_reverse",
                latex=(
                    r"\left|\left|{a}\right| - \left|{b}\right|\right| "
                    r"\leq \left|{a} - {b}\right|"
                ),
                slots={
                    "a": S(_COEFF_POOL, idx=0.35),
                    "b": S(_COEFF_POOL, idx=0.35),
                },
                distinct=[["a", "b"]],
            ),
            Template(
                name="triangle_inequality_norm",
                latex=(r"\left\|{u} + {v}\right\| \leq \left\|{u}\right\| + \left\|{v}\right\|"),
                slots={
                    "u": S(tuple(_VEC_POOL), idx=0.35),
                    "v": X(tuple(_VEC_POOL), ("u",), idx=0.35),
                },
            ),
            Template(
                name="triangle_inequality_nterms",
                latex=(
                    r"\left|{a}_1 + {a}_2 + \cdots + {a}_{{{n}}}\right| "
                    r"\leq \left|{a}_1\right| + \left|{a}_2\right| + \cdots + \left|{a}_{{{n}}}\right|"
                ),
                slots={
                    "a": S(_COEFF_POOL),
                    "n": S(_GEO_N),
                },
            ),
        ],
    ),
    Template(
        name="power_sums",
        latex="",
        slots={},
        variants=[
            Template(
                name="sum_of_integers",
                latex=r"\sum{lim_mod}_{{k=1}}^{{{n}}} k = \frac{{{n}({n}+1)}}{{2}}",
                slots={"lim_mod": _LIM_MOD, "n": S(_GEO_N)},
            ),
            Template(
                name="sum_of_integers_ellipsis",
                latex="",
                slots={},
                variants=[
                    Template(
                        name="sum_of_integers_ellipsis_2",
                        latex=r"1 + 2 + \cdots + {n} = \frac{{{n}({n}+1)}}{{2}}",
                        slots={"n": S(_GEO_N)},
                    ),
                    Template(
                        name="sum_of_integers_ellipsis_3",
                        latex=r"1 + 2 + 3 + \cdots + {n} = \frac{{{n}({n}+1)}}{{2}}",
                        slots={"n": S(_GEO_N)},
                    ),
                    Template(
                        name="sum_of_integers_ellipsis_4",
                        latex=r"1 + 2 + 3 + 4 + \cdots + {n} = \frac{{{n}({n}+1)}}{{2}}",
                        slots={"n": S(_GEO_N)},
                    ),
                ],
            ),
            Template(
                name="sum_of_squares",
                latex=r"\sum{lim_mod}_{{k=1}}^{{{n}}} k^2 = \frac{{{n}({n}+1)(2{n}+1)}}{{6}}",
                slots={"lim_mod": _LIM_MOD, "n": S(_GEO_N)},
            ),
            Template(
                name="sum_of_cubes",
                latex=r"\sum{lim_mod}_{{k=1}}^{{{n}}} k^3 = \left(\frac{{{n}({n}+1)}}{{2}}\right)^2",
                slots={"lim_mod": _LIM_MOD, "n": S(_GEO_N)},
            ),
            Template(
                name="arithmetic_progression_sum",
                latex=(
                    r"\sum{lim_mod}_{{k=0}}^{{{n}}} \left({a} + k{d}\right) "
                    r"= \frac{{({n}+1)(2{a} + {n}{d})}}{{2}}"
                ),
                slots={
                    "lim_mod": _LIM_MOD,
                    "a": S(_COEFF_POOL, idx=0.35),
                    "d": X(_COEFF_POOL, ("a",), idx=0.35),
                    "n": S(_GEO_N),
                },
                distinct=[["a", "d"]],
            ),
            Template(
                name="arithmetic_progression_ellipsis",
                latex="",
                slots={},
                variants=[
                    Template(
                        name="arithmetic_progression_ellipsis_2",
                        latex=(
                            r"{a} + ({a}+{d}) + \cdots + ({a}+{n}{d}) "
                            r"= \frac{{({n}+1)(2{a}+{n}{d})}}{{2}}"
                        ),
                        slots={
                            "a": S(_COEFF_POOL, idx=0.35),
                            "d": X(_COEFF_POOL, ("a",), idx=0.35),
                            "n": S(_GEO_N),
                        },
                        distinct=[["a", "d"]],
                    ),
                    Template(
                        name="arithmetic_progression_ellipsis_3",
                        latex=(
                            r"{a} + ({a}+{d}) + ({a}+2{d}) + \cdots + ({a}+{n}{d}) "
                            r"= \frac{{({n}+1)(2{a}+{n}{d})}}{{2}}"
                        ),
                        slots={
                            "a": S(_COEFF_POOL, idx=0.35),
                            "d": X(_COEFF_POOL, ("a",), idx=0.35),
                            "n": S(_GEO_N),
                        },
                        distinct=[["a", "d"]],
                    ),
                    Template(
                        name="arithmetic_progression_ellipsis_4",
                        latex=(
                            r"{a} + ({a}+{d}) + ({a}+2{d}) + ({a}+3{d}) + \cdots + ({a}+{n}{d}) "
                            r"= \frac{{({n}+1)(2{a}+{n}{d})}}{{2}}"
                        ),
                        slots={
                            "a": S(_COEFF_POOL, idx=0.35),
                            "d": X(_COEFF_POOL, ("a",), idx=0.35),
                            "n": S(_GEO_N),
                        },
                        distinct=[["a", "d"]],
                    ),
                ],
            ),
            Template(
                name="telescoping_sum",
                latex=(
                    r"\sum{lim_mod}_{{k=1}}^{{{n}}} \left({fn}(k+1) - {fn}(k)\right) "
                    r"= {fn}({n}+1) - {fn}(1)"
                ),
                slots={
                    "lim_mod": _LIM_MOD,
                    "fn": _FN_SLOT,
                    "n": S(_GEO_N),
                },
            ),
        ],
    ),
    Template(
        name="geometric_series",
        latex="",
        slots={},
        variants=[
            Template(
                name="geometric_series_finite",
                latex=(r"\sum{lim_mod}_{{k=0}}^{{{n}}} {a} {r}^k = {a} \, \frac{{1 - {r}^{{{n}+1}}}}{{1 - {r}}}"),
                slots={
                    "lim_mod": _LIM_MOD,
                    "a": S(_COEFF_POOL, idx=0.35),
                    "r": X(_COEFF_POOL, ("a",), idx=0.35),
                    "n": S(_GEO_N),
                },
                distinct=[["a", "r"]],
            ),
            Template(
                name="geometric_series_finite_unit",
                latex=(r"\sum{lim_mod}_{{k=0}}^{{{n}}} {r}^k = \frac{{1 - {r}^{{{n}+1}}}}{{1 - {r}}}"),
                slots={
                    "lim_mod": _LIM_MOD,
                    "r": S(_COEFF_POOL, idx=0.35),
                    "n": S(_GEO_N),
                },
            ),
            Template(
                name="geometric_series_infinite",
                latex=r"\sum{lim_mod}_{{k=0}}^{{\infty}} {a} {r}^k = \frac{{{a}}}{{1 - {r}}}",
                slots={
                    "lim_mod": _LIM_MOD,
                    "a": S(_COEFF_POOL, idx=0.35),
                    "r": X(_COEFF_POOL, ("a",), idx=0.35),
                },
                distinct=[["a", "r"]],
            ),
            Template(
                name="geometric_series_closed_form",
                latex=(
                    r"{a} + {a} {r} + {a} {r}^2 + \cdots + {a} {r}^{{{n}}} "
                    r"= {a} \, \frac{{{r}^{{{n}+1}} - 1}}{{{r} - 1}}"
                ),
                slots={
                    "a": S(_COEFF_POOL, idx=0.35),
                    "r": X(_COEFF_POOL, ("a",), idx=0.35),
                    "n": S(_GEO_N),
                },
                distinct=[["a", "r"]],
            ),
        ],
    ),
    # ── Sophie Germain identity ──────────────────────────────────────────────
    Template(
        name="sophie_germain",
        latex="",
        slots={},
        variants=[
            Template(
                name="sophie_germain_standard",
                latex=(
                    r"{a}^4 + 4{b}^4 = "
                    r"\left({a}^2 + 2{b}^2 + 2{a} {b}\right)"
                    r"\left({a}^2 + 2{b}^2 - 2{a} {b}\right)"
                ),
                slots={"a": S(_COEFF_POOL, idx=0.35), "b": S(_COEFF_POOL, idx=0.35)},
                distinct=[["a", "b"]],
            ),
            Template(
                name="sophie_germain_sum_of_squares",
                latex=(
                    r"{a}^4 + 4{b}^4 = "
                    r"\left(({a}+{b})^2 + {b}^2\right)"
                    r"\left(({a}-{b})^2 + {b}^2\right)"
                ),
                slots={"a": S(_COEFF_POOL, idx=0.35), "b": S(_COEFF_POOL, idx=0.35)},
                distinct=[["a", "b"]],
            ),
            Template(
                name="sophie_germain_lhs_only",
                latex=r"{a}^4 + 4{b}^4",
                slots={"a": S(_COEFF_POOL, idx=0.35), "b": S(_COEFF_POOL, idx=0.35)},
                distinct=[["a", "b"]],
            ),
        ],
    ),
    # ── Difference / sum of nth powers ───────────────────────────────────────
    Template(
        name="nth_power_factoring",
        latex="",
        slots={},
        variants=[
            Template(
                name="difference_nth_powers_ellipsis2",
                latex=(
                    r"{a}^{{{n}}} - {b}^{{{n}}} = "
                    r"({a}-{b})\left({a}^{{{n}-1}} + {a}^{{{n}-2}}{b} + \cdots + {b}^{{{n}-1}}\right)"
                ),
                slots={
                    "a": S(_COEFF_POOL, idx=0.35),
                    "b": S(_COEFF_POOL, idx=0.35),
                    "n": S(_GEO_N),
                },
                distinct=[["a", "b"]],
            ),
            Template(
                name="difference_nth_powers_ellipsis3",
                latex=(
                    r"{a}^{{{n}}} - {b}^{{{n}}} = "
                    r"({a}-{b})\left({a}^{{{n}-1}} + {a}^{{{n}-2}}{b} + {a}^{{{n}-3}}{b}^2 + \cdots + {b}^{{{n}-1}}\right)"
                ),
                slots={
                    "a": S(_COEFF_POOL, idx=0.35),
                    "b": S(_COEFF_POOL, idx=0.35),
                    "n": S(_GEO_N),
                },
                distinct=[["a", "b"]],
            ),
            Template(
                name="sum_odd_powers_ellipsis2",
                latex=(
                    r"{a}^{{2{k}+1}} + {b}^{{2{k}+1}} = "
                    r"({a}+{b})\left({a}^{{2{k}}} - {a}^{{2{k}-1}}{b} + \cdots + {b}^{{2{k}}}\right)"
                ),
                slots={
                    "a": S(_COEFF_POOL, idx=0.35),
                    "b": S(_COEFF_POOL, idx=0.35),
                    "k": S(("n", "m", "p", "r", "j", "l")),
                },
                distinct=[["a", "b"]],
            ),
            Template(
                name="sum_odd_powers_ellipsis3",
                latex=(
                    r"{a}^{{2{k}+1}} + {b}^{{2{k}+1}} = "
                    r"({a}+{b})\left({a}^{{2{k}}} - {a}^{{2{k}-1}}{b} + {a}^{{2{k}-2}}{b}^2 - \cdots + {b}^{{2{k}}}\right)"
                ),
                slots={
                    "a": S(_COEFF_POOL, idx=0.35),
                    "b": S(_COEFF_POOL, idx=0.35),
                    "k": S(("n", "m", "p", "r", "j", "l")),
                },
                distinct=[["a", "b"]],
            ),
        ],
    ),
    # ── Conjugate radical pairs ───────────────────────────────────────────────
    Template(
        name="conjugate_pairs",
        latex="",
        slots={},
        variants=[
            Template(
                name="conjugate_radical_basic",
                latex=r"({a} + \sqrt{{{b}}})({a} - \sqrt{{{b}}}) = {a}^2 - {b}",
                slots={
                    "a": S(_COEFF_POOL, idx=0.35),
                    "b": S(_COEFF_POOL),
                },
                distinct=[["a", "b"]],
            ),
            Template(
                name="conjugate_radical_both",
                latex=r"(\sqrt{{{a}}} + \sqrt{{{b}}})(\sqrt{{{a}}} - \sqrt{{{b}}}) = {a} - {b}",
                slots={
                    "a": S(_COEFF_POOL, idx=0.35),
                    "b": S(_COEFF_POOL, idx=0.35),
                },
                distinct=[["a", "b"]],
            ),
            Template(
                name="conjugate_radical_coeff",
                latex=r"({a} + {c}\sqrt{{{b}}})({a} - {c}\sqrt{{{b}}}) = {a}^2 - {c}^2 {b}",
                slots={
                    "a": S(_COEFF_POOL, idx=0.35),
                    "b": S(_COEFF_POOL),
                    "c": S(_COEFF_POOL, idx=0.35),
                },
                distinct=[["a", "b", "c"]],
            ),
            Template(
                name="conjugate_complex",
                latex=r"({a} + {b}\,i)({a} - {b}\,i) = {a}^2 + {b}^2",
                slots={
                    "a": S(_COEFF_POOL, idx=0.35),
                    "b": S(_COEFF_POOL, idx=0.35),
                },
                distinct=[["a", "b"]],
            ),
            Template(
                name="conjugate_rationalise",
                latex=r"\frac{{1}}{{{a} + \sqrt{{{b}}}}} = \frac{{{a} - \sqrt{{{b}}}}}{{{a}^2 - {b}}}",
                slots={
                    "a": S(_COEFF_POOL, idx=0.35),
                    "b": S(_COEFF_POOL),
                },
                distinct=[["a", "b"]],
            ),
            Template(
                name="rationalize_scalar_plus_radical",
                latex=r"\frac{{{a}}}{{{b} + \sqrt{{{c}}}}} = \frac{{{a}({b} - \sqrt{{{c}}})}}{{ {b}^2 - {c} }}",
                slots={
                    "a": S(_COEFF_POOL, idx=0.35),
                    "b": S(_COEFF_POOL, idx=0.35),
                    "c": S(_COEFF_POOL),
                },
                distinct=[["a", "b", "c"]],
            ),
            Template(
                name="rationalize_two_radicals",
                latex=r"\frac{{{a}}}{{\sqrt{{{b}}} + \sqrt{{{c}}}}} = \frac{{{a}(\sqrt{{{b}}} - \sqrt{{{c}}})}}{{ {b} - {c} }}",
                slots={
                    "a": S(_COEFF_POOL, idx=0.35),
                    "b": S(_COEFF_POOL, idx=0.35),
                    "c": S(_COEFF_POOL, idx=0.35),
                },
                distinct=[["a", "b", "c"]],
            ),
        ],
    ),
    # ── Rational exponent rules ───────────────────────────────────────────────
    Template(
        name="rational_exponent_rules",
        latex="",
        slots={},
        variants=[
            Template(
                name="rational_exp_radical",
                latex=r"{x}^{{{m}/{n}}} = \sqrt[{{{n}}}]{{{x}^{{{m}}}}}",
                slots={
                    "x": S(_VARS, 0.35),
                    "m": S(_EXP_POOL),
                    "n": X(_EXP_POOL, ("m",)),
                },
            ),
            Template(
                name="negative_exp",
                latex=r"{x}^{{-{n}}} = \frac{{1}}{{{x}^{{{n}}}}}",
                slots={
                    "x": S(_VARS, 0.35),
                    "n": S(_EXP_POOL),
                },
            ),
            Template(
                name="product_of_powers",
                latex=r"{x}^{{{m}}} \cdot {x}^{{{n}}} = {x}^{{{m}+{n}}}",
                slots={
                    "x": S(_VARS, 0.35),
                    "m": S(_EXP_POOL),
                    "n": X(_EXP_POOL, ("m",)),
                },
            ),
            Template(
                name="power_of_power",
                latex=r"\left({x}^{{{m}}}\right)^{{{n}}} = {x}^{{{m} \cdot {n}}}",
                slots={
                    "x": S(_VARS, 0.35),
                    "m": S(_EXP_POOL),
                    "n": X(_EXP_POOL, ("m",)),
                },
            ),
            Template(
                name="power_of_product",
                latex=r"({x}{y})^{{{n}}} = {x}^{{{n}}} {y}^{{{n}}}",
                slots={
                    "x": S(_VARS, 0.35),
                    "y": X(_VARS, ("x",), 0.35),
                    "n": S(_EXP_POOL),
                },
            ),
            Template(
                name="quotient_of_powers",
                latex=r"\frac{{{x}^{{{m}}}}}{{{x}^{{{n}}}}} = {x}^{{{m}-{n}}}",
                slots={
                    "x": S(_VARS, 0.35),
                    "m": S(_EXP_POOL),
                    "n": X(_EXP_POOL, ("m",)),
                },
            ),
            Template(
                name="zero_exponent",
                latex=r"{x}^0 = 1 \quad ({x} \neq 0)",
                slots={"x": S(_VARS, 0.35)},
            ),
        ],
    ),
    # ── Polynomial root factoring form ────────────────────────────────────────
    Template(
        name="polynomial_root_form",
        latex="",
        slots={},
        variants=[
            Template(
                name="polynomial_root_form_general",
                latex=r"{fn}({v}) = {lc}({v} - {r1})({v} - {r2}) \cdots ({v} - {rn})",
                slots={
                    "v": S(_VARS, 0.35),
                    "fn": _FN_SLOT,
                    "lc": S(_COEFF_POOL, idx=0.35),
                    "r1": S(_COEFF_POOL, idx=0.35),
                    "r2": S(_COEFF_POOL, idx=0.35),
                    "rn": S(_COEFF_POOL, idx=0.35),
                },
                distinct=[["lc", "r1", "r2", "rn"]],
            ),
            Template(
                name="polynomial_root_form_monic",
                latex=r"{fn}({v}) = ({v} - {r1})({v} - {r2}) \cdots ({v} - {rn})",
                slots={
                    "v": S(_VARS, 0.35),
                    "fn": _FN_SLOT,
                    "r1": S(_COEFF_POOL, idx=0.35),
                    "r2": S(_COEFF_POOL, idx=0.35),
                    "rn": S(_COEFF_POOL, idx=0.35),
                },
                distinct=[["r1", "r2", "rn"]],
            ),
            Template(
                name="polynomial_root_form_quadratic",
                latex=r"{fn}({v}) = {lc}({v} - {r1})({v} - {r2})",
                slots={
                    "v": S(_VARS, 0.35),
                    "fn": _FN_SLOT,
                    "lc": S(_COEFF_POOL, idx=0.35),
                    "r1": S(_COEFF_POOL, idx=0.35),
                    "r2": S(_COEFF_POOL, idx=0.35),
                },
                distinct=[["lc", "r1", "r2"]],
            ),
            Template(
                name="polynomial_root_form_subscript",
                latex=r"{fn}({v}) = {lc}({v} - {r}_1)({v} - {r}_2) \cdots ({v} - {r}_{{{n}}})",
                slots={
                    "v": S(_VARS, 0.35),
                    "fn": _FN_SLOT,
                    "lc": S(_COEFF_POOL, idx=0.35),
                    "r": S(_COEFF_POOL),
                    "n": S(_GEO_N),
                },
                distinct=[["lc", "r"]],
            ),
        ],
    ),
]


_ALGEBRA_TEMPLATES += [
    # \leqslant / \geqslant
    Template(
        name="leqslant_chain",
        latex=r"{aa} \leqslant {bb} \leqslant {cc}",
        slots={
            "aa": S(_VARS_SCALARS),
            "bb": S(_VARS_SCALARS),
            "cc": S(_VARS_SCALARS),
        },
        distinct=[["aa", "bb", "cc"]],
    ),
    Template(
        name="norm_leqslant_bound",
        latex=r"\|{vv}\| \leqslant {aa} \|{ww}\|",
        slots={
            "vv": S(_VARS),
            "ww": S(_VARS),
            "aa": S(_SCALARS),
        },
        distinct=[["vv", "ww"]],
    ),
    Template(
        name="abs_geqslant_eps",
        latex=r"|{xx}| \geqslant {eps}",
        slots={
            "xx": S(_VARS),
            "eps": E(_eps_sub, n=2),
        },
    ),
    # \lll / \ggg
    Template(
        name="much_less_than",
        latex=r"{aa} \lll {bb}",
        slots={"aa": S(_VARS_SCALARS), "bb": S(_VARS_SCALARS)},
        distinct=[["aa", "bb"]],
    ),
    Template(
        name="much_greater_than",
        latex=r"{aa} \ggg {bb}",
        slots={"aa": S(_VARS_SCALARS), "bb": S(_VARS_SCALARS)},
        distinct=[["aa", "bb"]],
    ),
    # \lesssim / \gtrsim
    Template(
        name="norm_lesssim_power",
        latex=r"\|{ff}({xx})\| \lesssim \|{xx}\|^{{{nn}}}",
        slots={
            "ff": _FN_SLOT,
            "xx": S(_VARS),
            "nn": S(_EXP_POOL),
        },
    ),
    Template(
        name="gtrsim_scalars",
        latex=r"{aa} \gtrsim {bb}",
        slots={"aa": S(_VARS_SCALARS), "bb": S(_VARS_SCALARS)},
        distinct=[["aa", "bb"]],
    ),
    # \dagger — adjoint operator
    Template(
        name="adjoint_involutive",
        latex=r"({AA}^{{\dagger}})^{{\dagger}} = {AA}",
        slots={"AA": S(_MAT_POOL)},
    ),
    Template(
        name="adjoint_anti_multiplicative",
        latex=r"({AA} {BB})^{{\dagger}} = {BB}^{{\dagger}} {AA}^{{\dagger}}",
        slots={"AA": S(_MAT_POOL), "BB": S(_MAT_POOL)},
        distinct=[["AA", "BB"]],
    ),
    Template(
        name="adjoint_inner_product",
        latex=r"\langle {AA} {uu}, {vv} \rangle = \langle {uu}, {AA}^{{\dagger}} {vv} \rangle",
        slots={
            "AA": S(_MAT_POOL),
            "uu": S(_VARS),
            "vv": S(_VARS),
        },
        distinct=[["uu", "vv"]],
    ),
    # \ddagger — bidual
    Template(
        name="bidual_notation",
        latex=r"{AA}^{{\ddagger}} = ({AA}^{{\dagger}})^{{\dagger}}",
        slots={"AA": S(_MAT_POOL)},
    ),
    Template(
        name="bidual_canonical_embedding",
        latex=r"\iota : V \hookrightarrow V^{{\ddagger}},\quad \iota({vv})({ff}) = {ff}({vv})",
        slots={
            "vv": S(_VARS),
            "ff": S(("f", "g", "h", "F", "G")),
        },
        distinct=[["vv", "ff"]],
    ),
]


# group_homomorphism_rightarrow, conjugation_mapsto → group_theory.py
# ring_homomorphism_rightarrow, frobenius_endomorphism_longmapsto → ring_field_theory.py

# ---------------------------------------------------------------------------
# Large manual bracket-size templates (\biggl/\biggr, \Biggl/\Biggr)
# ---------------------------------------------------------------------------

_LARGE_BRACKET: list[Template] = [
    # Product of two sums — \biggl( \sum ... \biggr)\biggl( \sum ... \biggr)
    Template(
        name="prod_of_sums_biggl",
        latex=(
            r"\biggl( \sum{lim_mod}_{{k=1}}^{{{n}}} {a}_k \biggr)"
            r"\biggl( \sum{lim_mod}_{{k=1}}^{{{n}}} {b}_k \biggr)"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "a": S(list("abcdefg")),
            "b": S(list("hijklmn")),
            "n": S(("n", "N", "m")),
        },
    ),
    # Cauchy-Schwarz with \Biggl| absolute value and \Biggl( squared sums
    Template(
        name="cauchy_schwarz_biggl",
        latex=(
            r"\Biggl| \sum{lim_mod}_{{k=1}}^{{{n}}} {a}_k {b}_k \Biggr|^2"
            r" \leq \Biggl( \sum{lim_mod}_{{k=1}}^{{{n}}} {a}_k^2 \Biggr)"
            r"\Biggl( \sum{lim_mod}_{{k=1}}^{{{n}}} {b}_k^2 \Biggr)"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "a": S(list("abcde")),
            "b": S(list("fghij")),
            "n": S(("n", "N", "m")),
        },
    ),
    # Set comprehension with \Biggl\{ ... \Bigg| ... \Biggr\}
    Template(
        name="set_comprehension_biggl",
        latex=r"\Biggl\{{ {v} \in {sp} \;\Bigg|\; {f}({v}) \leq {c} \Biggr\}}",
        slots={
            "v": S(_VARS),
            "sp": S((r"\mathbb{R}", r"\mathbb{Z}", r"\mathbb{N}")),
            "f": _FN_SLOT,
            "c": _ATOM_SLOT,
        },
    ),
]
_ALGEBRA_TEMPLATES += _LARGE_BRACKET

# ---------------------------------------------------------------------------
# Math style modifier overrides (Gap 1)
# ---------------------------------------------------------------------------


_STYLE_MODIFIER_TEMPLATES: list[Template] = [
    Template(
        name="style_frac",
        latex=r"{style} \frac{{{num}}}{{{den}}}",
        slots={
            "style": S(_STYLE_CMDS),
            "num": _EXPR_SLOT,
            "den": _EXPR_SLOT,
        },
    ),
    Template(
        name="style_binom",
        latex=r"{style} \binom{{{top}}}{{{bot}}}",
        slots={
            "style": S(_STYLE_CMDS),
            "top": S(_GEO_N),
            "bot": X(_GEO_N, ("top",)),
        },
    ),
    Template(
        name="style_sum_frac",
        latex=r"{style} \sum_{{{lo}=0}}^{{{hi}}} \frac{{{num}}}{{{den}}}",
        slots={
            "style": S(_STYLE_CMDS),
            "lo": S(("k", "j", "i", "m")),
            "hi": S(("n", "N", "M", "p")),
            "num": _EXPR_SLOT,
            "den": _EXPR_SLOT,
        },
    ),
    Template(
        name="style_expr",
        latex=r"{style} {expr}",
        slots={
            "style": S(_STYLE_CMDS),
            "expr": _EXPR_SLOT,
        },
    ),
]

_ALGEBRA_TEMPLATES += _STYLE_MODIFIER_TEMPLATES

# ---------------------------------------------------------------------------
# \genfrac as general-purpose fraction builder (Gap 3)
# ---------------------------------------------------------------------------


_GENFRAC_TEMPLATES: list[Template] = [
    Template(
        name="genfrac_general",
        latex=r"\genfrac{{{lft}}}{{{rgt}}}{{{thk}}}{{{sty}}}{{{num}}}{{{den}}}",
        slots={
            "lft": S(_GENFRAC_LEFT),
            "rgt": S(_GENFRAC_RIGHT),
            "thk": S(_GENFRAC_THICK),
            "sty": S(_GENFRAC_STYLE),
            "num": _EXPR_SLOT,
            "den": _EXPR_SLOT,
        },
    ),
    Template(
        name="genfrac_no_rule",
        latex=r"\genfrac{{}}{{}}{{0pt}}{{}}{{{num}}}{{{den}}}",
        slots={
            "num": _EXPR_SLOT,
            "den": _EXPR_SLOT,
        },
    ),
    Template(
        name="genfrac_angle",
        latex=r"\genfrac{{\langle}}{{\rangle}}{{0pt}}{{}}{{{num}}}{{{den}}}",
        slots={
            "num": _EXPR_SLOT,
            "den": _EXPR_SLOT,
        },
    ),
]

_ALGEBRA_TEMPLATES += _GENFRAC_TEMPLATES

# ---------------------------------------------------------------------------
# Text-in-fraction patterns (Gap 5)
# ---------------------------------------------------------------------------


_TEXT_FRAC_TEMPLATES: list[Template] = [
    Template(
        name="text_over_text_frac",
        latex=r"\frac{{{num}}}{{{den}}}",
        slots={
            "num": S(_TEXT_NUMER),
            "den": X(_TEXT_DENOM, ("num",)),
        },
    ),
    Template(
        name="text_over_expr_frac",
        latex=r"\frac{{{num}}}{{{den}}}",
        slots={
            "num": S(_TEXT_NUMER),
            "den": _EXPR_SLOT,
        },
    ),
    Template(
        name="expr_over_text_frac",
        latex=r"\frac{{{num}}}{{{den}}}",
        slots={
            "num": _EXPR_SLOT,
            "den": S(_TEXT_DENOM),
        },
    ),
    Template(
        name="text_frac_equality",
        latex=r"{lhs} = \frac{{{num}}}{{{den}}}",
        slots={
            "lhs": S(_TEXT_NUMER),
            "num": S(_TEXT_NUMER),
            "den": X(_TEXT_DENOM, ("num",)),
        },
    ),
]

_ALGEBRA_TEMPLATES += _TEXT_FRAC_TEMPLATES


_DEG_TEMPLATES: list[Template] = [
    Template(
        name="deg_definition",
        latex=r"\deg({pp}) = {nn}",
        slots={
            "pp": S(_DEG_POLY_POOL),
            "nn": S(_DEG_N_POOL),
        },
        distinct=[["pp", "nn"]],
    ),
    Template(
        name="deg_product",
        latex=r"\deg({pp} \cdot {qq}) = \deg({pp}) + \deg({qq})",
        slots={
            "pp": S(_DEG_POLY_POOL),
            "qq": X(_DEG_POLY_POOL, ("pp",)),
        },
    ),
    Template(
        name="deg_sum_ineq",
        latex=r"\deg({pp} + {qq}) \leq \max(\deg({pp}),\, \deg({qq}))",
        slots={
            "pp": S(_DEG_POLY_POOL),
            "qq": X(_DEG_POLY_POOL, ("pp",)),
        },
    ),
    Template(
        name="deg_composition",
        latex=r"\deg({pp} \circ {qq}) = \deg({pp}) \cdot \deg({qq})",
        slots={
            "pp": S(_DEG_POLY_POOL),
            "qq": X(_DEG_POLY_POOL, ("pp",)),
        },
    ),
    Template(
        name="deg_monomial",
        latex=r"\deg({aa} {vv}^{{{nn}}}) = {nn}",
        slots={
            "aa": S(_SCALARS),
            "vv": S(_VARS),
            "nn": S(_DEG_N_POOL),
        },
        distinct=[["aa", "nn"]],
    ),
]

_ALGEBRA_TEMPLATES += _DEG_TEMPLATES

# ---------------------------------------------------------------------------
# Division operator templates
# ---------------------------------------------------------------------------

_DIV_TEMPLATES: list[Template] = [
    Template(
        name="div_expr_eq",
        latex=r"{a} \div {b} = {c}",
        slots={"a": E(_expr, n=1e4), "b": E(_expr, n=1e4), "c": E(_expr, n=1e4)},
    ),
    Template(
        name="div_remainder",
        latex=r"{a} \div {b} = {q} \cdots {r}",
        slots={"a": E(_atom, n=100), "b": E(_atom, n=100), "q": E(_atom, n=100), "r": E(_atom, n=100)},
    ),
    Template(
        name="div_fraction_identity",
        latex=r"{a} \div {b} = \frac{{{a2}}}{{{b2}}}",
        slots={"a": E(_atom, n=100), "b": E(_atom, n=100), "a2": E(_atom, n=100), "b2": E(_atom, n=100)},
    ),
    Template(
        name="div_paren_expr",
        latex=r"\left({a} + {b}\right) \div {c}",
        slots={"a": E(_expr, n=1e4), "b": E(_expr, n=1e4), "c": E(_expr, n=1e4)},
    ),
]

_ALGEBRA_TEMPLATES += _DIV_TEMPLATES

_SUBSTACK_ALG_TEMPLATES: list[Template] = [
    Template(
        name="substack_sum_alg",
        latex=r"{s}",
        slots={"s": E(_substack_sum, n=5_000_000)},
    ),
    Template(
        name="substack_prod_alg",
        latex=r"{s}",
        slots={"s": E(_substack_prod, n=5_000_000)},
    ),
    Template(
        name="substack_sum_eq",
        latex=r"{s} = {v}",
        slots={"s": E(_substack_sum, n=5_000_000), "v": E(_expr, n=5_000_000)},
    ),
]

_ALGEBRA_TEMPLATES += _SUBSTACK_ALG_TEMPLATES


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# cap=75M: prevents _expr-heavy branches (n_eff~10^14) from crowding out named identities
GENERATORS, WEIGHTS, TEMPLATES = register_domain("algebra", _ALGEBRA_TEMPLATES)
