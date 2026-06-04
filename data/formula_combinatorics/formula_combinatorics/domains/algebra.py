"""Algebra and trigonometry domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import E, P, S, Template, X, compute_weights, make_dispatcher
from .._templates import _poly
from .._vocab import (
    _GREEK,
    _SCALARS,
    _VARS,
    _atom,
    _expr,
    _fn_rich,
    _fn_rich_nosub,
    _idx_atom,
    _maybe_idx,
    _s,
)

# ---------------------------------------------------------------------------
# Shared pools
# ---------------------------------------------------------------------------

_UNION: list[str] = sorted(set(_VARS) | set(_SCALARS))
_VEC_POOL: list[str] = list("abcdefghijklmnopqrstuvwxyz")
_LOG_BASES: list[str] = ["2", "10", "e"] + _SCALARS
# Coefficients/ratios: scalars + Greek constants common in series/sequences
_COEFF_POOL: tuple[str, ...] = tuple(_SCALARS) + (
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\lambda",
    r"\mu",
    r"\rho",
    r"\kappa",
    r"\theta",
)
_GEO_N: tuple[str, ...] = ("n", "m", "N", "M", "K", "p", "r")
_EXP_POOL: tuple[str, ...] = ("2", "3", "4", "m", "n", "p", "q")

# ---------------------------------------------------------------------------
# Inline sub-generators
# ---------------------------------------------------------------------------


def _rel_sub(rng: random.Random) -> str:
    return rng.choice([">", "=", "<"])


def _exp_sym_sub(rng: random.Random) -> str:
    return rng.choice(["n", "m", "k", "p", "r", "2", "3", "4", "5", "6"])


def _fn_7_sub(rng: random.Random) -> str:
    return rng.choice(["f", "g", "h", "p", "q", "F", "G"])


def _fn_3_sub(rng: random.Random) -> str:
    return rng.choice(["f", "g", "h"])


def _eps_sub(rng: random.Random) -> str:
    return rng.choice([r"\epsilon", r"\varepsilon"])


def _tol_sub(rng: random.Random) -> str:
    return rng.choice([r"\epsilon", r"\varepsilon", r"\delta"])


def _scalar_sub(rng: random.Random) -> str:
    return rng.choice(_SCALARS)


def _log_n_sub(rng: random.Random) -> str:
    return rng.choice(["2", "3", "n", "k"] + _SCALARS)


def _poly_mid_3(rng: random.Random, v: str) -> str:
    """Middle terms for expanded polynomial of degree 3: one intermediate term."""
    return rf"{_s(rng)} {v}^{{2}}"


def _poly_mid_4(rng: random.Random, v: str) -> str:
    """Middle terms for expanded polynomial of degree 4: two intermediate terms."""
    return rf"{_s(rng)} {v}^{{3}} + {_s(rng)} {v}^{{2}}"


def _poly_mid_5(rng: random.Random, v: str) -> str:
    """Middle terms for expanded polynomial of degree 5: three intermediate terms."""
    return rf"{_s(rng)} {v}^{{4}} + {_s(rng)} {v}^{{3}} + {_s(rng)} {v}^{{2}}"


def _prod_n_sub(rng: random.Random) -> str:
    return _maybe_idx(rng, rng.choice(["n", "m", "N", "M", "r"]))


def _prod_a_sub(rng: random.Random) -> str:
    return _maybe_idx(rng, rng.choice([s for s in _SCALARS if s != "k"]))


def _prod_start_sub(rng: random.Random) -> str:
    return rng.choice(["1", "0", "2"])


def _cs_idx_sub(rng: random.Random) -> str:
    return rng.choice(["i", "j", "k", "l", "m", "r"])


def _cs_ub_sub(rng: random.Random) -> str:
    return rng.choice(["n", "m", "N", "M", "K", "L", "P"])


# ---------------------------------------------------------------------------
# Algebra templates
# ---------------------------------------------------------------------------

_QUAD_SLOTS: dict = {k: S(tuple(_UNION), idx=0.35) for k in ["v0", "p", "q", "r"]}
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
        slots={**_QUAD_SLOTS, "rel": E(_rel_sub, n=3)},
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
                    "v": S(tuple(_VARS), 0.35),
                    "a": S(tuple(_SCALARS), 0.35),
                    "b": X(tuple(_SCALARS), ("a",)),
                    "mid": P(_poly_mid_3, "v", n=9),
                    "s_lin": E(_scalar_sub, n=9),
                },
            ),
            Template(
                name="expanded_polynomial_deg4",
                latex=r"{a} {v}^{{4}} + {mid} + {s_lin} {v} + {b}",
                slots={
                    "v": S(tuple(_VARS), 0.35),
                    "a": S(tuple(_SCALARS), 0.35),
                    "b": X(tuple(_SCALARS), ("a",)),
                    "mid": P(_poly_mid_4, "v", n=81),
                    "s_lin": E(_scalar_sub, n=9),
                },
            ),
            Template(
                name="expanded_polynomial_deg5",
                latex=r"{a} {v}^{{5}} + {mid} + {s_lin} {v} + {b}",
                slots={
                    "v": S(tuple(_VARS), 0.35),
                    "a": S(tuple(_SCALARS), 0.35),
                    "b": X(tuple(_SCALARS), ("a",)),
                    "mid": P(_poly_mid_5, "v", n=729),
                    "s_lin": E(_scalar_sub, n=9),
                },
            ),
        ],
    ),
    Template(
        name="rational_fraction",
        latex=r"\frac{{{num}}}{{{den}}}",
        slots={
            "v": S(tuple(_VARS), 0.35),
            "num": P(_poly, "v", n=500),
            "den": P(_poly, "v", n=500),
        },
    ),
    Template(
        name="difference_of_squares",
        latex=r"\left({v} - {u}\right)\left({v} + {u}\right) = {v}^2 - \left({u}\right)^2",
        slots={
            "v": S(tuple(_VARS), 0.35),
            "u": E(_expr, n=5000),
        },
    ),
    Template(
        name="polynomial_nth_root",
        latex=r"\sqrt[{n}]{{{poly}}}",
        slots={
            "v": S(tuple(_VARS), 0.35),
            "n": E(lambda rng: rng.choice(["2", "3", "4", "5", "6", "n", "m", "k", "p"]), n=9),
            "poly": P(_poly, "v", n=5000),
        },
    ),
    Template(
        name="sum_of_cubes",
        latex=r"\left({u}\right)^3 + \left({w}\right)^3 = \left({u}+{w}\right)\left(\left({u}\right)^2 - {u} {w} + \left({w}\right)^2\right)",
        slots={
            "u": E(_expr, n=5000),
            "w": E(_expr, n=5000),
        },
    ),
    Template(
        name="difference_of_cubes",
        latex=r"\left({u}\right)^3 - \left({w}\right)^3 = \left({u}-{w}\right)\left(\left({u}\right)^2 + {u} {w} + \left({w}\right)^2\right)",
        slots={
            "u": E(_expr, n=5000),
            "w": E(_expr, n=5000),
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
                slots={"u": E(_expr, n=5000), "w": E(_expr, n=5000)},
            ),
            Template(
                name="perfect_square_minus",
                latex=r"\left({u} - {w}\right)^2 = \left({u}\right)^2 - 2 {u} {w} + \left({w}\right)^2",
                slots={"u": E(_expr, n=5000), "w": E(_expr, n=5000)},
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
                    "v": S(tuple(_VARS), 0.35),
                    "a": S(tuple(_SCALARS), idx=0.35),
                    "r0": X(tuple(_SCALARS), ("a",), idx=0.35),
                    "r1": X(tuple(_SCALARS), ("a", "r0"), idx=0.35),
                },
            ),
            Template(
                name="general_factored_3roots",
                latex=r"{a}\left({v} - {r0}\right)\left({v} - {r1}\right)\left({v} - {r2}\right) = 0",
                slots={
                    "v": S(tuple(_VARS), 0.35),
                    "a": S(tuple(_SCALARS), idx=0.35),
                    "r0": X(tuple(_SCALARS), ("a",), idx=0.35),
                    "r1": X(tuple(_SCALARS), ("a", "r0"), idx=0.35),
                    "r2": X(tuple(_SCALARS), ("a", "r0", "r1"), idx=0.35),
                },
            ),
            Template(
                name="general_factored_4roots",
                latex=(
                    r"{a}\left({v} - {r0}\right)\left({v} - {r1}\right)"
                    r"\left({v} - {r2}\right)\left({v} - {r3}\right) = 0"
                ),
                slots={
                    "v": S(tuple(_VARS), 0.35),
                    "a": S(tuple(_SCALARS), idx=0.35),
                    "r0": X(tuple(_SCALARS), ("a",), idx=0.35),
                    "r1": X(tuple(_SCALARS), ("a", "r0"), idx=0.35),
                    "r2": X(tuple(_SCALARS), ("a", "r0", "r1"), idx=0.35),
                    "r3": X(tuple(_SCALARS), ("a", "r0", "r1", "r2"), idx=0.35),
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
                    "v": S(tuple(_VARS), 0.35),
                    "fn": E(_fn_rich_nosub, n=100),
                    "pt": E(_expr, n=5000),
                },
            ),
            Template(
                name="factor_theorem",
                latex=r"{fn}({pt}) = 0 \implies ({v} - {pt}) \mid {fn}({v})",
                slots={
                    "v": S(tuple(_VARS), 0.35),
                    "fn": E(_fn_rich, n=272),
                    "pt": E(_expr, n=5000),
                },
            ),
        ],
    ),
    # ── General algebra (c=17..29) ───────────────────────────────────────────
    Template(
        name="binomial_theorem",
        latex=(
            r"\left({u} + {w}\right)^{{{exp}}} = "
            r"\sum_{{k=0}}^{{{exp}}} \binom{{{exp}}}{{k}} \left({u}\right)^k \left({w}\right)^{{{exp}-k}}"
        ),
        slots={
            "exp": E(_exp_sym_sub, n=10),
            "u": E(_expr, n=5000),
            "w": E(_expr, n=5000),
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
                    "base": S(tuple(_LOG_BASES)),
                    "v": S(tuple(_VARS), 0.35),
                    "a": S(tuple(_SCALARS), 0.35),
                    "b": X(tuple(_SCALARS), ("a",)),
                },
            ),
            Template(
                name="log_power_rule",
                latex=r"\log_{{{base}}}\!\left(\left({arg}\right)^{{{n}}}\right) = {n} \log_{{{base}}} {arg}",
                slots={
                    "base": S(tuple(_LOG_BASES)),
                    "n": E(_log_n_sub, n=13),
                    "arg": E(_expr, n=5000),
                },
            ),
            Template(
                name="log_change_of_base",
                latex=r"\log_{{{base}}} {arg} = \frac{{\log_{{{base2}}} {arg}}}{{\log_{{{base2}}} {base}}}",
                slots={
                    "base": S(tuple(_LOG_BASES)),
                    "base2": X(tuple(_LOG_BASES), ("base",)),
                    "arg": E(_expr, n=5000),
                },
            ),
            Template(
                name="log_product_rule",
                latex=(
                    r"\log_{{{base}}}\!\left({u} \cdot {w}\right) = "
                    r"\log_{{{base}}} {u} + \log_{{{base}}} {w}"
                ),
                slots={
                    "base": S(tuple(_LOG_BASES)),
                    "u": E(_expr, n=5000),
                    "w": E(_expr, n=5000),
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
                    "v": S(tuple(_VARS), 0.35),
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
                    "v": S(tuple(_VARS), 0.35),
                    "pt": E(_idx_atom, n=200),
                    "fn": E(_fn_rich, n=272),
                    "b": S(tuple(_SCALARS), 0.35),
                    "eps": E(_eps_sub, n=2),
                },
            ),
            Template(
                name="epsilon_delta_fn_nearness",
                latex=r"\left|{fn}({v}) - {b}\right| < {eps}",
                slots={
                    "v": S(tuple(_VARS), 0.35),
                    "fn": E(_fn_rich, n=272),
                    "b": S(tuple(_SCALARS), 0.35),
                    "eps": E(_eps_sub, n=2),
                },
            ),
            Template(
                name="epsilon_delta_interval",
                latex=r"{pt} - {eps} < {v} < {pt} + {eps}",
                slots={
                    "v": S(tuple(_VARS), 0.35),
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
                    "v": S(tuple(_VARS), 0.35),
                    "a": S(tuple(_SCALARS), 0.35),
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
                    "v": S(tuple(_VARS), 0.35),
                    "a": S(tuple(_SCALARS), 0.35),
                    "b": X(tuple(_SCALARS), ("a",)),
                    "p": X(tuple(_SCALARS), ("a", "b")),
                },
            ),
            Template(
                name="completing_square_partial",
                latex=(
                    r"{v}^2 + {coeff} {v} = "
                    r"\left({v} + \frac{{{coeff}}}{{2}}\right)^2 - \frac{{{coeff}^2}}{{4}}"
                ),
                slots={
                    "v": S(tuple(_VARS), 0.35),
                    "coeff": E(_idx_atom, n=200),
                },
            ),
            Template(
                name="completing_square_vertex",
                latex=r"f({v}) = {a}\!\left({v} - {b}\right)^2 + {c3}",
                slots={
                    "v": S(tuple(_VARS), 0.35),
                    "a": S(tuple(_SCALARS), 0.35),
                    "b": X(tuple(_SCALARS), ("a",)),
                    "c3": X(tuple(_SCALARS), ("a", "b")),
                },
            ),
        ],
    ),
    Template(
        name="floor_fraction",
        latex=r"\left\lfloor \frac{{{num}}}{{{den}}} \right\rfloor",
        slots={
            "num": E(_expr, n=5000),
            "den": E(_atom, n=150),
        },
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
                    "v": S(tuple(_VARS), 0.35),
                    "a": S(tuple(_SCALARS), 0.35),
                    "g": S(tuple(_GREEK), idx=0.35),
                },
            ),
            Template(
                name="exponential_decay",
                latex=r"{v} = {a} e^{{-{g} t}}",
                slots={
                    "v": S(tuple(_VARS), 0.35),
                    "a": S(tuple(_SCALARS), 0.35),
                    "g": S(tuple(_GREEK), idx=0.35),
                },
            ),
            Template(
                name="exponential_general_base",
                latex=r"{v} = {a} \cdot {b}^t",
                slots={
                    "v": S(tuple(_VARS), 0.35),
                    "a": S(tuple(_SCALARS), 0.35),
                    "b": X(tuple(_SCALARS), ("a",)),
                },
            ),
            Template(
                name="logistic_growth",
                latex=r"{v}(t) = \frac{{{a}}}{{1 + {b} e^{{-{g} t}}}}",
                slots={
                    "v": S(tuple(_VARS), 0.35),
                    "a": S(tuple(_SCALARS), 0.35),
                    "b": X(tuple(_SCALARS), ("a",)),
                    "g": S(tuple(_GREEK), idx=0.35),
                },
            ),
        ],
    ),
    Template(
        name="product_formula",
        latex=r"\prod_{{k={start}}}^{{{n}}} \left(1 + \frac{{{a11}}}{{k + {v}}}\right)",
        slots={
            "v": S(tuple(_VARS), 0.35),
            "n": E(_prod_n_sub, n=40),
            "a11": E(_prod_a_sub, n=70),
            "start": E(_prod_start_sub, n=3),
        },
    ),
    Template(
        name="proportion_identity",
        latex=r"\frac{{{e1}}}{{{e2}}} = \frac{{{e3}}}{{{e4}}}",
        slots={
            "e1": E(_expr, n=5000),
            "e2": E(_expr, n=5000),
            "e3": E(_expr, n=5000),
            "e4": E(_expr, n=5000),
        },
    ),
    Template(
        name="sum_of_squares",
        latex=r"\left({e1}\right)^2 + \left({e2}\right)^2 = {at}^2",
        slots={
            "e1": E(_expr, n=5000),
            "e2": E(_expr, n=5000),
            "at": E(_atom, n=150),
        },
    ),
    Template(
        name="partial_fraction",
        latex=r"\frac{{{a}}}{{{v}({v} - {b})}} = \frac{{{s1}}}{{{v}}} + \frac{{{s2}}}{{{v} - {b}}}",
        slots={
            "v": S(tuple(_VARS), 0.35),
            "a": S(tuple(_SCALARS), 0.35),
            "b": X(tuple(_SCALARS), ("a",)),
            "s1": E(_scalar_sub, n=9),
            "s2": E(_scalar_sub, n=9),
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
                    "vv": S(tuple(_VARS)),
                    "a": S(tuple(_SCALARS), 0.35),
                    "b": X(tuple(_SCALARS), ("a",)),
                    "c_coef": X(tuple(_SCALARS), ("a", "b")),
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
                    "vv": S(tuple(_VARS)),
                    "a": S(tuple(_SCALARS), 0.35),
                    "b": X(tuple(_SCALARS), ("a",)),
                    "c_coef": X(tuple(_SCALARS), ("a", "b")),
                    "d_coef": X(tuple(_SCALARS), ("a", "b", "c_coef")),
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
                    "v": S(tuple(_VARS), 0.35),
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
                    r"\frac{{{n}}}{{\sum_{{k=1}}^{{{n}}} \frac{{1}}{{{a}_k}}}} "
                    r"\leq \left(\prod_{{k=1}}^{{{n}}} {a}_k\right)^{{1/{n}}} "
                    r"\leq \frac{{1}}{{{n}}} \sum_{{k=1}}^{{{n}}} {a}_k"
                ),
                slots={
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
                    r"\left(\sum_{{{idx}=1}}^{{{ub}}} {p1}_{{{idx}}} \, {p2}_{{{idx}}}\right)^2 "
                    r"\leq \sum_{{{idx}=1}}^{{{ub}}} {p1}_{{{idx}}}^2 "
                    r"\cdot \sum_{{{idx}=1}}^{{{ub}}} {p2}_{{{idx}}}^2"
                ),
                slots={
                    "idx": E(_cs_idx_sub, n=6),
                    "ub": E(_cs_ub_sub, n=7),
                    "p1": S(tuple(_SCALARS)),
                    "p2": X(tuple(_SCALARS), ("p1",)),
                },
            ),
            Template(
                name="cauchy_schwarz_integral",
                latex=(
                    r"\left(\int_{{{lo}}}^{{{hi}}} {f1}({v}) \, {f2}({v}) \, d{v}\right)^2 "
                    r"\leq \int_{{{lo}}}^{{{hi}}} {f1}({v})^2 \, d{v} "
                    r"\cdot \int_{{{lo}}}^{{{hi}}} {f2}({v})^2 \, d{v}"
                ),
                slots={
                    "v": S(tuple(_VARS), 0.35),
                    "lo": E(_atom, n=150),
                    "hi": E(_atom, n=150),
                    "f1": E(_fn_rich, n=272),
                    "f2": E(_fn_rich, n=272),
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
                    "v": S(tuple(_VARS), 0.35),
                    "f": E(_fn_rich_nosub, n=100),
                    "q": E(_fn_rich_nosub, n=100),
                    "g": E(_fn_rich_nosub, n=100),
                    "r": E(_fn_rich_nosub, n=100),
                },
            ),
            Template(
                name="polynomial_division_remainder_form",
                latex=(r"{f}({v}) = {g}({v}) \cdot {q}({v}) + {r}({v})"),
                slots={
                    "v": S(tuple(_VARS), 0.35),
                    "f": E(_fn_rich_nosub, n=100),
                    "g": E(_fn_rich_nosub, n=100),
                    "q": E(_fn_rich_nosub, n=100),
                    "r": E(_fn_rich_nosub, n=100),
                },
            ),
            Template(
                name="polynomial_division_linear_divisor",
                latex=(r"{f}({v}) = ({v} - {a}) \cdot {q}({v}) + {f}({a})"),
                slots={
                    "v": S(tuple(_VARS), 0.35),
                    "a": S(_COEFF_POOL, idx=0.35),
                    "f": E(_fn_rich_nosub, n=100),
                    "q": E(_fn_rich_nosub, n=100),
                },
            ),
            Template(
                name="polynomial_division_uniqueness",
                latex=(
                    r"\exists!\, {q}, {r} : {f} = {g} \cdot {q} + {r}, "
                    r"\quad \deg {r} < \deg {g}"
                ),
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "g": E(_fn_rich_nosub, n=100),
                    "q": E(_fn_rich_nosub, n=100),
                    "r": E(_fn_rich_nosub, n=100),
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
                latex=r"\sum_{{k=1}}^{{{n}}} k = \frac{{{n}({n}+1)}}{{2}}",
                slots={"n": S(_GEO_N)},
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
                latex=r"\sum_{{k=1}}^{{{n}}} k^2 = \frac{{{n}({n}+1)(2{n}+1)}}{{6}}",
                slots={"n": S(_GEO_N)},
            ),
            Template(
                name="sum_of_cubes",
                latex=r"\sum_{{k=1}}^{{{n}}} k^3 = \left(\frac{{{n}({n}+1)}}{{2}}\right)^2",
                slots={"n": S(_GEO_N)},
            ),
            Template(
                name="arithmetic_progression_sum",
                latex=(
                    r"\sum_{{k=0}}^{{{n}}} \left({a} + k{d}\right) "
                    r"= \frac{{({n}+1)(2{a} + {n}{d})}}{{2}}"
                ),
                slots={
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
                    r"\sum_{{k=1}}^{{{n}}} \left({fn}(k+1) - {fn}(k)\right) "
                    r"= {fn}({n}+1) - {fn}(1)"
                ),
                slots={
                    "fn": E(_fn_rich_nosub, n=100),
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
                latex=(r"\sum_{{k=0}}^{{{n}}} {a} {r}^k = {a} \, \frac{{1 - {r}^{{{n}+1}}}}{{1 - {r}}}"),
                slots={
                    "a": S(_COEFF_POOL, idx=0.35),
                    "r": X(_COEFF_POOL, ("a",), idx=0.35),
                    "n": S(_GEO_N),
                },
                distinct=[["a", "r"]],
            ),
            Template(
                name="geometric_series_finite_unit",
                latex=(r"\sum_{{k=0}}^{{{n}}} {r}^k = \frac{{1 - {r}^{{{n}+1}}}}{{1 - {r}}}"),
                slots={
                    "r": S(_COEFF_POOL, idx=0.35),
                    "n": S(_GEO_N),
                },
            ),
            Template(
                name="geometric_series_infinite",
                latex=r"\sum_{{k=0}}^{{\infty}} {a} {r}^k = \frac{{{a}}}{{1 - {r}}}",
                slots={
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
                    "x": S(tuple(_VARS), 0.35),
                    "m": S(_EXP_POOL),
                    "n": X(_EXP_POOL, ("m",)),
                },
            ),
            Template(
                name="negative_exp",
                latex=r"{x}^{{-{n}}} = \frac{{1}}{{{x}^{{{n}}}}}",
                slots={
                    "x": S(tuple(_VARS), 0.35),
                    "n": S(_EXP_POOL),
                },
            ),
            Template(
                name="product_of_powers",
                latex=r"{x}^{{{m}}} \cdot {x}^{{{n}}} = {x}^{{{m}+{n}}}",
                slots={
                    "x": S(tuple(_VARS), 0.35),
                    "m": S(_EXP_POOL),
                    "n": X(_EXP_POOL, ("m",)),
                },
            ),
            Template(
                name="power_of_power",
                latex=r"\left({x}^{{{m}}}\right)^{{{n}}} = {x}^{{{m} \cdot {n}}}",
                slots={
                    "x": S(tuple(_VARS), 0.35),
                    "m": S(_EXP_POOL),
                    "n": X(_EXP_POOL, ("m",)),
                },
            ),
            Template(
                name="power_of_product",
                latex=r"({x}{y})^{{{n}}} = {x}^{{{n}}} {y}^{{{n}}}",
                slots={
                    "x": S(tuple(_VARS), 0.35),
                    "y": X(tuple(_VARS), ("x",), 0.35),
                    "n": S(_EXP_POOL),
                },
            ),
            Template(
                name="quotient_of_powers",
                latex=r"\frac{{{x}^{{{m}}}}}{{{x}^{{{n}}}}} = {x}^{{{m}-{n}}}",
                slots={
                    "x": S(tuple(_VARS), 0.35),
                    "m": S(_EXP_POOL),
                    "n": X(_EXP_POOL, ("m",)),
                },
            ),
            Template(
                name="zero_exponent",
                latex=r"{x}^0 = 1 \quad ({x} \neq 0)",
                slots={"x": S(tuple(_VARS), 0.35)},
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
                    "v": S(tuple(_VARS), 0.35),
                    "fn": E(_fn_rich_nosub, n=100),
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
                    "v": S(tuple(_VARS), 0.35),
                    "fn": E(_fn_rich_nosub, n=100),
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
                    "v": S(tuple(_VARS), 0.35),
                    "fn": E(_fn_rich_nosub, n=100),
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
                    "v": S(tuple(_VARS), 0.35),
                    "fn": E(_fn_rich_nosub, n=100),
                    "lc": S(_COEFF_POOL, idx=0.35),
                    "r": S(_COEFF_POOL),
                    "n": S(_GEO_N),
                },
                distinct=[["lc", "r"]],
            ),
        ],
    ),
]

# Sampling weights (sqrt of n_eff for balanced coverage)
# ---------------------------------------------------------------------------

# cap=75M: prevents _expr-heavy branches (n_eff~10^14) from crowding out named identities
_W_ALGEBRA: list[float] = compute_weights(_ALGEBRA_TEMPLATES, cap=75_000_000)


# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

_algebra = make_dispatcher(_ALGEBRA_TEMPLATES, _W_ALGEBRA)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "algebra": _algebra,
}

WEIGHTS: dict[str, float] = {
    "algebra": 0.09,
}

TEMPLATES: dict[str, list[Template]] = {
    "algebra": _ALGEBRA_TEMPLATES,
}
