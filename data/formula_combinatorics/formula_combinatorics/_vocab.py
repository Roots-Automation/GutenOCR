"""Shared vocabulary constants and atomic sampling helpers."""

from __future__ import annotations

import random

# ---------------------------------------------------------------------------
# Vocabulary pools
# ---------------------------------------------------------------------------

_VARS = ["x", "y", "z", "t", "u", "v", "r", "s"]
_GREEK = [
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\delta",
    r"\epsilon",
    r"\zeta",
    r"\eta",
    r"\theta",
    r"\kappa",
    r"\lambda",
    r"\mu",
    r"\nu",
    r"\xi",
    r"\rho",
    r"\sigma",
    r"\tau",
    r"\phi",
    r"\chi",
    r"\psi",
    r"\omega",
    r"\varepsilon",
    r"\varphi",
    r"\vartheta",
]
_GREEK_UPPER = [
    r"\Gamma",
    r"\Delta",
    r"\Lambda",
    r"\Sigma",
    r"\Omega",
    r"\Phi",
    r"\Psi",
    r"\Pi",
    r"\Xi",
    r"\Theta",
]
_SCALARS = list("abcdkmnpq")
_INDICES = list("ijklmn")
_POS_INTS = ["1", "2", "3", "4", "5", "6"]
_BOUNDS = ["0", "1", "a", "b", r"\pi", "T", "L", "-1"]
_MATRIX_NAMES = ["A", "B", "M", "P", "Q", "R"]
_CALLIGRAPHIC = [
    r"\mathcal{F}",
    r"\mathcal{G}",
    r"\mathcal{H}",
    r"\mathcal{L}",
    r"\mathcal{O}",
    r"\mathcal{S}",
    r"\mathcal{T}",
    r"\mathcal{B}",
]
_FRAKTUR = [r"\mathfrak{g}", r"\mathfrak{h}", r"\mathfrak{n}", r"\mathfrak{m}"]
_FUNCS = [
    r"\sin",
    r"\cos",
    r"\tan",
    r"\exp",
    r"\ln",
    r"\log",
    r"\cosh",
    r"\sinh",
    r"\tanh",
    r"\arcsin",
    r"\arccos",
    r"\arctan",
    r"\sec",
    r"\csc",
    r"\cot",
    r"\operatorname{erf}",
    r"\operatorname{sgn}",
]
_BBOLD = [
    r"\mathbb{R}",
    r"\mathbb{Z}",
    r"\mathbb{N}",
    r"\mathbb{Q}",
    r"\mathbb{C}",
    r"\mathbb{P}",
    r"\mathbb{F}",
]
_SETS = ["A", "B", "C", "S", "T", "U"]
_PROPS = ["P", "Q", "R"]
_RELATIONS = [
    r"\sim",
    r"\cong",
    r"\simeq",
    r"\approx",
    r"\propto",
    r"\ll",
    r"\gg",
    r"\preceq",
    r"\succeq",
]

# ---------------------------------------------------------------------------
# Single-item samplers
# ---------------------------------------------------------------------------


def _v(rng: random.Random) -> str:
    return rng.choice(_VARS)


def _g(rng: random.Random) -> str:
    return rng.choice(_GREEK)


def _gu(rng: random.Random) -> str:
    return rng.choice(_GREEK_UPPER)


def _s(rng: random.Random) -> str:
    return rng.choice(_SCALARS)


def _i(rng: random.Random) -> str:
    return rng.choice(_INDICES)


def _cal(rng: random.Random) -> str:
    return rng.choice(_CALLIGRAPHIC)


def _deco(rng: random.Random) -> str:
    cmd = rng.choice(
        [
            r"\hat",
            r"\bar",
            r"\dot",
            r"\ddot",
            r"\vec",
            r"\tilde",
            r"\widetilde",
            r"\widehat",
            r"\acute",
            r"\breve",
        ]
    )
    target = rng.choice(_VARS + list("abcfghpqrs"))
    return rf"{cmd}{{{target}}}"


def _atom(rng: random.Random) -> str:
    """One random atomic LaTeX symbol — evaluated lazily (one RNG draw only)."""
    builders = [
        lambda: _v(rng),
        lambda: _g(rng),
        lambda: _s(rng),
        lambda: f"{_s(rng)}_{{{_i(rng)}}}",
        lambda: f"{_v(rng)}_{{{rng.choice(_POS_INTS[:4])}}}",
        lambda: rng.choice(_POS_INTS),
        lambda: _deco(rng),
        lambda: _gu(rng),
        lambda: rng.choice(_CALLIGRAPHIC),
    ]
    return rng.choice(builders)()


def _expr(rng: random.Random, depth: int = 2) -> str:
    """Compositional expression builder with bounded depth."""
    if depth <= 0:
        return _atom(rng)
    r = rng.random()
    if r < 0.20:
        return _atom(rng)
    exp = rng.choice(["2", "3", "n", r"\alpha"])
    if r < 0.30:
        return rf"{_atom(rng)}^{{{exp}}}"
    if r < 0.42:
        return rf"\frac{{{_expr(rng, depth - 1)}}}{{{_expr(rng, depth - 1)}}}"
    if r < 0.50:
        return rf"\sqrt{{{_expr(rng, depth - 1)}}}"
    if r < 0.58:
        return rf"{_expr(rng, depth - 1)} + {_expr(rng, depth - 1)}"
    if r < 0.65:
        return rf"{_expr(rng, depth - 1)} - {_expr(rng, depth - 1)}"
    if r < 0.71:
        return rf"{_atom(rng)} {_expr(rng, depth - 1)}"
    if r < 0.78:
        fn = rng.choice(_FUNCS)
        return rf"{fn}\!\left({_expr(rng, depth - 1)}\right)"
    if r < 0.83:
        n = rng.choice(["n", "m", "N"])
        k = rng.choice(["k", "r", "j"])
        return rf"\binom{{{n}}}{{{k}}}"
    if r < 0.88:
        op = rng.choice([r"\max", r"\min", r"\sup", r"\inf"])
        return rf"{op}\!\left({_expr(rng, depth - 1)}\right)"
    if r < 0.92:
        return rf"\left\lfloor {_expr(rng, depth - 1)} \right\rfloor"
    if r < 0.96:
        return rf"\left\lceil {_expr(rng, depth - 1)} \right\rceil"
    return rf"\left({_expr(rng, depth - 1)}\right)^{{{exp}}}"


# ---------------------------------------------------------------------------
# Multi-item and structural helpers
# ---------------------------------------------------------------------------


def _idx_atom(rng: random.Random, prob: float = 0.35) -> str:
    """Sample _atom and optionally subscript it if it isn't already indexed or a bare digit."""
    a = _atom(rng)
    if "_" not in a and not a.lstrip("\\").isdigit():
        a = _maybe_idx(rng, a, prob)
    return a


def _maybe_idx(rng: random.Random, var: str, prob: float = 0.35) -> str:
    """Optionally append a subscript index to a plain variable, e.g. 'a' → 'a_{k}'.

    Only call on bare single-letter or Greek variables — never on already-decorated
    atoms (e.g. a_{j}, x_1, \\bar{x}) to avoid double-subscript output.
    """
    if rng.random() < prob:
        return rf"{var}_{{{rng.choice(['0', '1', '2', 'i', 'j', 'k', 'n', 'm'])}}}"
    return var


def _two(rng: random.Random, pool: list[str]) -> tuple[str, str]:
    a, b = rng.sample(pool, 2)
    return a, b


def _lower(rng: random.Random) -> str:
    return rng.choice(_BOUNDS + [r"-\infty"])


def _upper(rng: random.Random) -> str:
    return rng.choice(_BOUNDS + [r"\infty", r"+\infty"])


def _underbrace(expr: str, label: str) -> str:
    return rf"\underbrace{{{expr}}}_{{{label}}}"


def _overbrace(expr: str, label: str) -> str:
    return rf"\overbrace{{{expr}}}^{{{label}}}"


def _overset(rel: str, symbol: str) -> str:
    return rf"\overset{{{rel}}}{{{symbol}}}"


# ---------------------------------------------------------------------------
# Function-name decoration helpers (reusable across all domains)
# ---------------------------------------------------------------------------

_FN_BASE: tuple[str, ...] = (
    "f",
    "g",
    "h",
    "p",
    "q",
    "r",
    "F",
    "G",
    "H",
    "P",
    "Q",
    "R",
    r"\phi",
    r"\psi",
    r"\chi",
    r"\Phi",
    r"\Psi",
)
_FN_IDX: tuple[str, ...] = ("1", "2", "3", "i", "j", "k", "n", "m")


def _fn_rich(rng: random.Random) -> str:
    """Rich function-name token: base optionally decorated with subscript/hat/tilde/bar/star/dot."""
    base = rng.choice(_FN_BASE)
    d = rng.random()
    if d < 0.28:
        return f"{base}_{{{rng.choice(_FN_IDX)}}}"
    elif d < 0.42:
        return rf"\hat{{{base}}}"
    elif d < 0.56:
        return rf"\tilde{{{base}}}"
    elif d < 0.66:
        return rf"\bar{{{base}}}"
    elif d < 0.76:
        return f"{base}^{{*}}"
    elif d < 0.84:
        return rf"\dot{{{base}}}"
    else:
        return base


def _fn_rich_nosub(rng: random.Random) -> str:
    """Like _fn_rich but no subscripts or superscript decorations — safe where the template
    appends its own ^{(n)}, ', or '' to the function name."""
    base = rng.choice(_FN_BASE)
    d = rng.random()
    if d < 0.23:
        return rf"\hat{{{base}}}"
    elif d < 0.46:
        return rf"\tilde{{{base}}}"
    elif d < 0.63:
        return rf"\bar{{{base}}}"
    elif d < 0.77:
        return rf"\dot{{{base}}}"
    else:
        return base


# ---------------------------------------------------------------------------
# Trig function-name pools and selectors (reusable across algebra, calculus, …)
# ---------------------------------------------------------------------------

# Forward trig
_SIN_NAMES: tuple[str, ...] = (r"\sin", r"\text{sine}")
_COS_NAMES: tuple[str, ...] = (r"\cos", r"\text{cosine}")
_TAN_NAMES: tuple[str, ...] = (r"\tan", r"\text{tangent}")
_SEC_NAMES: tuple[str, ...] = (r"\sec", r"\text{secant}")
_CSC_NAMES: tuple[str, ...] = (r"\csc", r"\text{cosecant}")
_COT_NAMES: tuple[str, ...] = (r"\cot", r"\text{cotangent}")
# Hyperbolic (sh/ch/th = Russian/European shorthand)
_SINH_NAMES: tuple[str, ...] = (r"\sinh", r"\text{sh}")
_COSH_NAMES: tuple[str, ...] = (r"\cosh", r"\text{ch}")
_TANH_NAMES: tuple[str, ...] = (r"\tanh", r"\text{th}")
# Inverse — 4 visually distinct forms each
_ARCSIN_NAMES: tuple[str, ...] = (r"\arcsin", r"\sin^{-1}", r"\text{asin}", r"\text{Arcsin}")
_ARCCOS_NAMES: tuple[str, ...] = (r"\arccos", r"\cos^{-1}", r"\text{acos}", r"\text{Arccos}")
_ARCTAN_NAMES: tuple[str, ...] = (r"\arctan", r"\tan^{-1}", r"\text{atan}", r"\text{Arctan}")
_ARCCOT_NAMES: tuple[str, ...] = (r"\text{arccot}", r"\cot^{-1}", r"\text{acot}")
_ARCSEC_NAMES: tuple[str, ...] = (r"\text{arcsec}", r"\sec^{-1}", r"\text{asec}")
_ARCCSC_NAMES: tuple[str, ...] = (r"\text{arccsc}", r"\csc^{-1}", r"\text{acsc}")
_ARCTANH_NAMES: tuple[str, ...] = (r"\text{arctanh}", r"\tanh^{-1}", r"\text{atanh}")
_ARCSINH_NAMES: tuple[str, ...] = (r"\text{arcsinh}", r"\sinh^{-1}", r"\text{asinh}")
_ARCCOSH_NAMES: tuple[str, ...] = (r"\text{arccosh}", r"\cosh^{-1}", r"\text{acosh}")


def _sin_nm(rng: random.Random) -> str:
    return rng.choice(_SIN_NAMES)


def _cos_nm(rng: random.Random) -> str:
    return rng.choice(_COS_NAMES)


def _tan_nm(rng: random.Random) -> str:
    return rng.choice(_TAN_NAMES)


def _sec_nm(rng: random.Random) -> str:
    return rng.choice(_SEC_NAMES)


def _csc_nm(rng: random.Random) -> str:
    return rng.choice(_CSC_NAMES)


def _cot_nm(rng: random.Random) -> str:
    return rng.choice(_COT_NAMES)


def _sinh_nm(rng: random.Random) -> str:
    return rng.choice(_SINH_NAMES)


def _cosh_nm(rng: random.Random) -> str:
    return rng.choice(_COSH_NAMES)


def _tanh_nm(rng: random.Random) -> str:
    return rng.choice(_TANH_NAMES)


def _arcsin_nm(rng: random.Random) -> str:
    return rng.choice(_ARCSIN_NAMES)


def _arccos_nm(rng: random.Random) -> str:
    return rng.choice(_ARCCOS_NAMES)


def _arctan_nm(rng: random.Random) -> str:
    return rng.choice(_ARCTAN_NAMES)


def _arccot_nm(rng: random.Random) -> str:
    return rng.choice(_ARCCOT_NAMES)


def _arcsec_nm(rng: random.Random) -> str:
    return rng.choice(_ARCSEC_NAMES)


def _arccsc_nm(rng: random.Random) -> str:
    return rng.choice(_ARCCSC_NAMES)


def _arctanh_nm(rng: random.Random) -> str:
    return rng.choice(_ARCTANH_NAMES)


def _arcsinh_nm(rng: random.Random) -> str:
    return rng.choice(_ARCSINH_NAMES)


def _arccosh_nm(rng: random.Random) -> str:
    return rng.choice(_ARCCOSH_NAMES)
