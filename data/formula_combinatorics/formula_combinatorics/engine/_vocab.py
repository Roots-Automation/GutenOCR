"""Shared vocabulary constants and atomic sampling helpers."""

from __future__ import annotations

import random
from collections.abc import Callable

# ---------------------------------------------------------------------------
# Vocabulary pools
# ---------------------------------------------------------------------------

_VARS: tuple[str, ...] = ("x", "y", "z", "t", "u", "v", "r", "s")
_GREEK: tuple[str, ...] = (
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
)
_GREEK_UPPER: tuple[str, ...] = (
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
)
_SCALARS: tuple[str, ...] = tuple("abcdkmnpq")
_INDICES: tuple[str, ...] = tuple("ijklmn")
_POS_INTS: tuple[str, ...] = ("1", "2", "3", "4", "5", "6")
_BOUNDS: tuple[str, ...] = ("0", "1", "a", "b", r"\pi", "T", "L", "-1")
# Canonical "how many" / loop-bound pool (replaces domain-local _N_POOL_STYLE etc.)
_LOOP_N: tuple[str, ...] = ("n", "m", "N", "M", "K", "k", "p", "r")
# Backward-compat alias used across domains
_GEO_N: tuple[str, ...] = _LOOP_N

# Composite pools (derived from the primitives above)
_VARS_SCALARS: tuple[str, ...] = tuple(sorted(set(_VARS) | set(_SCALARS)))
_GREEK_SCALARS: tuple[str, ...] = _SCALARS + _GREEK

# Generic cross-domain pools (import + alias in domain files; keep local only when semantics diverge)
# Generic function names: simple Latin/Greek, no calligraphic/script variants (those live in _FN_BASE)
_FUNC_NAMES: tuple[str, ...] = ("f", "g", "h", "F", "G", r"\phi", r"\psi", r"\varphi", r"\chi", r"\xi", r"\eta")
# Extended index pool: superset of _INDICES, includes iteration indices p/q/r/s/t
_GENERIC_IDX: tuple[str, ...] = tuple("ijklmnpqrst")
# Algebraic homomorphism notation (group_theory, ring_field_theory; richer versions keep local)
_HOMO_POOL: tuple[str, ...] = (
    r"\phi",
    r"\varphi",
    r"\psi",
    "f",
    r"\theta",
    r"\rho",
    r"\pi",
    "g",
    r"\alpha",
    r"\sigma",
)

# Abstract-algebra name pools (used across algebra, group_theory, ring_field_theory)
_GRP_NAMES: tuple[str, ...] = ("G", "H", "K", "N", "Q")
_RING_NAMES: tuple[str, ...] = ("R", "S", "A", "B")
_ELT_POOL: tuple[str, ...] = ("g", "h", "x", "r", "s", "a")

# Logarithm base pool (algebra, analysis, calculus)
_LOG_BASES: tuple[str, ...] = ("2", "10", "e") + _SCALARS + (r"\alpha", r"\beta", r"\lambda", r"\mu")

_COEFF_POOL: tuple[str, ...] = _SCALARS + (
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\lambda",
    r"\mu",
    r"\rho",
    r"\kappa",
    r"\theta",
)
_VEC_POOL: tuple[str, ...] = tuple("abcdefghijklmnopqrstuvwxyz")
_MATRIX_NAMES: tuple[str, ...] = ("A", "B", "M", "P", "Q", "R")
_CALLIGRAPHIC: tuple[str, ...] = (
    r"\mathcal{A}",
    r"\mathcal{B}",
    r"\mathcal{C}",
    r"\mathcal{D}",
    r"\mathcal{E}",
    r"\mathcal{F}",
    r"\mathcal{G}",
    r"\mathcal{H}",
    r"\mathcal{K}",
    r"\mathcal{L}",
    r"\mathcal{M}",
    r"\mathcal{N}",
    r"\mathcal{O}",
    r"\mathcal{P}",
    r"\mathcal{S}",
    r"\mathcal{T}",
    r"\mathcal{U}",
    r"\mathcal{V}",
)
_FRAKTUR: tuple[str, ...] = (
    r"\mathfrak{a}",
    r"\mathfrak{b}",
    r"\mathfrak{g}",
    r"\mathfrak{h}",
    r"\mathfrak{m}",
    r"\mathfrak{n}",
    r"\mathfrak{p}",
    r"\mathfrak{q}",
)
_FUNCS: tuple[str, ...] = (
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
)
_BBOLD: tuple[str, ...] = (
    r"\mathbb{C}",
    r"\mathbb{F}",
    r"\mathbb{H}",
    r"\mathbb{N}",
    r"\mathbb{P}",
    r"\mathbb{Q}",
    r"\mathbb{R}",
    r"\mathbb{T}",
    r"\mathbb{Z}",
)
_SETS: tuple[str, ...] = ("A", "B", "C", "S", "T", "U")
_PROPS: tuple[str, ...] = ("P", "Q", "R")
# Bold Latin and Greek vector/tensor names
_BOLD_VECS: tuple[str, ...] = (
    r"\mathbf{a}",
    r"\mathbf{b}",
    r"\mathbf{e}",
    r"\mathbf{f}",
    r"\mathbf{n}",
    r"\mathbf{r}",
    r"\mathbf{u}",
    r"\mathbf{v}",
    r"\mathbf{w}",
    r"\mathbf{x}",
    r"\mathbf{y}",
    r"\mathbf{z}",
)
_BOLD_GREEK: tuple[str, ...] = (
    r"\boldsymbol{\alpha}",
    r"\boldsymbol{\beta}",
    r"\boldsymbol{\gamma}",
    r"\boldsymbol{\delta}",
    r"\boldsymbol{\lambda}",
    r"\boldsymbol{\mu}",
    r"\boldsymbol{\omega}",
    r"\boldsymbol{\phi}",
    r"\boldsymbol{\psi}",
    r"\boldsymbol{\sigma}",
    r"\boldsymbol{\theta}",
    r"\boldsymbol{\xi}",
)
_RELATIONS: tuple[str, ...] = (
    r"\sim",
    r"\cong",
    r"\simeq",
    r"\approx",
    r"\propto",
    r"\ll",
    r"\gg",
    r"\preceq",
    r"\succeq",
    r"\parallel",
    r"\perp",
)

# ---------------------------------------------------------------------------
# Shared statistical / probability pools
# ---------------------------------------------------------------------------

_RV_BASE: tuple[str, ...] = ("X", "Y", "Z", "W", "U", "V")
_STATS_N: tuple[str, ...] = ("n", "m", "N", "M")

# Generic combinatorics / summation index pools — reusable across domains
_COMB_N: tuple[str, ...] = ("n", "m", "N", "M", "p", "q", "r", "s", "t", "i", "l", r"n_0")
_COMB_K: tuple[str, ...] = ("k", "r", "j", "l", "i", "s", "t", "p")
_LAM_STATS: tuple[str, ...] = (r"\lambda", r"\mu", r"\nu", r"\alpha", r"\beta")
_MU_STATS: tuple[str, ...] = (r"\mu", r"\mu_0", r"\nu", "m")
_SIG_STATS: tuple[str, ...] = (r"\sigma", r"\sigma_0", r"\tau", r"\eta")
_EXP_OP: tuple[str, ...] = ("E", r"\mathbb{E}", r"\mathrm{E}", r"\mathbf{E}", r"\hat{E}", r"\mathbb{E}_\theta")
_PROB_OP_FULL: tuple[str, ...] = ("P", r"\mathbb{P}", r"\Pr", r"\mathbf{P}", r"\hat{P}", r"\tilde{P}")

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


def _eps_sub(rng: random.Random) -> str:
    return rng.choice((r"\epsilon", r"\varepsilon"))


def _tol_sub(rng: random.Random) -> str:
    return rng.choice((r"\epsilon", r"\varepsilon", r"\delta"))


def _cal(rng: random.Random) -> str:
    return rng.choice(_CALLIGRAPHIC)


def _bvec(rng: random.Random) -> str:
    return rng.choice(_BOLD_VECS)


def _bgreek(rng: random.Random) -> str:
    return rng.choice(_BOLD_GREEK)


_DECO_CMDS: tuple[str, ...] = (
    r"\hat",
    r"\bar",
    r"\tilde",
    r"\vec",
    r"\widehat",
    r"\widetilde",
    r"\check",
    r"\overline",
    r"\dot",
    r"\mathring",
)
_DECO_WEIGHTS = [22, 18, 14, 12, 8, 8, 6, 6, 4, 2]


def _deco(rng: random.Random) -> str:
    cmd = rng.choices(_DECO_CMDS, weights=_DECO_WEIGHTS, k=1)[0]
    target = rng.choice(_VARS + tuple("abcfghpqrs"))
    return rf"{cmd}{{{target}}}"


def _prime_deco(rng: random.Random, base: str) -> str:
    return rng.choice(
        (
            f"{base}'",
            f"{base}''",
            rf"{base}^{{\prime}}",
            rf"{base}^{{\prime\prime}}",
        )
    )


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
        lambda: _bvec(rng),
        lambda: _bgreek(rng),
        lambda: _prime_deco(rng, _v(rng)),
        lambda: _prime_deco(rng, rng.choice(("f", "g", "h", "F", "G"))),
    ]
    return rng.choice(builders)()


def _expr(rng: random.Random, depth: int = 2) -> str:
    """Compositional expression builder with bounded depth."""
    if depth <= 0:
        return _atom(rng)
    r = rng.random()
    if r < 0.20:
        return _atom(rng)
    exp = rng.choice(("2", "3", "n", r"\alpha"))
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
        n = rng.choice(("n", "m", "N"))
        k = rng.choice(("k", "r", "j"))
        return rf"\binom{{{n}}}{{{k}}}"
    if r < 0.88:
        op = rng.choice((r"\max", r"\min", r"\sup", r"\inf"))
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
        return rf"{var}_{{{rng.choice(('0', '1', '2', 'i', 'j', 'k', 'n', 'm'))}}}"
    return var


def _two(rng: random.Random, pool: tuple[str, ...]) -> tuple[str, str]:
    a, b = rng.sample(pool, 2)
    return a, b


def _lower(rng: random.Random) -> str:
    return rng.choice(_BOUNDS + (r"-\infty",))


def _upper(rng: random.Random) -> str:
    return rng.choice(_BOUNDS + (r"\infty", r"+\infty"))


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
    r"\mathcal{F}",
    r"\mathcal{G}",
    r"\mathcal{H}",
    r"\mathcal{L}",
)
_FN_IDX: tuple[str, ...] = ("1", "2", "3", "i", "j", "k", "n", "m")


def _fn_rich(rng: random.Random) -> str:
    """Rich function-name token: base optionally decorated with subscript/hat/tilde/bar/star."""
    base = rng.choice(_FN_BASE)
    d = rng.random()
    if d < 0.30:
        return f"{base}_{{{rng.choice(_FN_IDX)}}}"
    elif d < 0.44:
        return rf"\hat{{{base}}}"
    elif d < 0.56:
        return rf"\tilde{{{base}}}"
    elif d < 0.66:
        return rf"\bar{{{base}}}"
    elif d < 0.76:
        return f"{base}^{{*}}"
    elif d < 0.82:
        return rf"\check{{{base}}}"
    else:
        return base


def _fn_rich_nosub(rng: random.Random) -> str:
    """Like _fn_rich but no subscripts or superscript decorations — safe where the template
    appends its own ^{(n)}, ', or '' to the function name."""
    base = rng.choice(_FN_BASE)
    d = rng.random()
    if d < 0.28:
        return rf"\hat{{{base}}}"
    elif d < 0.50:
        return rf"\tilde{{{base}}}"
    elif d < 0.66:
        return rf"\bar{{{base}}}"
    elif d < 0.76:
        return rf"\check{{{base}}}"
    else:
        return base


# ---------------------------------------------------------------------------
# Trig function-name pools and selectors (reusable across algebra, calculus, …)
# ---------------------------------------------------------------------------

_TRIG_NAME_POOLS: dict[str, tuple[str, ...]] = {
    # Forward trig
    "sin": (r"\sin", r"\text{sine}"),
    "cos": (r"\cos", r"\text{cosine}"),
    "tan": (r"\tan", r"\text{tangent}"),
    "sec": (r"\sec", r"\text{secant}"),
    "csc": (r"\csc", r"\text{cosecant}"),
    "cot": (r"\cot", r"\text{cotangent}"),
    # Hyperbolic (sh/ch/th = Russian/European shorthand)
    "sinh": (r"\sinh", r"\text{sh}"),
    "cosh": (r"\cosh", r"\text{ch}"),
    "tanh": (r"\tanh", r"\text{th}"),
    # Inverse — 4 visually distinct forms each
    "arcsin": (r"\arcsin", r"\sin^{-1}", r"\text{asin}", r"\text{Arcsin}"),
    "arccos": (r"\arccos", r"\cos^{-1}", r"\text{acos}", r"\text{Arccos}"),
    "arctan": (r"\arctan", r"\tan^{-1}", r"\text{atan}", r"\text{Arctan}"),
    "arccot": (r"\text{arccot}", r"\cot^{-1}", r"\text{acot}"),
    "arcsec": (r"\text{arcsec}", r"\sec^{-1}", r"\text{asec}"),
    "arccsc": (r"\text{arccsc}", r"\csc^{-1}", r"\text{acsc}"),
    "arctanh": (r"\text{arctanh}", r"\tanh^{-1}", r"\text{atanh}"),
    "arcsinh": (r"\text{arcsinh}", r"\sinh^{-1}", r"\text{asinh}"),
    "arccosh": (r"\text{arccosh}", r"\cosh^{-1}", r"\text{acosh}"),
}

# Individual pools — aliases into _TRIG_NAME_POOLS for direct use as slot pools
_SIN_NAMES = _TRIG_NAME_POOLS["sin"]
_COS_NAMES = _TRIG_NAME_POOLS["cos"]
_TAN_NAMES = _TRIG_NAME_POOLS["tan"]
_SEC_NAMES = _TRIG_NAME_POOLS["sec"]
_CSC_NAMES = _TRIG_NAME_POOLS["csc"]
_COT_NAMES = _TRIG_NAME_POOLS["cot"]
_SINH_NAMES = _TRIG_NAME_POOLS["sinh"]
_COSH_NAMES = _TRIG_NAME_POOLS["cosh"]
_TANH_NAMES = _TRIG_NAME_POOLS["tanh"]
_ARCSIN_NAMES = _TRIG_NAME_POOLS["arcsin"]
_ARCCOS_NAMES = _TRIG_NAME_POOLS["arccos"]
_ARCTAN_NAMES = _TRIG_NAME_POOLS["arctan"]
_ARCCOT_NAMES = _TRIG_NAME_POOLS["arccot"]
_ARCSEC_NAMES = _TRIG_NAME_POOLS["arcsec"]
_ARCCSC_NAMES = _TRIG_NAME_POOLS["arccsc"]
_ARCTANH_NAMES = _TRIG_NAME_POOLS["arctanh"]
_ARCSINH_NAMES = _TRIG_NAME_POOLS["arcsinh"]
_ARCCOSH_NAMES = _TRIG_NAME_POOLS["arccosh"]

# Flat pool of common trig function LaTeX commands (forward + hyperbolic; primary forms only).
# Used by domains that need a random trig function name in integrals / identities.
_TRIG_INT_FN: tuple[str, ...] = (
    r"\sin",
    r"\cos",
    r"\tan",
    r"\sinh",
    r"\cosh",
    r"\sec",
    r"\text{sine}",
    r"\text{cosine}",
    r"\text{tangent}",
    r"\text{sh}",
    r"\text{ch}",
)

# Pool for Fourier series degree / harmonic index
_FOURIER_N: tuple[str, ...] = ("2", "3", "4", "5", "6", "n", "m", "N", "M", "K", "p")


def _trig_nm_factory(key: str) -> Callable[[random.Random], str]:
    pool = _TRIG_NAME_POOLS[key]
    return lambda rng: rng.choice(pool)


# Single-item trig-name samplers — one-liner shims generated from the dict above
_sin_nm = _trig_nm_factory("sin")
_cos_nm = _trig_nm_factory("cos")
_tan_nm = _trig_nm_factory("tan")
_sec_nm = _trig_nm_factory("sec")
_csc_nm = _trig_nm_factory("csc")
_cot_nm = _trig_nm_factory("cot")
_sinh_nm = _trig_nm_factory("sinh")
_cosh_nm = _trig_nm_factory("cosh")
_tanh_nm = _trig_nm_factory("tanh")
_arcsin_nm = _trig_nm_factory("arcsin")
_arccos_nm = _trig_nm_factory("arccos")
_arctan_nm = _trig_nm_factory("arctan")
_arccot_nm = _trig_nm_factory("arccot")
_arcsec_nm = _trig_nm_factory("arcsec")
_arccsc_nm = _trig_nm_factory("arccsc")
_arctanh_nm = _trig_nm_factory("arctanh")
_arcsinh_nm = _trig_nm_factory("arcsinh")
_arccosh_nm = _trig_nm_factory("arccosh")
