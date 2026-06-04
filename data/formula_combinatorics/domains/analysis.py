"""Real analysis domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import E, S, Template, compute_weights, make_dispatcher

# ---------------------------------------------------------------------------
# Inline sub-generators
# ---------------------------------------------------------------------------


def _p_norm(rng: random.Random) -> str:
    return rng.choice(["p", "2", r"\infty"])


def _other_func(rng: random.Random) -> str:
    return rng.choice(["g", "h", "u"])


def _s_exp(rng: random.Random) -> str:
    return rng.choice(["2", "p", "s"])


def _c_sym(rng: random.Random) -> str:
    return rng.choice([r"\lambda", "k", "c"])


# ---------------------------------------------------------------------------
# Real analysis templates
# ---------------------------------------------------------------------------

_ANALYSIS_TEMPLATES: list[Template] = [
    # c=0 — epsilon-delta definition of limit
    Template(
        name="epsilon_delta_limit",
        latex=(
            r"\forall \epsilon > 0 \; \exists \delta > 0 : "
            r"|x - a| < \delta \Rightarrow |{f}(x) - L| < \epsilon"
        ),
        slots={"f": S(("f", "u", "g"))},
    ),
    # c=1 — L^p norm
    Template(
        name="lp_norm",
        latex=r"\|{f}\|_{{{p}}} = \left(\int \left|{f}(t)\right|^{{{p}}} dt\right)^{{1/{p}}}",
        slots={
            "f": S(("f", "u", "g")),
            "p": E(_p_norm, n=3),
        },
    ),
    # c=2 — Cauchy sequence criterion (fixed)
    Template(
        name="cauchy_criterion",
        latex=r"|x_m - x_n| < \epsilon \quad \forall m, n \geq N",
        slots={},
    ),
    # c=3 — Cauchy-Schwarz for sums
    Template(
        name="cauchy_schwarz_sums",
        latex=(
            r"\left|\sum_{{i=1}}^{{{n}}} a_i b_i\right|^2 \leq "
            r"\sum_{{i=1}}^{{{n}}} a_i^2 \cdot \sum_{{i=1}}^{{{n}}} b_i^2"
        ),
        slots={"n": S(("n", "N", "m"))},
    ),
    # c=4 — triangle inequality (fixed)
    Template(
        name="triangle_inequality",
        latex=r"\|u + v\| \leq \|u\| + \|v\|",
        slots={},
    ),
    # c=5 — big-O asymptotics
    Template(
        name="big_o_asymptotic",
        latex=r"{f}(n) = O\!\left({g}(n)\right) \text{{ as }} n \to \infty",
        slots={
            "f": S(("f", "u", "g")),
            "g": E(_other_func, n=3),
        },
    ),
    # c=6 — p-series convergence
    Template(
        name="p_series_convergence",
        latex=r"\sum_{{k=1}}^{{\infty}} \frac{{1}}{{k^{{{s}}}}} < \infty",
        slots={"s": E(_s_exp, n=3)},
    ),
    # c=7 — Hölder's inequality
    Template(
        name="holder_inequality",
        latex=(
            r"\int |{f} {g}| \leq "
            r"\left(\int |{f}|^p\right)^{{1/p}} "
            r"\left(\int |{g}|^q\right)^{{1/q}}"
        ),
        slots={
            "f": S(("f", "u", "g")),
            "g": E(_other_func, n=3),
        },
    ),
    # c=8 — Banach contraction mapping
    Template(
        name="banach_contraction",
        latex=r"\|T(x) - T(y)\| \leq {c_sym} \|x - y\|",
        slots={"c_sym": E(_c_sym, n=3)},
    ),
    # c=9 — uniform supremum bound
    Template(
        name="uniform_bound",
        latex=r"\sup_{{x \in X}} |{f}(x)| < \infty",
        slots={"f": S(("f", "u", "g"))},
    ),
    # c=10 — limsup definition (fixed)
    Template(
        name="limsup_definition",
        latex=r"\limsup_{{n \to \infty}} a_n = \inf_{{n \geq 1}} \sup_{{k \geq n}} a_k",
        slots={},
    ),
    # c=11 — Weierstrass M-test
    Template(
        name="weierstrass_m_test",
        latex=r"\sum |{f}_n| \leq M_n, \; \sum M_n < \infty \implies \sum {f}_n \text{{ converges uniformly}}",
        slots={"f": S(("f", "u", "g"))},
    ),
    # c=12 — Arzelà-Ascoli theorem
    Template(
        name="arzela_ascoli",
        latex=(
            r"\left\{{{f}_n\right\}} \text{{ equicontinuous and uniformly bounded}}"
            r" \implies \text{{has convergent subsequence}}"
        ),
        slots={"f": S(("f", "u", "g"))},
    ),
    # c=13 — Banach fixed-point theorem (fixed)
    Template(
        name="banach_fixed_point",
        latex=r"\exists! x^* : T(x^*) = x^*, \quad x_{{n+1}} = T(x_n) \to x^*",
        slots={},
    ),
]

# ---------------------------------------------------------------------------
# Sampling weights
# ---------------------------------------------------------------------------

_W_ANALYSIS: list[float] = compute_weights(_ANALYSIS_TEMPLATES)

# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

_analysis = make_dispatcher(_ANALYSIS_TEMPLATES, _W_ANALYSIS)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "analysis": _analysis,
}

WEIGHTS: dict[str, float] = {
    "analysis": 0.04,
}

TEMPLATES: dict[str, list[Template]] = {
    "analysis": _ANALYSIS_TEMPLATES,
}
