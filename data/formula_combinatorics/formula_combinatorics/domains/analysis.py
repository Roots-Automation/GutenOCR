"""Real analysis domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import E, S, Template, compute_weights, make_dispatcher
from .._vocab import (
    _GEO_N,
    _VARS,
    _atom,
    _eps_sub,
    _fn_rich_nosub,
    _tol_sub,
)

# ---------------------------------------------------------------------------
# Real analysis templates
# ---------------------------------------------------------------------------

_ANALYSIS_TEMPLATES: list[Template] = [
    # --- Epsilon-delta definitions (limit, continuity, uniform continuity) ---
    Template(
        name="epsilon_delta_limit",
        latex="",
        slots={},
        variants=[
            Template(
                name="limit_def",
                latex=(
                    r"\forall {eps} > 0 \; \exists {tol} > 0 : "
                    r"|{x} - {a}| < {tol} \Rightarrow |{f}({x}) - {L}| < {eps}"
                ),
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "x": E(_atom, n=150),
                    "a": E(_atom, n=150),
                    "L": E(_atom, n=150),
                    "eps": E(_eps_sub, n=2),
                    "tol": E(_tol_sub, n=3),
                },
            ),
            Template(
                name="continuity_def",
                latex=(
                    r"\forall {eps} > 0 \; \exists {tol} > 0 : "
                    r"|{x} - {c}| < {tol} \Rightarrow |{f}({x}) - {f}({c})| < {eps}"
                ),
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "x": E(_atom, n=150),
                    "c": E(_atom, n=150),
                    "eps": E(_eps_sub, n=2),
                    "tol": E(_tol_sub, n=3),
                },
            ),
            Template(
                name="uniform_continuity_def",
                latex=(
                    r"\forall {eps} > 0 \; \exists {tol} > 0 \; \forall {x}, {y} : "
                    r"|{x} - {y}| < {tol} \Rightarrow |{f}({x}) - {f}({y})| < {eps}"
                ),
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "x": E(_atom, n=150),
                    "y": E(_atom, n=150),
                    "eps": E(_eps_sub, n=2),
                    "tol": E(_tol_sub, n=3),
                },
            ),
        ],
    ),
    # --- L^p norm ---
    Template(
        name="lp_norm",
        latex=r"\|{f}\|_{{{p}}} = \left(\int \left|{f}({t})\right|^{{{p}}} d{t}\right)^{{1/{p}}}",
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "p": S(("p", "2", "q", "r")),
            "t": E(_atom, n=150),
        },
    ),
    # --- Cauchy sequence criterion ---
    Template(
        name="cauchy_criterion",
        latex=r"|{seq}_m - {seq}_n| < {eps} \quad \forall m, n \geq {N}",
        slots={
            "seq": S(("x", "a", "y", "z", "u", "v")),
            "eps": E(_eps_sub, n=2),
            "N": E(_atom, n=150),
        },
    ),
    # --- Cauchy-Schwarz for sums ---
    Template(
        name="cauchy_schwarz_sums",
        latex=(
            r"\left|\sum_{{i=1}}^{{{n}}} {a}_i {b}_i\right|^2 \leq "
            r"\sum_{{i=1}}^{{{n}}} {a}_i^2 \cdot \sum_{{i=1}}^{{{n}}} {b}_i^2"
        ),
        slots={
            "n": S(("n", "N", "m", "M", "K", "L", "P")),
            "a": S(("a", "u", "p", "x", "c")),
            "b": S(("b", "v", "q", "y", "d")),
        },
        distinct=[["a", "b"]],
    ),
    # --- Triangle inequality (scalar, reverse, sum, Minkowski) ---
    Template(
        name="triangle_inequality",
        latex="",
        slots={},
        variants=[
            Template(
                name="triangle_scalar",
                latex=r"\left\|{u} + {v}\right\| \leq \left\|{u}\right\| + \left\|{v}\right\|",
                slots={"u": E(_fn_rich_nosub, n=100), "v": E(_fn_rich_nosub, n=100)},
            ),
            Template(
                name="triangle_reverse",
                latex=r"\left| \left\|{u}\right\| - \left\|{v}\right\| \right| \leq \left\|{u} - {v}\right\|",
                slots={"u": E(_fn_rich_nosub, n=100), "v": E(_fn_rich_nosub, n=100)},
            ),
            Template(
                name="triangle_sum_n",
                latex=(
                    r"\left\|\sum_{{k=1}}^{{{n}}} {u}_k\right\| "
                    r"\leq \sum_{{k=1}}^{{{n}}} \left\|{u}_k\right\|"
                ),
                slots={"u": E(_fn_rich_nosub, n=100), "n": S(tuple(_GEO_N))},
            ),
            Template(
                name="minkowski_integral",
                latex=(
                    r"\left(\int \left|{f}({x}) + {g}({x})\right|^{{{p}}} d{x}\right)^{{1/{p}}} "
                    r"\leq \left(\int |{f}({x})|^{{{p}}} d{x}\right)^{{1/{p}}} "
                    r"+ \left(\int |{g}({x})|^{{{p}}} d{x}\right)^{{1/{p}}}"
                ),
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "g": E(_fn_rich_nosub, n=100),
                    "x": E(_atom, n=150),
                    "p": S(("p", "2", "q", "r")),
                },
            ),
        ],
    ),
    # --- Big-O / little-o / Theta asymptotics ---
    Template(
        name="big_o_asymptotic",
        latex="",
        slots={},
        variants=[
            Template(
                name="big_o",
                latex=r"{f}({v}) = O\!\left({g}({v})\right) \text{{ as }} {v} \to \infty",
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "g": E(_fn_rich_nosub, n=100),
                    "v": S(tuple(_GEO_N)),
                },
            ),
            Template(
                name="little_o",
                latex=r"{f}({v}) = o\!\left({g}({v})\right) \text{{ as }} {v} \to \infty",
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "g": E(_fn_rich_nosub, n=100),
                    "v": S(tuple(_GEO_N)),
                },
            ),
            Template(
                name="big_theta",
                latex=r"{f}({v}) = \Theta\!\left({g}({v})\right) \text{{ as }} {v} \to \infty",
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "g": E(_fn_rich_nosub, n=100),
                    "v": S(tuple(_GEO_N)),
                },
            ),
        ],
    ),
    # --- p-series convergence ---
    Template(
        name="p_series_convergence",
        latex=r"\sum_{{k=1}}^{{\infty}} \frac{{1}}{{k^{{{s}}}}} < \infty \iff {s} > 1",
        slots={"s": S(("p", "2", "3", "q", "r", "4", "5", r"\alpha", r"\beta"))},
    ),
    # --- Hölder's inequality (with variable exponent slots) ---
    Template(
        name="holder_inequality",
        latex=(
            r"\int |{f}({x}) \cdot {g}({x})| \, d{x} \leq "
            r"\left(\int |{f}({x})|^{{{p}}}\, d{x}\right)^{{1/{p}}} "
            r"\left(\int |{g}({x})|^{{{q}}}\, d{x}\right)^{{1/{q}}}"
        ),
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "g": E(_fn_rich_nosub, n=100),
            "x": E(_atom, n=150),
            "p": S(("p", "2", "r", "s", "3")),
            "q": S(("q", "2", "t", "r", "4")),
        },
    ),
    # --- Banach contraction mapping (with x/y variable slots) ---
    Template(
        name="banach_contraction",
        latex=r"\|T({x}) - T({y})\| \leq {c_sym} \|{x} - {y}\|",
        slots={
            "x": S(tuple(_VARS), idx=0.35),
            "y": S(tuple(_VARS), idx=0.35),
            "c_sym": S((r"\lambda", "k", "c", r"\kappa", r"\rho", r"\alpha", "L")),
        },
        distinct=[["x", "y"]],
    ),
    # --- Uniform supremum bound ---
    Template(
        name="uniform_bound",
        latex=r"\sup_{{{x} \in {D}}} |{f}({x})| < \infty",
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "x": E(_atom, n=150),
            "D": S(("X", "D", "A", r"\Omega", "K", "U")),
        },
    ),
    # --- Limsup / liminf / squeeze theorem ---
    Template(
        name="limsup_definition",
        latex="",
        slots={},
        variants=[
            Template(
                name="limsup_def",
                latex=(
                    r"\limsup_{{{idx} \to {inf}}} {seq}_{{{idx}}} "
                    r"= \inf_{{{idx} \geq 1}} \sup_{{k \geq {idx}}} {seq}_k"
                ),
                slots={
                    "seq": S(("a", "b", "x", "y", "u", "c", "z", "v"), idx=0.35),
                    "idx": S(("n", "m", "j", "r"), idx=0.25),
                    "inf": S((r"\infty", r"+\infty")),
                },
            ),
            Template(
                name="liminf_def",
                latex=(
                    r"\liminf_{{{idx} \to {inf}}} {seq}_{{{idx}}} "
                    r"= \sup_{{{idx} \geq 1}} \inf_{{k \geq {idx}}} {seq}_k"
                ),
                slots={
                    "seq": S(("a", "b", "x", "y", "u", "c", "z", "v"), idx=0.35),
                    "idx": S(("n", "m", "j", "r"), idx=0.25),
                    "inf": S((r"\infty", r"+\infty")),
                },
            ),
            Template(
                name="squeeze_theorem",
                latex=(
                    r"{a}_{{{idx}}} \leq {b}_{{{idx}}} \leq {c}_{{{idx}}}, \quad "
                    r"\lim_{{{idx} \to \infty}} {a}_{{{idx}}} = \lim_{{{idx} \to \infty}} {c}_{{{idx}}} = L "
                    r"\implies \lim_{{{idx} \to \infty}} {b}_{{{idx}}} = L"
                ),
                slots={
                    "a": S(("a", "x", "u")),
                    "b": S(("b", "y", "v")),
                    "c": S(("c", "z", "w")),
                    "idx": S(("n", "m", "j", "k")),
                },
                distinct=[["a", "b", "c"]],
            ),
        ],
    ),
    # --- Limit notation (finite point, one-sided, ±∞, sequence, arithmetic) ---
    Template(
        name="limit_notation",
        latex="",
        slots={},
        variants=[
            Template(
                name="limit_at_point",
                latex=r"\lim_{{{x} \to {a}}} {f}({x}) = {L}",
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "x": E(_atom, n=150),
                    "a": E(_atom, n=150),
                    "L": E(_atom, n=150),
                },
            ),
            Template(
                name="limit_right_sided",
                latex=r"\lim_{{{x} \to {a}^+}} {f}({x}) = {L}",
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "x": E(_atom, n=150),
                    "a": E(_atom, n=150),
                    "L": E(_atom, n=150),
                },
            ),
            Template(
                name="limit_left_sided",
                latex=r"\lim_{{{x} \to {a}^-}} {f}({x}) = {L}",
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "x": E(_atom, n=150),
                    "a": E(_atom, n=150),
                    "L": E(_atom, n=150),
                },
            ),
            Template(
                name="limit_pos_infinity",
                latex=r"\lim_{{{x} \to +\infty}} {f}({x}) = {L}",
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "x": E(_atom, n=150),
                    "L": E(_atom, n=150),
                },
            ),
            Template(
                name="limit_neg_infinity",
                latex=r"\lim_{{{x} \to -\infty}} {f}({x}) = {L}",
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "x": E(_atom, n=150),
                    "L": E(_atom, n=150),
                },
            ),
            Template(
                name="limit_sequence",
                latex=r"\lim_{{{idx} \to \infty}} {seq}_{{{idx}}} = {L}",
                slots={
                    "seq": S(("a", "b", "x", "y", "u", "c", "z", "v"), idx=0.35),
                    "idx": S(("n", "m", "j", "k"), idx=0.25),
                    "L": E(_atom, n=150),
                },
            ),
            Template(
                name="limit_newton_quotient",
                latex=r"\lim_{{{x} \to {a}}} \frac{{{f}({x}) - {f}({a})}}{{{x} - {a}}} = {L}",
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "x": E(_atom, n=150),
                    "a": E(_atom, n=150),
                    "L": E(_atom, n=150),
                },
            ),
            Template(
                name="limit_sum_rule",
                latex=(
                    r"\lim_{{{x} \to {a}}} \bigl({f}({x}) + {g}({x})\bigr) "
                    r"= \lim_{{{x} \to {a}}} {f}({x}) + \lim_{{{x} \to {a}}} {g}({x})"
                ),
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "g": E(_fn_rich_nosub, n=100),
                    "x": E(_atom, n=150),
                    "a": E(_atom, n=150),
                },
            ),
            Template(
                name="limit_product_rule",
                latex=(
                    r"\lim_{{{x} \to {a}}} {f}({x}) {g}({x}) "
                    r"= \lim_{{{x} \to {a}}} {f}({x}) \cdot \lim_{{{x} \to {a}}} {g}({x})"
                ),
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "g": E(_fn_rich_nosub, n=100),
                    "x": E(_atom, n=150),
                    "a": E(_atom, n=150),
                },
            ),
            Template(
                name="limit_diverges_inf",
                latex=r"\lim_{{{x} \to {a}}} {f}({x}) = {inf_sym}",
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "x": E(_atom, n=150),
                    "a": E(_atom, n=150),
                    "inf_sym": S((r"\infty", r"+\infty", r"-\infty")),
                },
            ),
        ],
    ),
    # --- Weierstrass M-test ---
    Template(
        name="weierstrass_m_test",
        latex=(
            r"\sum_{{n=1}}^\infty \left|{f}_n({x})\right| \leq {M}_n, \; "
            r"\sum_{{n=1}}^\infty {M}_n < \infty "
            r"\implies \sum_{{n=1}}^\infty {f}_n \text{{ converges uniformly}}"
        ),
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "x": E(_atom, n=150),
            "M": S(("M", "C", "K", "B", "A", "L")),
        },
    ),
    # --- Arzelà-Ascoli theorem ---
    Template(
        name="arzela_ascoli",
        latex=(
            r"\left\{{{f}_n({x})\right\}} \text{{ equicontinuous and uniformly bounded}}"
            r" \implies \text{{has uniformly convergent subsequence}}"
        ),
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "x": E(_atom, n=150),
        },
    ),
    # --- Banach fixed-point theorem ---
    Template(
        name="banach_fixed_point",
        latex=r"\exists!\, x^* : {T}(x^*) = x^*, \quad x_{{n+1}} = {T}(x_n) \to x^*",
        slots={"T": E(_fn_rich_nosub, n=100)},
    ),
    # --- Series convergence tests (rich parametric variants) ---
    Template(
        name="convergence_tests",
        latex="",
        slots={},
        variants=[
            Template(
                name="comparison_test",
                latex=(
                    r"0 \leq {a}_n \leq {b}_n, \; \sum {b}_n < \infty "
                    r"\implies \sum {a}_n < \infty"
                ),
                slots={
                    "a": S(("a", "x", "u", "p", "c")),
                    "b": S(("b", "y", "v", "q", "d")),
                },
                distinct=[["a", "b"]],
            ),
            Template(
                name="integral_test",
                latex=(
                    r"\sum_{{n=1}}^\infty {f}(n) \text{{ converges}} "
                    r"\iff \int_1^\infty {f}({x})\,d{x} < \infty"
                ),
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "x": E(_atom, n=150),
                },
            ),
            Template(
                name="limit_comparison_test",
                latex=(
                    r"\lim_{{{v} \to \infty}} \frac{{{a}_{{{v}}}}}{{{b}_{{{v}}}}} = {L} \in (0, \infty) "
                    r"\implies \sum {a}_n \text{{ and }} \sum {b}_n \text{{ same convergence}}"
                ),
                slots={
                    "a": S(("a", "x", "u", "p")),
                    "b": S(("b", "y", "v", "q")),
                    "v": S(("n", "m", "k", "j")),
                    "L": E(_atom, n=150),
                },
                distinct=[["a", "b"]],
            ),
        ],
    ),
    # --- Ratio and root tests (standalone — inherently low n_eff, kept separate
    #     so compute_weights assigns them proportionally fewer draws) ---
    Template(
        name="ratio_test",
        latex=(
            r"\lim_{{{v} \to \infty}} \left|\frac{{{seq}_{{{v}+1}}}}{{{seq}_{{{v}}}}}\right| = {L} < 1 "
            r"\implies \sum {seq}_n \text{{ converges absolutely}}"
        ),
        slots={
            "seq": S(("a", "b", "c", "x", "y", "u", "v", "p")),
            "v": S(("n", "m", "k", "j")),
            "L": E(_atom, n=150),
        },
    ),
    Template(
        name="root_test",
        latex=(
            r"\limsup_{{{v} \to \infty}} \left|{seq}_{{{v}}}\right|^{{1/{v}}} = {L} < 1 "
            r"\implies \sum {seq}_n \text{{ converges absolutely}}"
        ),
        slots={
            "seq": S(("a", "b", "c", "x", "y", "u", "v", "p")),
            "v": S(("n", "m", "k", "j")),
            "L": E(_atom, n=150),
        },
    ),
    Template(
        name="alternating_series",
        latex=(
            r"{a}_{{{v}}} \searrow 0 \implies "
            r"\sum_{{{v}={v0}}}^\infty (-1)^{{{v}}} {a}_{{{v}}} \text{{ converges}}"
        ),
        slots={
            "a": S(("a", "b", "c", "x", "y", "u", "v", "p")),
            "v": S(("n", "m", "k", "j")),
            "v0": S(("1", "0", "2")),
        },
    ),
    # --- Uniform / pointwise convergence ---
    Template(
        name="uniform_convergence",
        latex="",
        slots={},
        variants=[
            Template(
                name="pointwise_conv",
                latex=r"{f}_n({x}) \to {f}({x}) \quad \text{{for all }} {x} \in D",
                slots={"f": E(_fn_rich_nosub, n=100), "x": E(_atom, n=150)},
            ),
            Template(
                name="uniform_conv_def",
                latex=r"\sup_{{{x} \in D}} \left|{f}_n({x}) - {f}({x})\right| \to 0",
                slots={"f": E(_fn_rich_nosub, n=100), "x": E(_atom, n=150)},
            ),
            Template(
                name="uniform_implies_continuous",
                latex=(
                    r"{f}_n \rightrightarrows {f} \text{{ on }} {D}, \; "
                    r"{f}_n \text{{ continuous}} \implies {f} \text{{ continuous}}"
                ),
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "D": S(("D", "X", "K", r"\Omega", "A", "U")),
                },
            ),
            Template(
                name="uniform_implies_integrable",
                latex=(
                    r"{f}_n \rightrightarrows {f} \text{{ on }} [{a}, {b}] "
                    r"\implies \int_{{{a}}}^{{{b}}} {f}_n \to \int_{{{a}}}^{{{b}}} {f}"
                ),
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "a": E(_atom, n=150),
                    "b": E(_atom, n=150),
                },
            ),
        ],
    ),
    # --- Fundamental theorems (IVT, MVT, EVT) ---
    Template(
        name="fundamental_theorems",
        latex="",
        slots={},
        variants=[
            Template(
                name="ivt",
                latex=(
                    r"{f} \in C([{a}, {b}]), \; {f}({a}) {f}({b}) < 0 "
                    r"\implies \exists\, c \in ({a}, {b}) : {f}(c) = 0"
                ),
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "a": E(_atom, n=150),
                    "b": E(_atom, n=150),
                },
            ),
            Template(
                name="mvt",
                latex=(
                    r"\exists\, c \in ({a}, {b}) : {f}'(c) = "
                    r"\dfrac{{{f}({b}) - {f}({a})}}{{{b} - {a}}}"
                ),
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "a": E(_atom, n=150),
                    "b": E(_atom, n=150),
                },
            ),
            Template(
                name="extreme_value",
                latex=(
                    r"{f} \in C([{a}, {b}]) \implies "
                    r"{f} \text{{ attains its maximum and minimum on }} [{a}, {b}]"
                ),
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "a": E(_atom, n=150),
                    "b": E(_atom, n=150),
                },
            ),
        ],
    ),
    # --- Bolzano-Weierstrass (standalone — low n_eff, gets proportionally few draws) ---
    Template(
        name="bolzano_weierstrass",
        latex=(
            r"\text{{Every bounded sequence }} \left\{{{seq}_k\right\}} \text{{ in }} \mathbb{{R}}^{{{d}}} "
            r"\text{{ has a convergent subsequence}}"
        ),
        slots={
            "seq": S(("x", "a", "y", "z", "u", "v"), idx=0.35),
            "d": S(("1", "2", "3", "n", "d", "m", "N", "k")),
        },
    ),
    # --- Taylor remainder bound ---
    Template(
        name="taylor_remainder",
        latex=(
            r"\left|{f}({x}) - \sum_{{k=0}}^{{{n}}} \frac{{{f}^{{(k)}}({a})}}{{k!}}({x}-{a})^k\right|"
            r" \leq \frac{{M}}{{({n}+1)!}} \left|{x} - {a}\right|^{{{n}+1}}"
        ),
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "x": E(_atom, n=150),
            "a": E(_atom, n=150),
            "n": S(("n", "m", "N", "p")),
        },
    ),
    # --- Completeness ---
    Template(
        name="completeness",
        latex="",
        slots={},
        variants=[
            Template(
                name="metric_completeness",
                latex=(
                    r"({spc}, d) \text{{ is complete}} \iff "
                    r"\text{{every Cauchy sequence in }} {spc} \text{{ converges in }} {spc}"
                ),
                slots={"spc": S(("X", "M", "Y", "Z", r"\mathcal{X}", r"\mathcal{M}", r"\mathcal{Y}"))},
            ),
            Template(
                name="nested_intervals",
                latex=(
                    r"[{seq}_n, {seq}_n'] \supseteq [{seq}_{{n+1}}, {seq}_{{n+1}}'], \;"
                    r"\; {seq}_n' - {seq}_n \to 0 "
                    r"\implies \bigcap_n [{seq}_n, {seq}_n'] \neq \emptyset"
                ),
                slots={"seq": S(("a", "x", "u", "b", "c", "y", "z"), idx=0.35)},
            ),
        ],
    ),
]

# ---------------------------------------------------------------------------
# Sampling weights — cap at 10_000 to balance high-n_eff templates
# ---------------------------------------------------------------------------

_W_ANALYSIS: list[float] = compute_weights(_ANALYSIS_TEMPLATES, cap=10_000)

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
