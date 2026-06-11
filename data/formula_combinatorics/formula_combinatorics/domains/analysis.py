"""Real analysis and functional analysis domain generators."""

from __future__ import annotations

from .._template_dsl import _ATOM_SLOT, _EXPR_SLOT, _FN_SLOT, _LIM_MOD, E, S, Template, X
from .._vocab import (
    _GEO_N,
    _VARS,
    _eps_sub,
    _expr,
    _tol_sub,
)
from ._config import register_domain

# Bound-variable pool: proper letter variables (no digits, no calligraphic) with
# optional subscript decoration, used wherever a symbol is quantified over or
# acts as an integration / limit / function-argument dummy variable.
_BVAR = _VARS  # ("x","y","z","t","u","v","r","s")

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
                    "f": _FN_SLOT,
                    "x": S(_BVAR, idx=0.35),  # bound variable
                    "a": _ATOM_SLOT,  # limit point (any value)
                    "L": _ATOM_SLOT,  # limit value (any value)
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
                    "f": _FN_SLOT,
                    "x": S(_BVAR, idx=0.35),  # bound variable
                    "c": _ATOM_SLOT,  # fixed point (any value)
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
                    "f": _FN_SLOT,
                    "x": S(_BVAR, idx=0.35),  # bound variable
                    "y": S(_BVAR, idx=0.35),  # bound variable
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
            "f": _FN_SLOT,
            "p": S(("p", "2", "q", "r")),
            "t": S(_BVAR, idx=0.35),  # integration variable
        },
    ),
    # --- Cauchy sequence criterion ---
    Template(
        name="cauchy_criterion",
        latex=r"|{seq}_m - {seq}_n| < {eps} \quad \forall m, n \geq {N}",
        slots={
            "seq": S(("x", "a", "y", "z", "u", "v")),
            "eps": E(_eps_sub, n=2),
            "N": _ATOM_SLOT,  # bound value (any symbol)
        },
    ),
    # --- Cauchy-Schwarz for sums ---
    Template(
        name="cauchy_schwarz_sums",
        latex=(
            r"\left|\sum{lim_mod}_{{i=1}}^{{{n}}} {a}_i {b}_i\right|^2 \leq "
            r"\sum{lim_mod}_{{i=1}}^{{{n}}} {a}_i^2 \cdot \sum{lim_mod}_{{i=1}}^{{{n}}} {b}_i^2"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "n": S(("n", "N", "m", "M", "K", "L", "P")),
            "a": S(("a", "u", "p", "x", "c")),
            "b": S(("b", "v", "q", "y", "d")),
        },
        distinct=[["a", "b"]],
    ),
    # --- Triangle inequality in normed spaces (use algebra.triangle_inequality for the general form) ---
    Template(
        name="triangle_inequality_normed",
        latex="",
        slots={},
        variants=[
            Template(
                name="triangle_scalar",
                latex=r"\left\|{u} + {v}\right\| \leq \left\|{u}\right\| + \left\|{v}\right\|",
                slots={"u": _FN_SLOT, "v": _FN_SLOT},
            ),
            Template(
                name="triangle_reverse",
                latex=r"\left| \left\|{u}\right\| - \left\|{v}\right\| \right| \leq \left\|{u} - {v}\right\|",
                slots={"u": _FN_SLOT, "v": _FN_SLOT},
            ),
            Template(
                name="triangle_sum_n",
                latex=(
                    r"\left\|\sum{lim_mod}_{{k=1}}^{{{n}}} {u}_k\right\| "
                    r"\leq \sum{lim_mod}_{{k=1}}^{{{n}}} \left\|{u}_k\right\|"
                ),
                slots={"lim_mod": _LIM_MOD, "u": _FN_SLOT, "n": S(_GEO_N)},
            ),
            Template(
                name="minkowski_integral",
                latex=(
                    r"\left(\int \left|{f}({t}) + {g}({t})\right|^{{{p}}} d{t}\right)^{{1/{p}}} "
                    r"\leq \left(\int |{f}({t})|^{{{p}}} d{t}\right)^{{1/{p}}} "
                    r"+ \left(\int |{g}({t})|^{{{p}}} d{t}\right)^{{1/{p}}}"
                ),
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "t": S(_BVAR, idx=0.35),  # integration variable
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
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "v": S(_GEO_N),
                },
            ),
            Template(
                name="little_o",
                latex=r"{f}({v}) = o\!\left({g}({v})\right) \text{{ as }} {v} \to \infty",
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "v": S(_GEO_N),
                },
            ),
            Template(
                name="big_theta",
                latex=r"{f}({v}) = \Theta\!\left({g}({v})\right) \text{{ as }} {v} \to \infty",
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "v": S(_GEO_N),
                },
            ),
            Template(
                name="big_omega",
                latex=r"{f}({v}) = \Omega\!\left({g}({v})\right) \text{{ as }} {v} \to \infty",
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "v": S(_GEO_N),
                },
            ),
            Template(
                name="little_omega",
                latex=r"{f}({v}) = \omega\!\left({g}({v})\right) \text{{ as }} {v} \to \infty",
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "v": S(_GEO_N),
                },
            ),
            Template(
                name="asymptotic_equiv",
                latex=r"{f}({v}) \sim {g}({v}) \text{{ as }} {v} \to \infty",
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "v": S(_GEO_N),
                },
            ),
            Template(
                name="soft_o",
                latex=r"{f}({v}) = \tilde{{O}}\!\left({g}({v})\right) \text{{ as }} {v} \to \infty",
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "v": S(_GEO_N),
                },
            ),
        ],
    ),
    # --- Landau notation at finite / one-sided limit points ---
    Template(
        name="asymptotic_directional",
        latex="",
        slots={},
        variants=[
            Template(
                name="big_o_at",
                latex=r"{f}({v}) = O\!\left({g}({v})\right) \text{{ as }} {v} \to {lp}",
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "v": S(_BVAR),
                    "lp": S(("0", r"0^+", r"0^-", "1", "a", "b")),
                },
            ),
            Template(
                name="little_o_at",
                latex=r"{f}({v}) = o\!\left({g}({v})\right) \text{{ as }} {v} \to {lp}",
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "v": S(_BVAR),
                    "lp": S(("0", r"0^+", r"0^-", "1", "a", "b")),
                },
            ),
            Template(
                name="big_theta_at",
                latex=r"{f}({v}) = \Theta\!\left({g}({v})\right) \text{{ as }} {v} \to {lp}",
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "v": S(_BVAR),
                    "lp": S(("0", r"0^+", r"0^-", "1", "a", "b")),
                },
            ),
            Template(
                name="big_omega_at",
                latex=r"{f}({v}) = \Omega\!\left({g}({v})\right) \text{{ as }} {v} \to {lp}",
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "v": S(_BVAR),
                    "lp": S(("0", r"0^+", r"0^-", "1", "a", "b")),
                },
            ),
            Template(
                name="little_omega_at",
                latex=r"{f}({v}) = \omega\!\left({g}({v})\right) \text{{ as }} {v} \to {lp}",
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "v": S(_BVAR),
                    "lp": S(("0", r"0^+", r"0^-", "1", "a", "b")),
                },
            ),
            Template(
                name="asymptotic_equiv_at",
                latex=r"{f}({v}) \sim {g}({v}) \text{{ as }} {v} \to {lp}",
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "v": S(_BVAR),
                    "lp": S(("0", r"0^+", r"0^-", "1", "a", "b")),
                },
            ),
        ],
    ),
    # --- Asymptotic error-term / remainder form ---
    Template(
        name="asymptotic_error_term",
        latex="",
        slots={},
        variants=[
            Template(
                name="error_big_o",
                latex=r"{f}({v}) = {g}({v}) + O\!\left({h}({v})\right) \text{{ as }} {v} \to \infty",
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "h": _FN_SLOT,
                    "v": S(_GEO_N),
                },
            ),
            Template(
                name="error_little_o",
                latex=r"{f}({v}) = {g}({v}) + o\!\left({h}({v})\right) \text{{ as }} {v} \to \infty",
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "h": _FN_SLOT,
                    "v": S(_GEO_N),
                },
            ),
            Template(
                name="error_little_o_one",
                latex=r"{f}({v}) = {g}({v}) + o(1) \text{{ as }} {v} \to \infty",
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "v": S(_GEO_N),
                },
            ),
        ],
    ),
    # --- Formal definitions of Landau notation ---
    Template(
        name="asymptotic_definition",
        latex="",
        slots={},
        variants=[
            Template(
                name="big_o_existential",
                latex=(
                    r"\exists\, C > 0,\; {v}_0 :\;"
                    r"|{f}({v})| \leq C\,|{g}({v})|\;"
                    r"\text{{ for all }} {v} \geq {v}_0"
                ),
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "v": S(_GEO_N),
                },
            ),
            Template(
                name="little_o_limit",
                latex=r"\lim_{{{v} \to \infty}} \frac{{{f}({v})}}{{{g}({v})}} = 0",
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "v": S(_GEO_N),
                },
            ),
            Template(
                name="asymptotic_equiv_limit",
                latex=r"\lim_{{{v} \to \infty}} \frac{{{f}({v})}}{{{g}({v})}} = 1",
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "v": S(_GEO_N),
                },
            ),
            Template(
                name="big_theta_sandwich",
                latex=(
                    r"c_1\,{g}({v}) \leq {f}({v}) \leq c_2\,{g}({v})"
                    r"\text{{ for all large }} {v}"
                ),
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "v": S(_GEO_N),
                },
            ),
        ],
    ),
    # --- Poincaré asymptotic expansions ---
    Template(
        name="asymptotic_expansion",
        latex="",
        slots={},
        variants=[
            Template(
                name="poincare_series",
                latex=r"{f}({v}) \sim \sum{lim_mod}_{{{k}=0}}^{{\infty}} {a}_{{{k}}}\,{v}^{{-{k}}} \text{{ as }} {v} \to \infty",
                slots={
                    "lim_mod": _LIM_MOD,
                    "f": _FN_SLOT,
                    "v": S(_GEO_N),
                    "a": S(_BVAR),
                    "k": S(("k", "j", "m")),
                },
            ),
            Template(
                name="poincare_partial",
                latex=(
                    r"{f}({v}) = \sum{lim_mod}_{{{k}=0}}^{{{N}}}"
                    r" {a}_{{{k}}}\,{v}^{{-{k}}}"
                    r" + O\!\left({v}^{{-{N}-1}}\right)"
                ),
                slots={
                    "lim_mod": _LIM_MOD,
                    "f": _FN_SLOT,
                    "v": S(_GEO_N),
                    "a": S(_BVAR),
                    "k": S(("k", "j", "m")),
                    "N": S(("N", "n", "M", "K")),
                },
            ),
        ],
    ),
    # --- p-series convergence ---
    Template(
        name="p_series_convergence",
        latex=r"\sum{lim_mod}_{{k=1}}^{{\infty}} \frac{{1}}{{k^{{{s}}}}} < \infty \iff {s} > 1",
        slots={"lim_mod": _LIM_MOD, "s": S(("p", "2", "3", "q", "r", "4", "5", r"\alpha", r"\beta"))},
    ),
    # --- Hölder's inequality ---
    Template(
        name="holder_inequality",
        latex=(
            r"\int |{f}({t}) \cdot {g}({t})| \, d{t} \leq "
            r"\left(\int |{f}({t})|^{{{p}}}\, d{t}\right)^{{1/{p}}} "
            r"\left(\int |{g}({t})|^{{{q}}}\, d{t}\right)^{{1/{q}}}"
        ),
        slots={
            "f": _FN_SLOT,
            "g": _FN_SLOT,
            "t": S(_BVAR, idx=0.35),  # integration variable
            "p": S(("p", "2", "r", "s", "3")),
            "q": S(("q", "2", "t", "r", "4")),
        },
    ),
    # --- Banach contraction mapping ---
    Template(
        name="banach_contraction",
        latex=r"\|T({x}) - T({y})\| \leq {c_sym} \|{x} - {y}\|",
        slots={
            "x": S(_BVAR, idx=0.35),
            "y": S(_BVAR, idx=0.35),
            "c_sym": S((r"\lambda", "k", "c", r"\kappa", r"\rho", r"\alpha", "L")),
        },
        distinct=[["x", "y"]],
    ),
    # --- Uniform supremum bound ---
    Template(
        name="uniform_bound",
        latex=r"\sup_{{{x} \in {D}}} |{f}({x})| < \infty",
        slots={
            "f": _FN_SLOT,
            "x": S(_BVAR, idx=0.35),  # quantified variable
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
                    "seq": S(("a", "b", "x", "y", "u", "c", "z", "v")),
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
                    "seq": S(("a", "b", "x", "y", "u", "c", "z", "v")),
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
                    "f": _FN_SLOT,
                    "x": S(_BVAR, idx=0.35),  # limit variable
                    "a": _ATOM_SLOT,  # limit point (any value)
                    "L": _ATOM_SLOT,  # limit value (any value)
                },
            ),
            Template(
                name="limit_right_sided",
                latex=r"\lim_{{{x} \to {a}^+}} {f}({x}) = {L}",
                slots={
                    "f": _FN_SLOT,
                    "x": S(_BVAR, idx=0.35),
                    "a": _ATOM_SLOT,
                    "L": _ATOM_SLOT,
                },
            ),
            Template(
                name="limit_left_sided",
                latex=r"\lim_{{{x} \to {a}^-}} {f}({x}) = {L}",
                slots={
                    "f": _FN_SLOT,
                    "x": S(_BVAR, idx=0.35),
                    "a": _ATOM_SLOT,
                    "L": _ATOM_SLOT,
                },
            ),
            Template(
                name="limit_pos_infinity",
                latex=r"\lim_{{{x} \to +\infty}} {f}({x}) = {L}",
                slots={
                    "f": _FN_SLOT,
                    "x": S(_BVAR, idx=0.35),
                    "L": _ATOM_SLOT,
                },
            ),
            Template(
                name="limit_neg_infinity",
                latex=r"\lim_{{{x} \to -\infty}} {f}({x}) = {L}",
                slots={
                    "f": _FN_SLOT,
                    "x": S(_BVAR, idx=0.35),
                    "L": _ATOM_SLOT,
                },
            ),
            Template(
                name="limit_sequence",
                latex=r"\lim_{{{idx} \to \infty}} {seq}_{{{idx}}} = {L}",
                slots={
                    "seq": S(("a", "b", "x", "y", "u", "c", "z", "v")),
                    "idx": S(("n", "m", "j", "k"), idx=0.25),
                    "L": _ATOM_SLOT,
                },
            ),
            Template(
                name="limit_newton_quotient",
                latex=r"\lim_{{{x} \to {a}}} \frac{{{f}({x}) - {f}({a})}}{{{x} - {a}}} = {L}",
                slots={
                    "f": _FN_SLOT,
                    "x": S(_BVAR, idx=0.35),
                    "a": _ATOM_SLOT,
                    "L": _ATOM_SLOT,
                },
            ),
            Template(
                name="limit_sum_rule",
                latex=(
                    r"\lim_{{{x} \to {a}}} \bigl({f}({x}) + {g}({x})\bigr) "
                    r"= \lim_{{{x} \to {a}}} {f}({x}) + \lim_{{{x} \to {a}}} {g}({x})"
                ),
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "x": S(_BVAR, idx=0.35),
                    "a": _ATOM_SLOT,
                },
            ),
            Template(
                name="limit_product_rule",
                latex=(
                    r"\lim_{{{x} \to {a}}} {f}({x}) {g}({x}) "
                    r"= \lim_{{{x} \to {a}}} {f}({x}) \cdot \lim_{{{x} \to {a}}} {g}({x})"
                ),
                slots={
                    "f": _FN_SLOT,
                    "g": _FN_SLOT,
                    "x": S(_BVAR, idx=0.35),
                    "a": _ATOM_SLOT,
                },
            ),
            Template(
                name="limit_diverges_inf",
                latex=r"\lim_{{{x} \to {a}}} {f}({x}) = {inf_sym}",
                slots={
                    "f": _FN_SLOT,
                    "x": S(_BVAR, idx=0.35),
                    "a": _ATOM_SLOT,
                    "inf_sym": S((r"\infty", r"+\infty", r"-\infty")),
                },
            ),
        ],
    ),
    # --- Weierstrass M-test ---
    Template(
        name="weierstrass_m_test",
        latex=(
            r"\sum{lim_mod}_{{n=1}}^\infty \left|{f}_n({x})\right| \leq {M}_n, \; "
            r"\sum{lim_mod}_{{n=1}}^\infty {M}_n < \infty "
            r"\implies \sum{lim_mod}_{{n=1}}^\infty {f}_n \text{{ converges uniformly}}"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "f": _FN_SLOT,
            "x": S(_BVAR, idx=0.35),  # function argument variable
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
            "f": _FN_SLOT,
            "x": S(_BVAR, idx=0.35),  # function argument variable
        },
    ),
    # --- Banach fixed-point theorem ---
    Template(
        name="banach_fixed_point",
        latex=r"\exists!\, x^* : {T}(x^*) = x^*, \quad x_{{n+1}} = {T}(x_n) \to x^*",
        slots={"T": _FN_SLOT},
    ),
    # --- Series convergence tests ---
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
                    r"\sum{lim_mod}_{{n=1}}^\infty {f}(n) \text{{ converges}} "
                    r"\iff \int_1^\infty {f}({t})\,d{t} < \infty"
                ),
                slots={
                    "lim_mod": _LIM_MOD,
                    "f": _FN_SLOT,
                    "t": S(_BVAR, idx=0.35),  # integration variable
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
                    "L": _ATOM_SLOT,
                },
                distinct=[["a", "b"]],
            ),
        ],
    ),
    # --- Ratio / root tests / alternating series (standalone — low n_eff) ---
    Template(
        name="ratio_test",
        latex=(
            r"\lim_{{{v} \to \infty}} \left|\frac{{{seq}_{{{v}+1}}}}{{{seq}_{{{v}}}}}\right| = {L} < 1 "
            r"\implies \sum {seq}_n \text{{ converges absolutely}}"
        ),
        slots={
            "seq": S(("a", "b", "c", "x", "y", "u", "v", "p")),
            "v": S(("n", "m", "k", "j")),
            "L": _ATOM_SLOT,
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
            "L": _ATOM_SLOT,
        },
    ),
    Template(
        name="alternating_series",
        latex=(
            r"{a}_{{{v}}} \searrow 0 \implies "
            r"\sum{lim_mod}_{{{v}={v0}}}^\infty (-1)^{{{v}}} {a}_{{{v}}} \text{{ converges}}"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "a": S(("a", "b", "c", "x", "y", "u", "v", "p")),
            "v": S(("n", "m", "k", "j")),
            "v0": S(("0", "1", "2", "3", r"n_0", "N")),
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
                slots={
                    "f": _FN_SLOT,
                    "x": S(_BVAR, idx=0.35),  # quantified variable
                },
            ),
            Template(
                name="uniform_conv_def",
                latex=r"\sup_{{{x} \in D}} \left|{f}_n({x}) - {f}({x})\right| \to 0",
                slots={
                    "f": _FN_SLOT,
                    "x": S(_BVAR, idx=0.35),  # quantified variable
                },
            ),
            Template(
                name="uniform_implies_continuous",
                latex=(
                    r"{f}_n \rightrightarrows {f} \text{{ on }} {D}, \; "
                    r"{f}_n \text{{ continuous}} \implies {f} \text{{ continuous}}"
                ),
                slots={
                    "f": _FN_SLOT,
                    "D": S(("D", "X", "K", r"\Omega", "A", "U")),
                },
            ),
            Template(
                name="uniform_implies_integrable",
                latex=(
                    r"{f}_n \rightrightarrows {f} \text{{ on }} [{a}, {b}] "
                    r"\implies \int{lim_mod}_{{{a}}}^{{{b}}} {f}_n \to \int{lim_mod}_{{{a}}}^{{{b}}} {f}"
                ),
                slots={
                    "lim_mod": _LIM_MOD,
                    "f": _FN_SLOT,
                    "a": _ATOM_SLOT,  # integration bounds (any value)
                    "b": _ATOM_SLOT,
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
                    "f": _FN_SLOT,
                    "a": _ATOM_SLOT,  # endpoints (any value)
                    "b": _ATOM_SLOT,
                },
            ),
            Template(
                name="mvt",
                latex=(
                    r"\exists\, c \in ({a}, {b}) : {f}'(c) = "
                    r"\dfrac{{{f}({b}) - {f}({a})}}{{{b} - {a}}}"
                ),
                slots={
                    "f": _FN_SLOT,
                    "a": _ATOM_SLOT,
                    "b": _ATOM_SLOT,
                },
            ),
            Template(
                name="extreme_value",
                latex=(
                    r"{f} \in C([{a}, {b}]) \implies "
                    r"{f} \text{{ attains its maximum and minimum on }} [{a}, {b}]"
                ),
                slots={
                    "f": _FN_SLOT,
                    "a": _ATOM_SLOT,
                    "b": _ATOM_SLOT,
                },
            ),
        ],
    ),
    # --- Bolzano-Weierstrass (standalone — low n_eff) ---
    Template(
        name="bolzano_weierstrass",
        latex=(
            r"\text{{Every bounded sequence }} \left\{{{seq}_k\right\}} \text{{ in }} \mathbb{{R}}^{{{d}}} "
            r"\text{{ has a convergent subsequence}}"
        ),
        slots={
            "seq": S(("x", "a", "y", "z", "u", "v", "s", "b", "c", "w")),
            "d": S(("1", "2", "3", "n", "d", "m", "N", "k")),
        },
    ),
    # --- Taylor remainder bound ---
    Template(
        name="taylor_remainder",
        latex=(
            r"\left|{f}({x}) - \sum{lim_mod}_{{k=0}}^{{{n}}} \frac{{{f}^{{(k)}}({a})}}{{k!}}({x}-{a})^k\right|"
            r" \leq \frac{{M}}{{({n}+1)!}} \left|{x} - {a}\right|^{{{n}+1}}"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "f": _FN_SLOT,
            "x": _ATOM_SLOT,  # evaluation point (specific value)
            "a": _ATOM_SLOT,  # expansion point (specific value)
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

# Part C — high-n_eff fn-pair templates
_ANALYSIS_TEMPLATES += [
    Template(
        name="fn_limit_composition",
        latex=r"{fn1}\!\left(\lim_{{{x} \to {a}}} {fn2}({x})\right) = \lim_{{{x} \to {a}}} {fn1}\!\left({fn2}({x})\right)",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "x": S(_BVAR, idx=0.35),
            "a": _ATOM_SLOT,
        },
    ),
    Template(
        name="fn_continuity_bound",
        latex=r"|{fn1}({x}) - {fn1}({y})| \leq {fn2}(|{x} - {y}|)",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "x": S(_BVAR, idx=0.35),
            "y": S(_BVAR, idx=0.35),
        },
    ),
    Template(
        name="fn_uniform_convergence",
        latex=r"\sup_{{{x} \in D}} |{fn1}_n({x}) - {fn2}({x})| \to 0 \text{{ as }} n \to \infty",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "x": S(_BVAR, idx=0.35),
        },
    ),
    Template(
        name="fn_integral_bound",
        latex=r"\left|\int {fn1}({x})\,d{x}\right| \leq \int |{fn2}({x})|\,d{x}",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "x": S(_BVAR, idx=0.35),
        },
    ),
    Template(
        name="fn_derivative_chain",
        latex=r"({fn1} \circ {fn2})'({x}) = {fn1}'({fn2}({x})) \cdot {fn2}'({x})",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "x": S(_BVAR, idx=0.35),
        },
    ),
    Template(
        name="fn_sequence_bound",
        latex=r"|{fn1}(a_n) - {fn1}(L)| \leq {fn2}(|a_n - L|) \to 0",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
        },
    ),
    Template(
        name="fn_series_tail",
        latex=r"\sum{lim_mod}_{{n={v}}}^{{\infty}} {fn1}(a_n) \leq {fn2}\!\left(\sum{lim_mod}_{{n={v}}}^{{\infty}} |a_n|\right)",
        slots={
            "lim_mod": _LIM_MOD,
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "v": S(("n", "m", "k", "j", "N", "M")),
        },
    ),
    Template(
        name="fn_metric_bound",
        latex=r"{fn1}(d({x},{y})) \leq {fn2}(d({x},z) + d(z,{y}))",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "x": S(_BVAR, idx=0.35),
            "y": S(_BVAR, idx=0.35),
        },
    ),
]

# ---------------------------------------------------------------------------
# Greek symbol coverage: \varsigma, \digamma, \Digamma
# ---------------------------------------------------------------------------

_ANALYSIS_TEMPLATES += [
    # \varsigma — Stieltjes transform (uses \varsigma as the transform variable)
    Template(
        name="stieltjes_transform_def",
        latex=(
            r"\mathcal{{S}}[f](\varsigma)"
            r" = \int_0^{{\infty}} \frac{{f(t)}}{{t + \varsigma}} \, dt,"
            r"\quad \varsigma \notin (-\infty, 0]"
        ),
        slots={},
    ),
    Template(
        name="stieltjes_inversion",
        latex=(
            r"f(t) = -\frac{{1}}{{\pi}}"
            r" \lim_{{\varepsilon \to 0^+}}"
            r" \operatorname{{Im}} \mathcal{{S}}[f](-t + i\varepsilon)"
        ),
        slots={},
    ),
    Template(
        name="dirichlet_series_varsigma",
        latex=(
            r"F(\varsigma) = \sum{lim_mod}_{{n=1}}^{{\infty}} \frac{{a_n}}{{n^\varsigma}},"
            r"\quad \operatorname{{Re}}(\varsigma) > {cc}"
        ),
        slots={"lim_mod": _LIM_MOD, "cc": S(_BVAR)},
    ),
    # \digamma — digamma function (logarithmic derivative of \Gamma)
    Template(
        name="digamma_log_gamma",
        latex=(
            r"\digamma({xx}) = \frac{{d}}{{d{xx}}} \ln \Gamma({xx})"
            r" = \frac{{\Gamma'({xx})}}{{\Gamma({xx})}}"
        ),
        slots={"xx": S(_BVAR)},
    ),
    Template(
        name="digamma_recurrence",
        latex=r"\digamma({xx} + 1) = \digamma({xx}) + \frac{{1}}{{{xx}}}",
        slots={"xx": S(_BVAR)},
    ),
    Template(
        name="digamma_series_rep",
        latex=(
            r"\digamma({xx}) = -\gamma"
            r" + \sum{lim_mod}_{{n=0}}^{{\infty}}"
            r"\left(\frac{{1}}{{n+1}} - \frac{{1}}{{n+{xx}}}\right)"
        ),
        slots={"lim_mod": _LIM_MOD, "xx": S(_BVAR)},
    ),
    Template(
        name="digamma_integral_rep",
        latex=(
            r"\digamma({xx})"
            r" = \int_0^{{\infty}}\left("
            r"\frac{{e^{{-t}}}}{{t}} - \frac{{e^{{-{xx} t}}}}{{1-e^{{-t}}}}"
            r"\right)dt"
        ),
        slots={"xx": S(_BVAR)},
    ),
    Template(
        name="digamma_euler_mascheroni",
        latex=r"\digamma(1) = -\gamma \approx -0.5772",
        slots={},
    ),
    Template(
        name="digamma_reflection",
        latex=(
            r"\digamma(1-{xx}) - \digamma({xx})"
            r" = \pi \cot(\pi {xx})"
        ),
        slots={"xx": S(_BVAR)},
    ),
    # \Digamma — uppercase digamma, used as a formal antiderivative / generating function
    Template(
        name="digamma_antiderivative",
        latex=(
            r"\Digamma({xx}) = \int_1^{{{xx}}} \digamma(t)\, dt"
            r" = \ln \Gamma({xx}) - \ln \Gamma(1)"
        ),
        slots={"xx": S(_BVAR)},
    ),
    Template(
        name="digamma_partial_sum",
        latex=(
            r"\Digamma_n = \sum{lim_mod}_{{k=1}}^{{n}} \digamma(k)"
            r" = -n\gamma + \sum{lim_mod}_{{k=1}}^{{n}} H_{{k-1}}"
        ),
        slots={
            "lim_mod": _LIM_MOD,
        },
    ),
]

_SEQ_POOL = ("a", "b", "c", "f", "g", "h")
_SP_POOL = (r"L^2(\Omega)", r"L^p(\Omega)", r"H^1(\Omega)", "H")
_NN_POOL = ("n", "k", "m")
_DOM_POOL = (r"\Omega", r"[0,1]", r"\mathbb{R}^n")

_PART_ARROWS: list[Template] = [
    Template(
        name="sequence_limit_rightarrow",
        latex=r"{aa}_n \rightarrow {ll} \quad \text{{as }} n \rightarrow \infty",
        slots={"aa": S(_SEQ_POOL), "ll": S(_BVAR)},
    ),
    Template(
        name="function_measurable_rightarrow",
        latex=r"f: {dom} \rightarrow \mathbb{{R}} \text{{ is measurable}}",
        slots={"dom": S(_DOM_POOL)},
    ),
    Template(
        name="weak_convergence_lp",
        latex=r"f_{{{nn}}} \rightharpoonup f \text{{ in }} {sp}",
        slots={"nn": S(_NN_POOL), "sp": S(_SP_POOL)},
    ),
    Template(
        name="weak_convergence_hilbert",
        latex=r"\langle x_{{{nn}}}, y \rangle \rightarrow \langle x, y \rangle \;\forall\, y \implies x_{{{nn}}} \rightharpoonup x",
        slots={"nn": S(_NN_POOL)},
    ),
]
_ANALYSIS_TEMPLATES += _PART_ARROWS

# ---------------------------------------------------------------------------
# Evaluation-bar and large-bracket templates
# ---------------------------------------------------------------------------

_ANALYSIS_TEMPLATES += [
    # Function evaluated at a point with \left. ... \right|
    Template(
        name="limit_eval_bar",
        latex=r"\left.{f}({v})\right|_{{{v}={a}}} = \lim_{{{v} \to {a}}} {f}({v})",
        slots={
            "f": _FN_SLOT,
            "v": S(_BVAR),
            "a": _ATOM_SLOT,
        },
    ),
    # Norm of a fraction — \biggl\| ... \biggr\| exposes size-3 manual sizing
    Template(
        name="norm_frac_biggl",
        latex=r"\biggl\| \frac{{{f}({v})}}{{{g}({v})}} \biggr\|",
        slots={
            "f": _FN_SLOT,
            "g": _FN_SLOT,
            "v": S(_BVAR),
        },
    ),
    # Inner product of a fraction with a function — \Biggl\langle ... \Biggr\rangle (size 4)
    Template(
        name="inner_product_biggl",
        latex=r"\Biggl\langle \frac{{{expr1}}}{{{expr2}}},\; {f} \Biggr\rangle",
        slots={
            "expr1": E(_expr, n=3000),
            "expr2": E(_expr, n=3000),
            "f": _FN_SLOT,
        },
    ),
    # Absolute value of a fraction — \biggl| ... \biggr| (size 3)
    Template(
        name="abs_frac_biggl",
        latex=r"\biggl| \frac{{{expr1}}}{{{expr2}}} \biggr|",
        slots={
            "expr1": E(_expr, n=3000),
            "expr2": E(_expr, n=3000),
        },
    ),
]

# ---------------------------------------------------------------------------
# Backslash-space (\ ) qualifier patterns
# ---------------------------------------------------------------------------

_BSLVAR_POOL: tuple[str, ...] = ("A", "B", "C", "S", "T")
_DOM_POOL: tuple[str, ...] = (r"\mathbb{R}", r"\mathbb{R}^n", r"[a,b]", r"\mathbb{C}", r"\mathbb{Z}")
_SPACE_POOL: tuple[str, ...] = (r"L^p", r"L^2", r"L^\infty", r"\mathcal{H}", r"C([a,b])")

_PART_BSLSPACE: list[Template] = [
    Template(
        name="bslspace_forall_explicit",
        latex=r"{ff}({vv}) = {expr}, \ \forall {vv} \in {dom}",
        slots={
            "ff": _FN_SLOT,
            "vv": S(_VARS),
            "expr": _EXPR_SLOT,
            "dom": S(_DOM_POOL),
        },
    ),
    Template(
        name="bslspace_exists_explicit",
        latex=r"{ff}({vv}) \leq {cc}, \ \exists {vv} \in {dom}",
        slots={
            "ff": _FN_SLOT,
            "vv": S(_VARS),
            "cc": _ATOM_SLOT,
            "dom": S((r"\mathbb{R}", r"\mathbb{Z}", r"[0,\infty)")),
        },
    ),
    Template(
        name="bslspace_chain_implication",
        latex=r"{aa} \subseteq {bb}, \ {bb} \subseteq {cc} \implies {aa} \subseteq {cc}",
        slots={
            "aa": S(_BSLVAR_POOL),
            "bb": X(_BSLVAR_POOL, ("aa",)),
            "cc": X(_BSLVAR_POOL, ("aa", "bb")),
        },
    ),
    Template(
        name="bslspace_inequality_chain",
        latex=r"\|{ff}({vv})\| \leq {c}_1, \ \|{gg}({vv})\| \leq {c}_2",
        slots={
            "ff": _FN_SLOT,
            "gg": _FN_SLOT,
            "vv": S(_VARS),
            "c": S(("C", "M", "K", "L")),
        },
    ),
    Template(
        name="bslspace_condition_separation",
        latex=r"{lhs} = {rhs} \ \Rightarrow \ {consequence}",
        slots={
            "lhs": _EXPR_SLOT,
            "rhs": _EXPR_SLOT,
            "consequence": _EXPR_SLOT,
        },
    ),
    Template(
        name="bslspace_bound_qualifier",
        latex=r"\|{ff}\|_{{{pp}}} \leq {cc}, \ \forall {ff} \in {space}",
        slots={
            "ff": _FN_SLOT,
            "pp": S(("p", "2", "q", "1", r"\infty")),
            "cc": _ATOM_SLOT,
            "space": S(_SPACE_POOL),
        },
    ),
    Template(
        name="bslspace_forall_solution",
        latex=r"{ff}({vv}) = 0, \ {vv} \in {dom}",
        slots={
            "ff": _FN_SLOT,
            "vv": S(_VARS),
            "dom": S(_DOM_POOL),
        },
    ),
]
_ANALYSIS_TEMPLATES += _PART_BSLSPACE


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("analysis", _ANALYSIS_TEMPLATES)
