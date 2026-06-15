"""Asymptotics and numerical analysis domain generators.

Covers Landau notation (O, o, Θ, Ω, ω, ~, Õ), asymptotic expansions,
algorithm complexity, numerical convergence rates, finite-difference
approximations, classical asymptotic estimates, and numerical stability.
"""

from __future__ import annotations

from ..engine._template_dsl import _FN_SLOT, _LIM_MOD, S, Template
from ..engine._vocab import _GEO_N, _VARS
from ._config import register_domain

_BVAR = _VARS  # ("x","y","z","t","u","v","r","s")

# ---------------------------------------------------------------------------
# Asymptotics templates
# ---------------------------------------------------------------------------

_ASYMPTOTICS_TEMPLATES: list[Template] = [
    # --- Big-O / little-o / Theta at v → ∞  (extracted from analysis.py) ---
    Template(
        name="big_o_asymptotic",
        latex="",
        slots={},
        variants=[
            Template(
                name="big_o",
                latex=r"{f}({v}) = O\!\left({g}({v})\right) \text{{ as }} {v} \to \infty",
                slots={"f": _FN_SLOT, "g": _FN_SLOT, "v": S(_GEO_N)},
            ),
            Template(
                name="little_o",
                latex=r"{f}({v}) = o\!\left({g}({v})\right) \text{{ as }} {v} \to \infty",
                slots={"f": _FN_SLOT, "g": _FN_SLOT, "v": S(_GEO_N)},
            ),
            Template(
                name="big_theta",
                latex=r"{f}({v}) = \Theta\!\left({g}({v})\right) \text{{ as }} {v} \to \infty",
                slots={"f": _FN_SLOT, "g": _FN_SLOT, "v": S(_GEO_N)},
            ),
            Template(
                name="big_omega",
                latex=r"{f}({v}) = \Omega\!\left({g}({v})\right) \text{{ as }} {v} \to \infty",
                slots={"f": _FN_SLOT, "g": _FN_SLOT, "v": S(_GEO_N)},
            ),
            Template(
                name="little_omega",
                latex=r"{f}({v}) = \omega\!\left({g}({v})\right) \text{{ as }} {v} \to \infty",
                slots={"f": _FN_SLOT, "g": _FN_SLOT, "v": S(_GEO_N)},
            ),
            Template(
                name="asymptotic_equiv",
                latex=r"{f}({v}) \sim {g}({v}) \text{{ as }} {v} \to \infty",
                slots={"f": _FN_SLOT, "g": _FN_SLOT, "v": S(_GEO_N)},
            ),
            Template(
                name="soft_o",
                latex=r"{f}({v}) = \tilde{{O}}\!\left({g}({v})\right) \text{{ as }} {v} \to \infty",
                slots={"f": _FN_SLOT, "g": _FN_SLOT, "v": S(_GEO_N)},
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
                slots={"f": _FN_SLOT, "g": _FN_SLOT, "h": _FN_SLOT, "v": S(_GEO_N)},
            ),
            Template(
                name="error_little_o",
                latex=r"{f}({v}) = {g}({v}) + o\!\left({h}({v})\right) \text{{ as }} {v} \to \infty",
                slots={"f": _FN_SLOT, "g": _FN_SLOT, "h": _FN_SLOT, "v": S(_GEO_N)},
            ),
            Template(
                name="error_little_o_one",
                latex=r"{f}({v}) = {g}({v}) + o(1) \text{{ as }} {v} \to \infty",
                slots={"f": _FN_SLOT, "g": _FN_SLOT, "v": S(_GEO_N)},
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
                slots={"f": _FN_SLOT, "g": _FN_SLOT, "v": S(_GEO_N)},
            ),
            Template(
                name="little_o_limit",
                latex=r"\lim_{{{v} \to \infty}} \frac{{{f}({v})}}{{{g}({v})}} = 0",
                slots={"f": _FN_SLOT, "g": _FN_SLOT, "v": S(_GEO_N)},
            ),
            Template(
                name="asymptotic_equiv_limit",
                latex=r"\lim_{{{v} \to \infty}} \frac{{{f}({v})}}{{{g}({v})}} = 1",
                slots={"f": _FN_SLOT, "g": _FN_SLOT, "v": S(_GEO_N)},
            ),
            Template(
                name="big_theta_sandwich",
                latex=(
                    r"c_1\,{g}({v}) \leq {f}({v}) \leq c_2\,{g}({v})"
                    r"\text{{ for all large }} {v}"
                ),
                slots={"f": _FN_SLOT, "g": _FN_SLOT, "v": S(_GEO_N)},
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
                latex=(
                    r"{f}({v}) \sim \sum{lim_mod}_{{{k}=0}}^{{\infty}}"
                    r" {a}_{{{k}}}\,{v}^{{-{k}}} \text{{ as }} {v} \to \infty"
                ),
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
    # --- Algorithm time complexity ---
    Template(
        name="algorithm_complexity",
        latex="",
        slots={},
        variants=[
            Template(
                name="time_polynomial",
                latex=r"T({nn}) = O\!\left({nn}^{{{kk}}}\right)",
                slots={
                    "nn": S(("n", "m", "N")),
                    "kk": S(("2", "3", "k", "d", "p", "4")),
                },
            ),
            Template(
                name="time_log",
                latex=r"T({nn}) = O\!\left(\log {nn}\right)",
                slots={"nn": S(("n", "m", "N", "k"))},
            ),
            Template(
                name="time_linearithmic",
                latex=r"T({nn}) = O\!\left({nn} \log {nn}\right)",
                slots={"nn": S(("n", "m", "N"))},
            ),
            Template(
                name="time_exponential",
                latex=r"T({nn}) = O\!\left({bb}^{{{nn}}}\right)",
                slots={
                    "nn": S(("n", "m", "N")),
                    "bb": S(("2", "3", "c", "k")),
                },
            ),
            Template(
                name="time_linear",
                latex=r"T({nn}) = O\!\left({nn}\right)",
                slots={"nn": S(("n", "m", "N"))},
            ),
            Template(
                name="time_omega_lower",
                latex=r"T({nn}) = \Omega\!\left({nn} \log {nn}\right)",
                slots={"nn": S(("n", "m", "N"))},
            ),
        ],
    ),
    # --- Master theorem (divide-and-conquer recurrences) ---
    Template(
        name="master_theorem",
        latex="",
        slots={},
        variants=[
            Template(
                name="master_log_b_a",
                latex=(
                    r"T({nn}) = {aa}\,T\!\left(\tfrac{{{nn}}}{{{bb}}}\right) + O(1)"
                    r" \implies T({nn}) = O\!\left({nn}^{{\log_{{{bb}}} {aa}}}\right)"
                ),
                slots={
                    "nn": S(("n", "m", "N")),
                    "aa": S(("a", "2", "3", "4")),
                    "bb": S(("b", "2", "3", "4")),
                },
            ),
            Template(
                name="master_nlogn",
                latex=(
                    r"T({nn}) = {aa}\,T\!\left(\tfrac{{{nn}}}{{{bb}}}\right) + \Theta\!\left({nn}^{{{cc}}}\right)"
                    r" \implies T({nn}) = \Theta\!\left({nn}^{{{cc}}} \log {nn}\right)"
                ),
                slots={
                    "nn": S(("n", "m", "N")),
                    "aa": S(("a", "2", "3", "4")),
                    "bb": S(("b", "2", "3", "4")),
                    "cc": S(("c", "1", "2", "p")),
                },
            ),
            Template(
                name="master_dominant",
                latex=(
                    r"T({nn}) = {aa}\,T\!\left(\tfrac{{{nn}}}{{{bb}}}\right) + \Omega\!\left({nn}^{{{cc}}}\right)"
                    r" \implies T({nn}) = \Theta\!\left({nn}^{{{cc}}}\right)"
                ),
                slots={
                    "nn": S(("n", "m", "N")),
                    "aa": S(("a", "2", "3", "4")),
                    "bb": S(("b", "2", "3", "4")),
                    "cc": S(("c", "2", "3", "p")),
                },
            ),
        ],
    ),
    # --- Space complexity ---
    Template(
        name="space_complexity",
        latex="",
        slots={},
        variants=[
            Template(
                name="space_polynomial",
                latex=r"S({nn}) = O\!\left({nn}^{{{kk}}}\right)",
                slots={
                    "nn": S(("n", "m", "N")),
                    "kk": S(("2", "k", "1", "d")),
                },
            ),
            Template(
                name="space_log",
                latex=r"S({nn}) = O\!\left(\log {nn}\right)",
                slots={"nn": S(("n", "m", "N"))},
            ),
            Template(
                name="space_linear",
                latex=r"S({nn}) = O\!\left({nn}\right)",
                slots={"nn": S(("n", "m", "N"))},
            ),
        ],
    ),
    # --- Numerical convergence rates ---
    Template(
        name="numerical_convergence",
        latex="",
        slots={},
        variants=[
            Template(
                name="quadratic_convergence",
                latex=r"|{ee}_{{n+1}}| \leq C\,|{ee}_n|^2",
                slots={"ee": S(("e", r"\varepsilon", "r"))},
            ),
            Template(
                name="linear_convergence",
                latex=r"|{ee}_{{n+1}}| \leq {qq}\,|{ee}_n|, \quad 0 < {qq} < 1",
                slots={
                    "ee": S(("e", r"\varepsilon", "r")),
                    "qq": S(("q", r"\rho", r"\lambda", "c", "L")),
                },
            ),
            Template(
                name="superlinear_convergence",
                latex=r"|{ee}_{{n+1}}| \leq C\,|{ee}_n|^{{{pp}}}",
                slots={
                    "ee": S(("e", r"\varepsilon", "r")),
                    "pp": S((r"\frac{3}{2}", r"\frac{5}{3}", "p")),
                },
            ),
            Template(
                name="contraction_iterate",
                latex=r"|x_{{n+1}} - x^*| \leq {qq}\,|x_n - x^*|",
                slots={"qq": S(("q", r"\rho", r"\lambda", r"\alpha"))},
            ),
        ],
    ),
    # --- Order of convergence ---
    Template(
        name="order_of_convergence",
        latex=r"\lim_{{n \to \infty}} \frac{{|{ee}_{{n+1}}|}}{{|{ee}_n|^{{{pp}}}}} = {LL}",
        slots={
            "ee": S(("e", r"\varepsilon", "r")),
            "pp": S(("1", "2", "p", r"\frac{3}{2}")),
            "LL": S(("L", "C", r"\mu", r"\lambda")),
        },
    ),
    # --- Finite-difference approximations ---
    Template(
        name="finite_diff_forward",
        latex=r"{f}'({x}) = \frac{{{f}({x}+h) - {f}({x})}}{{h}} + O(h)",
        slots={
            "f": _FN_SLOT,
            "x": S(_BVAR, idx=0.35),
        },
    ),
    Template(
        name="finite_diff_central",
        latex=r"{f}'({x}) = \frac{{{f}({x}+h) - {f}({x}-h)}}{{2h}} + O(h^2)",
        slots={
            "f": _FN_SLOT,
            "x": S(_BVAR, idx=0.35),
        },
    ),
    Template(
        name="finite_diff_second",
        latex=r"{f}''({x}) = \frac{{{f}({x}+h) - 2{f}({x}) + {f}({x}-h)}}{{h^2}} + O(h^2)",
        slots={
            "f": _FN_SLOT,
            "x": S(_BVAR, idx=0.35),
        },
    ),
    # --- Quadrature error bounds ---
    Template(
        name="quadrature_error",
        latex="",
        slots={},
        variants=[
            Template(
                name="trapezoid_error",
                latex=(
                    r"|E_n| \leq \frac{{({bb}-{aa})^3}}{{12 n^2}}"
                    r" \max_{{{x} \in [{aa},{bb}]}} |{f}''({x})|"
                ),
                slots={
                    "f": _FN_SLOT,
                    "aa": S(("a", "0", r"\alpha")),
                    "bb": S(("b", "1", r"\beta", "T")),
                    "x": S(_BVAR, idx=0.35),
                },
            ),
            Template(
                name="simpson_error",
                latex=(
                    r"|E_n| \leq \frac{{({bb}-{aa})^5}}{{180 n^4}}"
                    r" \max_{{{x} \in [{aa},{bb}]}} |{f}^{{(4)}}({x})|"
                ),
                slots={
                    "f": _FN_SLOT,
                    "aa": S(("a", "0", r"\alpha")),
                    "bb": S(("b", "1", r"\beta", "T")),
                    "x": S(_BVAR, idx=0.35),
                },
            ),
            Template(
                name="generic_quadrature_error",
                latex=(
                    r"|E_n| = O\!\left(h^{{{pp}}}\right),"
                    r"\quad h = \frac{{{bb}-{aa}}}{{{nn}}}"
                ),
                slots={
                    "pp": S(("2", "4", "p", "6")),
                    "aa": S(("a", "0")),
                    "bb": S(("b", "1", "T")),
                    "nn": S(("n", "N", "m")),
                },
            ),
        ],
    ),
    # --- Stirling's approximation ---
    Template(
        name="stirling_approx",
        latex="",
        slots={},
        variants=[
            Template(
                name="stirling_product",
                latex=r"{nn}! \sim \sqrt{{2\pi {nn}}} \left(\frac{{{nn}}}{{e}}\right)^{{{nn}}}",
                slots={"nn": S(("n", "m", "N", "k"))},
            ),
            Template(
                name="stirling_log",
                latex=r"\ln({nn}!) = {nn} \ln {nn} - {nn} + O(\ln {nn})",
                slots={"nn": S(("n", "m", "N"))},
            ),
            Template(
                name="stirling_refined",
                latex=(
                    r"{nn}! = \sqrt{{2\pi {nn}}} \left(\frac{{{nn}}}{{e}}\right)^{{{nn}}}"
                    r" \left(1 + O\!\left(\tfrac{{1}}{{{nn}}}\right)\right)"
                ),
                slots={"nn": S(("n", "m", "N"))},
            ),
        ],
    ),
    # --- Prime number theorem ---
    Template(
        name="prime_number_theorem",
        latex="",
        slots={},
        variants=[
            Template(
                name="pnt_simple",
                latex=r"\pi({xx}) \sim \frac{{{xx}}}{{\ln {xx}}} \text{{ as }} {xx} \to \infty",
                slots={"xx": S(("x", "n", "N", "X"))},
            ),
            Template(
                name="pnt_li",
                latex=(
                    r"\pi({xx}) = \operatorname{{Li}}({xx})"
                    r" + O\!\left({xx}^{{{rr}}} \ln {xx}\right)"
                ),
                slots={
                    "xx": S(("x", "X")),
                    "rr": S((r"\frac{1}{2}", r"\theta", r"\frac{3}{4}")),
                },
            ),
        ],
    ),
    # --- Harmonic number asymptotics ---
    Template(
        name="harmonic_asymptotic",
        latex=r"H_{{{nn}}} = \ln {nn} + \gamma + O\!\left(\tfrac{{1}}{{{nn}}}\right)",
        slots={"nn": S(("n", "m", "N", "k"))},
    ),
    # --- Condition number ---
    Template(
        name="condition_number",
        latex="",
        slots={},
        variants=[
            Template(
                name="condition_number_2norm",
                latex=r"\kappa_2({AA}) = \|{AA}\|_2 \|{AA}^{{-1}}\|_2",
                slots={"AA": S(("A", "B", "M", r"\Sigma"))},
            ),
            Template(
                name="condition_number_pnorm",
                latex=r"\kappa_{{{pp}}}({AA}) = \|{AA}\|_{{{pp}}} \|{AA}^{{-1}}\|_{{{pp}}}",
                slots={
                    "AA": S(("A", "B", "M")),
                    "pp": S(("p", "1", "2", r"\infty")),
                },
            ),
            Template(
                name="condition_singular_ratio",
                latex=r"\kappa_2({AA}) = \frac{{{ss}_{{\max}}}}{{{ss}_{{\min}}}}",
                slots={"AA": S(("A", "M", "B")), "ss": S(("s", r"\sigma"))},
            ),
        ],
    ),
    # --- Relative error / perturbation bounds ---
    Template(
        name="relative_error_bound",
        latex="",
        slots={},
        variants=[
            Template(
                name="relative_error_rhs",
                latex=(
                    r"\frac{{\|\delta {xx}\|}}{{\|{xx}\|}}"
                    r" \leq \kappa({AA}) \frac{{\|\delta {bb}\|}}{{\|{bb}\|}}"
                ),
                slots={
                    "AA": S(("A", "B", "M")),
                    "xx": S(("x", "u", "v", "y")),
                    "bb": S(("b", "f", "g")),
                },
            ),
            Template(
                name="residual_bound",
                latex=(
                    r"\frac{{\|{xx} - \hat{{{xx}}}\|}}{{\|{xx}\|}}"
                    r" \leq \kappa({AA}) \frac{{\|{rr}\|}}{{\|{bb}\|}}"
                ),
                slots={
                    "AA": S(("A", "B", "M")),
                    "xx": S(("x", "u", "v")),
                    "bb": S(("b", "f", "g")),
                    "rr": S(("r", "e", r"\varepsilon")),
                },
            ),
        ],
    ),
]

GENERATORS, WEIGHTS, TEMPLATES = register_domain("asymptotics", _ASYMPTOTICS_TEMPLATES)
