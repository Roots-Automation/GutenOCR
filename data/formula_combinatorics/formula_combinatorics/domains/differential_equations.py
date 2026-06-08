"""Differential equations domain generators."""

from __future__ import annotations

from .._template_dsl import _FN_SLOT, _LIM_MOD, S, Template, X
from ._config import register_domain

# ---------------------------------------------------------------------------
# Shared pools
# ---------------------------------------------------------------------------

_FUNC_POOL: tuple[str, ...] = (
    "y",
    "u",
    "v",
    "w",
    r"\phi",
    r"\psi",
    r"\chi",
    r"\theta",
    r"\eta",
    r"\xi",
    "f",
    "g",
)  # 12

_VAR_POOL: tuple[str, ...] = (
    "x",
    "t",
    "r",
    "s",
    r"\tau",
    r"\sigma",
    r"\zeta",
    r"\rho",
)  # 8

_COEFF_POOL: tuple[str, ...] = (
    "a",
    "b",
    "c",
    "k",
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\lambda",
    r"\mu",
    r"\nu",
    r"\kappa",
)  # 11

_PARAM_POOL: tuple[str, ...] = (
    r"\omega",
    r"\Omega",
    r"\lambda",
    r"\mu",
    r"\alpha",
    r"\beta",
    "k",
    "m",
    "r",
    "c",
)  # 10

_OP_POOL: tuple[str, ...] = (
    "L",
    "A",
    "B",
    r"\mathcal{L}",
    r"\mathcal{A}",
    r"\mathcal{D}",
)  # 6

_IDX_POOL: tuple[str, ...] = (
    "n",
    "m",
    "k",
    "j",
    "N",
    "M",
)  # 6

_DOMAIN_POOL: tuple[str, ...] = (
    r"\Omega",
    r"\Omega_0",
    "D",
    r"\mathcal{D}",
    r"\Sigma",
    r"[0,L]",
    r"[0,T]",
)  # 7

_SPACE_POOL: tuple[str, ...] = (
    r"\mathbb{R}",
    r"\mathbb{R}^n",
    r"\mathbb{R}^2",
    r"\mathbb{R}^3",
    r"L^2(\Omega)",
    r"H^1(\Omega)",
)  # 6

# ---------------------------------------------------------------------------
# Part A — 14 reparameterized originals
# ---------------------------------------------------------------------------

_TEMPLATES_A: list[Template] = [
    Template(
        name="first_order_linear_ode",
        latex=r"\frac{{d{ff}}}{{d{vv}}} + {aa}({vv}) {ff} = g({vv})",
        slots={
            "vv": S(_VAR_POOL),
            "ff": S(_FUNC_POOL),
            "aa": S(_COEFF_POOL),
        },
    ),  # n_eff = 8 × 12 × 11 = 1,056
    Template(
        name="second_order_linear_ode",
        latex=(r"\frac{{d^2{ff}}}{{d{vv}^2}} + {aa} \frac{{d{ff}}}{{d{vv}}} + {om} {ff} = 0"),
        slots={
            "vv": S(_VAR_POOL),
            "ff": S(_FUNC_POOL),
            "aa": S(_COEFF_POOL),
            "om": S(_PARAM_POOL),
        },
    ),  # n_eff = 8 × 12 × 11 × 10 = 10,560
    Template(
        name="heat_equation",
        latex=(
            r"\frac{{\partial {ff}}}{{\partial t}}"
            r" = {kap} \frac{{\partial^2 {ff}}}{{\partial {vv}^2}}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "kap": S(_COEFF_POOL),
            "vv": S(_VAR_POOL),
        },
    ),  # n_eff = 12 × 11 × 8 = 1,056
    Template(
        name="wave_equation",
        latex=(
            r"\frac{{\partial^2 {ff}}}{{\partial t^2}}"
            r" = {cc}^2 \frac{{\partial^2 {ff}}}{{\partial {vv}^2}}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "cc": S(_PARAM_POOL),
            "vv": S(_VAR_POOL),
        },
    ),  # n_eff = 12 × 10 × 8 = 960
    Template(
        name="laplace_equation",
        latex=(
            r"\nabla^2 {ff} = "
            r"\frac{{\partial^2 {ff}}}{{\partial {vv1}^2}} + "
            r"\frac{{\partial^2 {ff}}}{{\partial {vv2}^2}} = 0"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "vv1": S(_VAR_POOL),
            "vv2": X(_VAR_POOL, ("vv1",)),
        },
    ),  # n_eff = 12 × 8 × 7 = 672
    Template(
        name="homogeneous_ode_solution",
        latex=r"{ff}({vv}) = C_1 e^{{{lam1} {vv}}} + C_2 e^{{{lam2} {vv}}}",
        slots={
            "vv": S(_VAR_POOL),
            "ff": S(_FUNC_POOL),
            "lam1": S(_PARAM_POOL),
            "lam2": X(_PARAM_POOL, ("lam1",)),
        },
    ),  # n_eff = 8 × 12 × 10 × 9 = 8,640
    Template(
        name="separable_ode",
        latex=r"\frac{{1}}{{{ff}}} \frac{{d{ff}}}{{d{vv}}} = {aa}({vv})",
        slots={
            "vv": S(_VAR_POOL),
            "ff": S(_FUNC_POOL),
            "aa": S(_COEFF_POOL),
        },
    ),  # n_eff = 8 × 12 × 11 = 1,056
    Template(
        name="exponential_growth_solution",
        latex=r"{ff}({vv}) = {ff}_0 \, e^{{{rate} {vv}}}",
        slots={
            "vv": S(_VAR_POOL),
            "ff": S(_FUNC_POOL),
            "rate": S(_COEFF_POOL),
        },
    ),  # n_eff = 8 × 12 × 11 = 1,056
    Template(
        name="sturm_liouville_bvp",
        latex=(
            r"{ff}''({vv}) + {om} {ff}({vv}) = 0,"
            r" \quad {ff}(0) = 0, \; {ff}(L) = 0"
        ),
        slots={
            "vv": S(_VAR_POOL),
            "ff": S(_FUNC_POOL),
            "om": S(_PARAM_POOL),
        },
    ),  # n_eff = 8 × 12 × 10 = 960
    Template(
        name="logistic_equation",
        latex=r"\frac{{d{ff}}}{{dt}} = {rr} {ff}\!\left(1 - \frac{{{ff}}}{{{KK}}}\right)",
        slots={
            "ff": S(_FUNC_POOL),
            "rr": S(_COEFF_POOL),
            "KK": X(_COEFF_POOL, ("rr",)),
        },
    ),  # n_eff = 12 × 11 × 10 = 1,320
    Template(
        name="driven_harmonic_oscillator",
        latex=(
            r"\frac{{\partial^2 {ff}}}{{\partial t^2}}"
            r" + 2{aa} \frac{{\partial {ff}}}{{\partial t}}"
            r" + {om0} {ff} = F_0 \cos({om} t)"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "aa": S(_COEFF_POOL),
            "om0": S(_PARAM_POOL),
            "om": X(_PARAM_POOL, ("om0",)),
        },
    ),  # n_eff = 12 × 11 × 10 × 9 = 11,880
    Template(
        name="integrating_factor",
        latex=r"\mu({vv}) = e^{{\int {aa}({vv}) \, d{vv}}}",
        slots={
            "vv": S(_VAR_POOL),
            "aa": S(_COEFF_POOL),
        },
    ),  # n_eff = 8 × 11 = 88
    Template(
        name="greens_function",
        latex=r"{op} G({vv}, {xi}) = \delta({vv} - {xi})",
        slots={
            "op": S(_OP_POOL),
            "vv": S(_VAR_POOL),
            "xi": X(_VAR_POOL, ("vv",)),
        },
    ),  # n_eff = 6 × 8 × 7 = 336
    Template(
        name="variation_of_parameters",
        latex=(
            r"{ff}_p({vv}) = {ff}_1({vv}) \int"
            r" \frac{{{gg}_2 h}}{{{ff}_1 {gg}_2' - {gg}_2 {ff}_1'}} \, d{vv}"
        ),
        slots={
            "vv": S(_VAR_POOL),
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
        },
    ),  # n_eff = 8 × 12 × 11 = 1,056
]

# ---------------------------------------------------------------------------
# Part B1 — ODE Theory (8)
# ---------------------------------------------------------------------------

_TEMPLATES_B1: list[Template] = [
    Template(
        name="wronskian_def",
        latex=r"W({ff1}, {ff2})({vv}) = {ff1} {ff2}' - {ff2} {ff1}'",
        slots={
            "ff1": S(_FUNC_POOL),
            "ff2": X(_FUNC_POOL, ("ff1",)),
            "vv": S(_VAR_POOL),
        },
    ),  # n_eff = 12 × 11 × 8 = 1,056
    Template(
        name="abel_identity",
        latex=(
            r"W_{{{ff1},{ff2}}}({vv}) = W_0 \,"
            r"e^{{-\int{lim_mod}_{{v_0}}^{{{vv}}} {aa}(s)\,ds}}"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "ff1": S(_FUNC_POOL),
            "ff2": X(_FUNC_POOL, ("ff1",)),
            "vv": S(_VAR_POOL),
            "aa": S(_COEFF_POOL),
        },
    ),  # n_eff = 12 × 11 × 8 × 11 = 11,616
    Template(
        name="exact_ode",
        latex=(
            r"{MM}\,d{vv} + {NN}\,d{ww} = 0 \text{{ exact iff }}"
            r"\frac{{\partial {MM}}}{{\partial {ww}}}"
            r" = \frac{{\partial {NN}}}{{\partial {vv}}}"
        ),
        slots={
            "MM": S(_FUNC_POOL),
            "NN": X(_FUNC_POOL, ("MM",)),
            "vv": S(_VAR_POOL),
            "ww": X(_VAR_POOL, ("vv",)),
        },
    ),  # n_eff = 12 × 11 × 8 × 7 = 7,392
    Template(
        name="bernoulli_substitution",
        latex=(
            r"{ff}' + {aa}({vv}) {ff} = {bb}({vv}) {ff}^{{{nn}}},"
            r"\quad z = {ff}^{{1-{nn}}}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "vv": S(_VAR_POOL),
            "aa": S(_COEFF_POOL),
            "bb": X(_COEFF_POOL, ("aa",)),
            "nn": S(_IDX_POOL),
        },
    ),  # n_eff = 12 × 8 × 11 × 10 × 6 = 63,360
    Template(
        name="reduction_of_order",
        latex=(
            r"{ff2}({vv}) = {ff1}({vv}) \int"
            r"\frac{{e^{{-\int {aa}({vv})\,d{vv}}}}}{{{ff1}^2({vv})}} \, d{vv}"
        ),
        slots={
            "ff1": S(_FUNC_POOL),
            "ff2": X(_FUNC_POOL, ("ff1",)),
            "vv": S(_VAR_POOL),
            "aa": S(_COEFF_POOL),
        },
    ),  # n_eff = 12 × 11 × 8 × 11 = 11,616
    Template(
        name="variation_of_parameters_full",
        latex=(
            r"{ff1}_p = -{ff1} \int \frac{{{ff2} {gg}}}{{W}} \, d{vv}"
            r" + {ff2} \int \frac{{{ff1} {gg}}}{{W}} \, d{vv}"
        ),
        slots={
            "ff1": S(_FUNC_POOL),
            "ff2": X(_FUNC_POOL, ("ff1",)),
            "gg": X(_FUNC_POOL, ("ff1", "ff2")),
            "vv": S(_VAR_POOL),
        },
    ),  # n_eff = 12 × 11 × 10 × 8 = 10,560
    Template(
        name="characteristic_equation",
        latex=(
            r"{ff}'' + {aa} {ff}' + {bb} {ff} = 0:"
            r"\quad \lambda^2 + {aa} \lambda + {bb} = 0"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "aa": S(_COEFF_POOL),
            "bb": X(_COEFF_POOL, ("aa",)),
        },
    ),  # n_eff = 12 × 11 × 10 = 1,320
    Template(
        name="laplace_ode_transform",
        latex=(
            r"\bigl({ss}^2 + {aa} {ss} + {bb}\bigr) \hat{{{ff}}}({ss})"
            r" = \hat{{g}}({ss}) + ({ss} + {aa}) {ff}(0) + {ff}'(0)"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "ss": S(_VAR_POOL),
            "aa": S(_COEFF_POOL),
            "bb": X(_COEFF_POOL, ("aa",)),
        },
    ),  # n_eff = 12 × 8 × 11 × 10 = 10,560
]

# ---------------------------------------------------------------------------
# Part B2 — PDE Theory (8)
# ---------------------------------------------------------------------------

_TEMPLATES_B2: list[Template] = [
    Template(
        name="poisson_equation",
        latex=r"\nabla^2 {ff} = {gg} \quad \text{{in }} {dom}",
        slots={
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "dom": S(_DOMAIN_POOL),
        },
    ),  # n_eff = 12 × 11 × 7 = 924
    Template(
        name="helmholtz_equation",
        latex=r"\nabla^2 {ff} + {kk}^2 {ff} = 0 \quad \text{{in }} {dom}",
        slots={
            "ff": S(_FUNC_POOL),
            "kk": S(_PARAM_POOL),
            "dom": S(_DOMAIN_POOL),
        },
    ),  # n_eff = 12 × 10 × 7 = 840
    Template(
        name="transport_equation",
        latex=(
            r"\frac{{\partial {ff}}}{{\partial t}}"
            r" + {cc} \frac{{\partial {ff}}}{{\partial {vv}}} = 0"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "cc": S(_PARAM_POOL),
            "vv": S(_VAR_POOL),
        },
    ),  # n_eff = 12 × 10 × 8 = 960
    Template(
        name="diffusion_3d",
        latex=(
            r"\frac{{\partial {ff}}}{{\partial t}}"
            r" = {kap} \nabla^2 {ff} \quad \text{{in }} {dom}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "kap": S(_COEFF_POOL),
            "dom": S(_DOMAIN_POOL),
        },
    ),  # n_eff = 12 × 11 × 7 = 924
    Template(
        name="characteristics_first_order",
        latex=(
            r"{aa} \frac{{\partial {ff}}}{{\partial {vv1}}}"
            r" + {bb} \frac{{\partial {ff}}}{{\partial {vv2}}} = {cc}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "aa": S(_COEFF_POOL),
            "bb": X(_COEFF_POOL, ("aa",)),
            "cc": X(_COEFF_POOL, ("aa", "bb")),
            "vv1": S(_VAR_POOL),
            "vv2": X(_VAR_POOL, ("vv1",)),
        },
    ),  # n_eff = 12 × 11 × 10 × 9 × 8 × 7 = 665,280
    Template(
        name="biharmonic_equation",
        latex=r"\nabla^4 {ff} = 0 \quad \text{{in }} {dom}",
        slots={
            "ff": S(_FUNC_POOL),
            "dom": S(_DOMAIN_POOL),
        },
    ),  # n_eff = 12 × 7 = 84
    Template(
        name="cauchy_initial_data",
        latex=(
            r"{ff}({vv}, 0) = {gg}({vv}), \quad"
            r"\frac{{\partial {ff}}}{{\partial t}}({vv}, 0) = {hh}({vv})"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "hh": X(_FUNC_POOL, ("ff", "gg")),
            "vv": S(_VAR_POOL),
        },
    ),  # n_eff = 12 × 11 × 10 × 8 = 10,560
    Template(
        name="maximum_principle",
        latex=(
            r"{op}\, {ff} \leq 0 \text{{ in }} {dom}"
            r" \Rightarrow \max {ff} \text{{ attained on }} \partial {dom}"
        ),
        slots={
            "op": S(_OP_POOL),
            "ff": S(_FUNC_POOL),
            "dom": S(_DOMAIN_POOL),
        },
    ),  # n_eff = 6 × 12 × 7 = 504
]

# ---------------------------------------------------------------------------
# Part B3 — Transforms & Green's Functions (7)
# ---------------------------------------------------------------------------

_TEMPLATES_B3: list[Template] = [
    Template(
        name="laplace_transform_def",
        latex=(
            r"\mathcal{{L}}[{ff}]({ss}) = "
            r"\int_0^{{\infty}} e^{{-{ss} {vv}}} {ff}({vv}) \, d{vv}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "vv": S(_VAR_POOL),
            "ss": X(_VAR_POOL, ("vv",)),
        },
    ),  # n_eff = 12 × 8 × 7 = 672
    Template(
        name="laplace_derivative_rule",
        latex=(
            r"\mathcal{{L}}[{ff}^{{({nn})}}]({vv}) = "
            r"{vv}^{{{nn}}} F({vv}) - \sum{lim_mod}_{{k=0}}^{{{nn}-1}}"
            r" {vv}^{{{nn}-1-k}} {ff}^{{(k)}}(0)"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "ff": S(_FUNC_POOL),
            "vv": S(_VAR_POOL),
            "nn": S(_IDX_POOL),
        },
    ),  # n_eff = 12 × 8 × 6 = 576
    Template(
        name="laplace_convolution",
        latex=(
            r"\mathcal{{L}}[{ff} * {gg}]({vv}) = "
            r"\mathcal{{L}}[{ff}]({vv}) \cdot \mathcal{{L}}[{gg}]({vv})"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "vv": S(_VAR_POOL),
        },
    ),  # n_eff = 12 × 11 × 8 = 1,056
    Template(
        name="laplace_shift_theorem",
        latex=(
            r"\mathcal{{L}}[e^{{{aa} {vv}}} {ff}({vv})]({ss}) = "
            r"\mathcal{{L}}[{ff}]({ss} - {aa})"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "aa": S(_COEFF_POOL),
            "vv": S(_VAR_POOL),
            "ss": X(_VAR_POOL, ("vv",)),
        },
    ),  # n_eff = 12 × 11 × 8 × 7 = 7,392
    Template(
        name="greens_function_solution",
        latex=(r"{ff}({vv}) = \int{lim_mod}_{{{dom}}} G({vv}, {xi}) \, {gg}({xi}) \, d{xi}"),
        slots={
            "lim_mod": _LIM_MOD,
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "vv": S(_VAR_POOL),
            "xi": X(_VAR_POOL, ("vv",)),
            "dom": S(_DOMAIN_POOL),
        },
    ),  # n_eff = 12 × 11 × 8 × 7 × 7 = 51,744
    Template(
        name="duhamel_principle",
        latex=(
            r"{ff}({vv}, {tt}) = \int_0^{{{tt}}}"
            r" S({tt} - {tau}) \, {gg}({tau}) \, d{tau}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "vv": S(_VAR_POOL),
            "tt": X(_VAR_POOL, ("vv",)),
            "tau": X(_VAR_POOL, ("vv", "tt")),
        },
    ),  # n_eff = 12 × 11 × 8 × 7 × 6 = 44,352
    Template(
        name="fourier_eigenfunction_series",
        latex=(
            r"{ff}({vv}, t) = \sum{lim_mod}_{{{nn}=1}}^{{\infty}}"
            r" a_{{{nn}}} e^{{-{lam}_{{{nn}}} t}} \varphi_{{{nn}}}({vv})"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "ff": S(_FUNC_POOL),
            "vv": S(_VAR_POOL),
            "nn": S(_IDX_POOL),
            "lam": S(_PARAM_POOL),
        },
    ),  # n_eff = 12 × 8 × 6 × 10 = 5,760
]

# ---------------------------------------------------------------------------
# Part B4 — Boundary Value Problems & Sturm-Liouville (7)
# ---------------------------------------------------------------------------

_TEMPLATES_B4: list[Template] = [
    Template(
        name="sturm_liouville_regular",
        latex=(
            r"-\frac{{d}}{{d{vv}}}\!\left({pp}({vv})"
            r" \frac{{d{ff}}}{{d{vv}}}\right)"
            r" + {qq}({vv}) {ff} = {lam} {ww}({vv}) {ff}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "vv": S(_VAR_POOL),
            "pp": S(_COEFF_POOL),
            "qq": X(_COEFF_POOL, ("pp",)),
            "ww": X(_COEFF_POOL, ("pp", "qq")),
            "lam": S(_PARAM_POOL),
        },
    ),  # n_eff = 12 × 8 × 11 × 10 × 9 × 10 = 950,400
    Template(
        name="rayleigh_quotient",
        latex=(
            r"{lam} = \frac{{\int_a^b \bigl("
            r" {pp}|{ff}'|^2 + {qq}|{ff}|^2 \bigr) d{vv}}}"
            r"{{\int_a^b {ww} |{ff}|^2 \, d{vv}}}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "vv": S(_VAR_POOL),
            "lam": S(_PARAM_POOL),
            "pp": S(_COEFF_POOL),
            "qq": X(_COEFF_POOL, ("pp",)),
            "ww": X(_COEFF_POOL, ("pp", "qq")),
        },
    ),  # n_eff = 12 × 8 × 10 × 11 × 10 × 9 = 950,400
    Template(
        name="eigenfunction_orthogonality",
        latex=(
            r"\int{lim_mod}_{{{dom}}} \varphi_{{{mm}}} \varphi_{{{nn}}} {ww} \, d{vv}"
            r" = \delta_{{{mm}{nn}}} \|\varphi_{{{nn}}}\|^2"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "dom": S(_DOMAIN_POOL),
            "mm": S(_IDX_POOL),
            "nn": X(_IDX_POOL, ("mm",)),
            "ww": S(_COEFF_POOL),
            "vv": S(_VAR_POOL),
        },
    ),  # n_eff = 7 × 6 × 5 × 11 × 8 = 18,480
    Template(
        name="eigenfunction_expansion",
        latex=(
            r"{ff} = \sum{lim_mod}_{{{nn}}} c_{{{nn}}} \varphi_{{{nn}}}, \quad"
            r" c_{{{nn}}} = \frac{{\langle {gg}, \varphi_{{{nn}}} \rangle}}"
            r"{{\|\varphi_{{{nn}}}\|^2}}"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "nn": S(_IDX_POOL),
            "vv": S(_VAR_POOL),
        },
    ),  # n_eff = 12 × 11 × 6 × 8 = 6,336
    Template(
        name="dirichlet_bvp",
        latex=(
            r"{op}\, {ff} = {gg} \text{{ in }} {dom},"
            r" \quad {ff} = {hh} \text{{ on }} \partial {dom}"
        ),
        slots={
            "op": S(_OP_POOL),
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "hh": X(_FUNC_POOL, ("ff", "gg")),
            "dom": S(_DOMAIN_POOL),
        },
    ),  # n_eff = 6 × 12 × 11 × 10 × 7 = 55,440
    Template(
        name="neumann_bvp",
        latex=(
            r"{op}\, {ff} = {gg} \text{{ in }} {dom},"
            r" \quad \frac{{\partial {ff}}}{{\partial n}}"
            r" = {hh} \text{{ on }} \partial {dom}"
        ),
        slots={
            "op": S(_OP_POOL),
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "hh": X(_FUNC_POOL, ("ff", "gg")),
            "dom": S(_DOMAIN_POOL),
        },
    ),  # n_eff = 6 × 12 × 11 × 10 × 7 = 55,440
    Template(
        name="separation_of_variables",
        latex=r"{ff}({vv}, {tt}) = {gg}({vv}) \cdot {hh}({tt})",
        slots={
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "hh": X(_FUNC_POOL, ("ff", "gg")),
            "vv": S(_VAR_POOL),
            "tt": X(_VAR_POOL, ("vv",)),
        },
    ),  # n_eff = 12 × 11 × 10 × 8 × 7 = 73,920
]

# ---------------------------------------------------------------------------
# Part B5 — Series Methods & Special Functions (7)
# ---------------------------------------------------------------------------

_TEMPLATES_B5: list[Template] = [
    Template(
        name="power_series_ode",
        latex=(
            r"{ff}({vv}) = \sum{lim_mod}_{{{nn}=0}}^{{\infty}}"
            r" a_{{{nn}}} ({vv} - {vv}_0)^{{{nn}}}"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "ff": S(_FUNC_POOL),
            "vv": S(_VAR_POOL),
            "nn": S(_IDX_POOL),
        },
    ),  # n_eff = 12 × 8 × 6 = 576
    Template(
        name="frobenius_method",
        latex=(
            r"{ff}({vv}) = {vv}^{{{rr}}}"
            r" \sum{lim_mod}_{{{nn}=0}}^{{\infty}} a_{{{nn}}} {vv}^{{{nn}}}"
            r" \text{{ near regular singular point}}"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "ff": S(_FUNC_POOL),
            "vv": S(_VAR_POOL),
            "rr": S(_PARAM_POOL),
            "nn": S(_IDX_POOL),
        },
    ),  # n_eff = 12 × 8 × 10 × 6 = 5,760
    Template(
        name="frobenius_indicial",
        latex=(
            r"{ff}'' + \frac{{{pp}({vv})}}{{{vv}}} {ff}'"
            r" + \frac{{{qq}({vv})}}{{{vv}^2}} {ff} = 0:"
            r"\quad {rr}({rr}-1) + {pp}_0 {rr} + {qq}_0 = 0"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "vv": S(_VAR_POOL),
            "pp": S(_COEFF_POOL),
            "qq": X(_COEFF_POOL, ("pp",)),
            "rr": S(_PARAM_POOL),
        },
    ),  # n_eff = 12 × 8 × 11 × 10 × 10 = 105,600
    Template(
        name="bessel_equation",
        latex=(r"{vv}^2 {ff}'' + {vv} {ff}' + ({vv}^2 - {nn}^2) {ff} = 0"),
        slots={
            "ff": S(_FUNC_POOL),
            "vv": S(_VAR_POOL),
            "nn": S(_IDX_POOL),
        },
    ),  # n_eff = 12 × 8 × 6 = 576
    Template(
        name="legendre_equation",
        latex=(r"(1 - {vv}^2) {ff}'' - 2{vv} {ff}' + {nn}({nn}+1) {ff} = 0"),
        slots={
            "ff": S(_FUNC_POOL),
            "vv": S(_VAR_POOL),
            "nn": S(_IDX_POOL),
        },
    ),  # n_eff = 12 × 8 × 6 = 576
    Template(
        name="power_series_recurrence",
        latex=(
            r"{ff}'' + {lam} {ff} = 0:\quad"
            r" a_{{{nn}+2}} = -\frac{{{lam}}}{{({nn}+1)({nn}+2)}} a_{{{nn}}}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "lam": S(_PARAM_POOL),
            "nn": S(_IDX_POOL),
        },
    ),  # n_eff = 12 × 10 × 6 = 720
    Template(
        name="fuchs_indicial_pair",
        latex=(
            r"{rr1} - {rr2} \notin \mathbb{{Z}}"
            r" \Rightarrow {ff}_1({vv}) = {vv}^{{{rr1}}} \textstyle\sum a_k {vv}^k,"
            r"\quad {ff}_2({vv}) = {vv}^{{{rr2}}} \textstyle\sum b_k {vv}^k"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "vv": S(_VAR_POOL),
            "rr1": S(_PARAM_POOL),
            "rr2": X(_PARAM_POOL, ("rr1",)),
        },
    ),  # n_eff = 12 × 8 × 10 × 9 = 8,640
]

# ---------------------------------------------------------------------------
# Part B6 — Stability & Dynamical Systems (7)
# ---------------------------------------------------------------------------

_TEMPLATES_B6: list[Template] = [
    Template(
        name="autonomous_system",
        latex=(
            r"\frac{{d{vv1}}}{{dt}} = {ff}({vv1}, {vv2}), \quad"
            r"\frac{{d{vv2}}}{{dt}} = {gg}({vv1}, {vv2})"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "vv1": S(_VAR_POOL),
            "vv2": X(_VAR_POOL, ("vv1",)),
        },
    ),  # n_eff = 12 × 11 × 8 × 7 = 7,392
    Template(
        name="jacobian_linearization",
        latex=(
            r"J = \begin{{pmatrix}}"
            r"\partial_{{{vv1}}} {ff} & \partial_{{{vv2}}} {ff} \\"
            r"\partial_{{{vv1}}} {gg} & \partial_{{{vv2}}} {gg}"
            r"\end{{pmatrix}}"
            r"\bigg|_{{({vv1}^*, {vv2}^*)}}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "vv1": S(_VAR_POOL),
            "vv2": X(_VAR_POOL, ("vv1",)),
        },
    ),  # n_eff = 12 × 11 × 8 × 7 = 7,392
    Template(
        name="lyapunov_stability",
        latex=(
            r"{VV}({vv}) > 0,\; \nabla {VV}({vv}) \cdot {ff}({vv}) \leq 0"
            r" \Rightarrow 0 \text{{ Lyapunov stable for }} {ff}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "VV": X(_FUNC_POOL, ("ff",)),
            "vv": S(_VAR_POOL),
        },
    ),  # n_eff = 12 × 11 × 8 = 1,056
    Template(
        name="lyapunov_asymptotic",
        latex=(
            r"{VV}({vv}) > 0,\; \nabla {VV}({vv}) \cdot {ff}({vv}) < 0"
            r" \Rightarrow 0 \text{{ asymptotically stable for }} {ff}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "VV": X(_FUNC_POOL, ("ff",)),
            "vv": S(_VAR_POOL),
        },
    ),  # n_eff = 12 × 11 × 8 = 1,056
    Template(
        name="pitchfork_normal_form",
        latex=r"\dot{{{ff}}} = {rr} {vv} \mp {vv}^3",
        slots={
            "ff": S(_FUNC_POOL),
            "vv": S(_VAR_POOL),
            "rr": S(_PARAM_POOL),
        },
    ),  # n_eff = 12 × 8 × 10 = 960
    Template(
        name="hopf_normal_form",
        latex=(
            r"\dot{{{vv1}}} = {rr} {vv1} - {om} {vv2}"
            r" - ({vv1}^2+{vv2}^2){vv1},"
            r"\quad \dot{{{vv2}}} = {om} {vv1} + {rr} {vv2}"
            r" - ({vv1}^2+{vv2}^2){vv2}"
        ),
        slots={
            "vv1": S(_VAR_POOL),
            "vv2": X(_VAR_POOL, ("vv1",)),
            "rr": S(_PARAM_POOL),
            "om": X(_PARAM_POOL, ("rr",)),
        },
    ),  # n_eff = 8 × 7 × 10 × 9 = 5,040
    Template(
        name="poincare_bendixson",
        latex=(
            r"\dot{{{vv1}}} = {ff}({vv1},{vv2}),\;"
            r"\dot{{{vv2}}} = {gg}({vv1},{vv2})"
            r"\text{{ bounded, no eq. pt}} \Rightarrow \text{{closed orbit}}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "vv1": S(_VAR_POOL),
            "vv2": X(_VAR_POOL, ("vv1",)),
        },
    ),  # n_eff = 12 × 11 × 8 × 7 = 7,392
]

# ---------------------------------------------------------------------------
# Part C — High-n_eff function-pair templates (6)
# ---------------------------------------------------------------------------

_TEMPLATES_C: list[Template] = [
    Template(
        name="fn_operator_eigenvalue",
        latex=(
            r"{fn1}\!\bigl({op}[{fn2}({ff})]\bigr)"
            r" = {fn3}\!\bigl({lam} {ff}\bigr)"
            r" \quad \text{{in }} {sp}"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "fn3": _FN_SLOT,
            "op": S(_OP_POOL),
            "ff": S(_FUNC_POOL),
            "lam": S(_PARAM_POOL),
            "sp": S(_SPACE_POOL),
        },
    ),  # n_eff = 100³ × 6 × 12 × 10 × 6 = 4,320,000,000
    Template(
        name="fn_ode_solution_pair",
        latex=(
            r"{fn1}\!\left({ff}' + {pp} {ff}\right)"
            r" = {fn2}\!\left({ff}\right)"
            r" \quad ({vv} \in {sp})"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "ff": S(_FUNC_POOL),
            "pp": S(_COEFF_POOL),
            "vv": S(_VAR_POOL),
            "sp": S(_SPACE_POOL),
        },
    ),  # n_eff = 100² × 12 × 11 × 8 × 6 = 63,360,000
    Template(
        name="fn_transform_pair",
        latex=(
            r"{fn1}\!\bigl(\mathcal{{L}}[{ff}]({ss})\bigr)"
            r" = {fn2}\!\bigl({ff}({vv})\bigr)"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "ff": S(_FUNC_POOL),
            "vv": S(_VAR_POOL),
            "ss": X(_VAR_POOL, ("vv",)),
        },
    ),  # n_eff = 100² × 12 × 8 × 7 = 67,200,000
    Template(
        name="fn_green_composition",
        latex=(
            r"{fn1}\!\left(\int G({vv},{xi})\,"
            r"{fn2}\!\bigl({gg}({xi})\bigr)\, d{xi}\right)"
            r" = {fn3}\!\bigl({ff}({vv})\bigr)"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "fn3": _FN_SLOT,
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "vv": S(_VAR_POOL),
            "xi": X(_VAR_POOL, ("vv",)),
        },
    ),  # n_eff = 100³ × 12 × 11 × 8 × 7 = 73,920,000,000
    Template(
        name="fn_stability_lyapunov",
        latex=(
            r"{fn1}\!\bigl(\dot{{{VV}}}({vv})\bigr)"
            r" \leq {fn2}\!\bigl({VV}({vv})\bigr)"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "VV": S(_FUNC_POOL),
            "vv": S(_VAR_POOL),
        },
    ),  # n_eff = 100² × 12 × 8 = 9,600,000
    Template(
        name="fn_pde_solution_form",
        latex=(
            r"{fn1}\!\bigl({ff}({vv}, {tt})\bigr)"
            r" = {fn2}\!\bigl({gg}({vv})\bigr)"
            r" \cdot {fn3}\!\bigl({hh}({tt})\bigr)"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "fn3": _FN_SLOT,
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "hh": X(_FUNC_POOL, ("ff", "gg")),
            "vv": S(_VAR_POOL),
            "tt": X(_VAR_POOL, ("vv",)),
        },
    ),  # n_eff = 100³ × 12 × 11 × 10 × 8 × 7 = 73,920,000,000
]

# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

_DIFFEQ_TEMPLATES: list[Template] = (
    _TEMPLATES_A
    + _TEMPLATES_B1
    + _TEMPLATES_B2
    + _TEMPLATES_B3
    + _TEMPLATES_B4
    + _TEMPLATES_B5
    + _TEMPLATES_B6
    + _TEMPLATES_C
)

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("differential_equations", _DIFFEQ_TEMPLATES)
