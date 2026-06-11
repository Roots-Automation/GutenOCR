"""Stochastic processes domain: Brownian motion, Itô calculus, SDEs,
continuous-time martingales, Markov generators, jump processes, control."""

from __future__ import annotations

from .._template_dsl import _FN_SLOT, _LIM_MOD, S, Template, X
from .._vocab import _FUNC_NAMES as _FUNC_POOL
from ._config import register_domain

# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------

_PROC_POOL: tuple[str, ...] = (
    "X",
    "Y",
    "Z",
    "M",
    "N",
    "S",
    "V",
    r"\xi",
    r"\eta",
)  # 9

_BM_POOL: tuple[str, ...] = (
    "B",
    "W",
    r"\tilde{B}",
    r"\tilde{W}",
    r"B^{(1)}",
    r"W^{(1)}",
)  # 6

_TIME_POOL: tuple[str, ...] = (
    "t",
    "s",
    "u",
    "T",
    "r",
    r"t_0",
    r"t_1",
)  # 7

_STOP_POOL: tuple[str, ...] = (
    r"\tau",
    r"\sigma",
    r"\rho",
    r"\zeta",
    "T",
    "S",
)  # 6

_COEFF_POOL: tuple[str, ...] = (
    r"\mu",
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\kappa",
    r"\theta",
    "a",
    "b",
    "c",
    "k",
    "r",
)  # 11

_DIFF_POOL: tuple[str, ...] = (
    r"\sigma",
    r"\varepsilon",
    r"\nu",
    "s",
    "v",
    r"\delta",
)  # 6


_FILT_POOL: tuple[str, ...] = (
    r"\mathcal{F}",
    r"\mathcal{G}",
    r"\mathcal{H}",
    r"\mathbb{F}",
    r"\mathbb{G}",
)  # 5

_IDX_POOL: tuple[str, ...] = (
    "n",
    "m",
    "k",
    "j",
    "N",
    "M",
)  # 6

_MEAS_POOL: tuple[str, ...] = (
    r"\mathbb{P}",
    r"\mathbb{Q}",
    r"\tilde{\mathbb{P}}",
    r"\mathbb{P}^*",
)  # 4

# ---------------------------------------------------------------------------
# Part A — Brownian Motion (10 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_A: list[Template] = [
    Template(
        name="brownian_increments",
        latex=(
            r"{bm}_{{{t2}}} - {bm}_{{{t1}}}"
            r" \sim \mathcal{{N}}(0,\, {t2} - {t1})"
        ),
        slots={
            "bm": S(_BM_POOL),
            "t1": S(_TIME_POOL),
            "t2": X(_TIME_POOL, ("t1",)),
        },
    ),  # 6×7×6 = 252
    Template(
        name="brownian_variance",
        latex=r"\mathbb{{E}}\!\left[{bm}_{{{t}}}^2\right] = {t}",
        slots={
            "bm": S(_BM_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 6×7 = 42
    Template(
        name="brownian_covariance",
        latex=(
            r"\mathbb{{E}}\!\left[{bm}_{{{t1}}} {bm}_{{{t2}}}\right]"
            r" = \min({t1},\, {t2})"
        ),
        slots={
            "bm": S(_BM_POOL),
            "t1": S(_TIME_POOL),
            "t2": X(_TIME_POOL, ("t1",)),
        },
    ),  # 6×7×6 = 252
    Template(
        name="quadratic_variation",
        latex=r"[{bm}]_{{{t}}} = {t}",
        slots={
            "bm": S(_BM_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 6×7 = 42
    Template(
        name="brownian_scaling",
        latex=(
            r"{coeff}\, {bm}_{{{{{t}}}/{coeff}^2}}"
            r" \stackrel{{d}}{{=}} {bm}_{{{t}}}"
        ),
        slots={
            "bm": S(_BM_POOL),
            "coeff": S(_COEFF_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 6×11×7 = 462
    Template(
        name="brownian_time_reversal",
        latex=(
            r"{bm}_{{{T}}} - {bm}_{{{T}-{t}}}"
            r" \stackrel{{d}}{{=}} {bm}_{{{t}}},"
            r"\quad 0 \le {t} \le {T}"
        ),
        slots={
            "bm": S(_BM_POOL),
            "T": S(_STOP_POOL),
            "t": X(_TIME_POOL, ("T",)),
        },
    ),  # 6×6×5 = 180  (X excludes T from _TIME_POOL; _STOP_POOL and _TIME_POOL overlap on "T","S")
    Template(
        name="reflection_principle",
        latex=(
            r"\mathbb{{P}}\!\left(\max_{{0 \le {t1} \le {t2}}}"
            r" {bm}_{{{t1}}} \ge {coeff}\right)"
            r" = 2\,\mathbb{{P}}\!\left({bm}_{{{t2}}} \ge {coeff}\right)"
        ),
        slots={
            "bm": S(_BM_POOL),
            "t1": S(_TIME_POOL),
            "t2": X(_TIME_POOL, ("t1",)),
            "coeff": S(_COEFF_POOL),
        },
    ),  # 6×7×6×11 = 2,772
    Template(
        name="levy_characterization",
        latex=(
            r"{proc}_0 = 0,\quad [{proc}]_{{{t}}} = {t}"
            r" \;\Longrightarrow\; {proc}"
            r" \text{{ is a Brownian motion}}"
        ),
        slots={
            "proc": S(_PROC_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 9×7 = 63
    Template(
        name="ito_martingale_part",
        latex=(
            r"{func}({bm}_{{{t}}})"
            r" - \int_0^{{{t}}} \tfrac{{1}}{{2}} {func}''({bm}_{{{s}}})\, d{s}"
            r" \text{{ is a local martingale}}"
        ),
        slots={
            "func": S(_FUNC_POOL),
            "bm": S(_BM_POOL),
            "t": S(_TIME_POOL),
            "s": X(_TIME_POOL, ("t",)),
        },
    ),  # 8×6×7×6 = 2,016
    Template(
        name="correlated_brownians",
        latex=(r"d{bm1}_{{{t}}}\, d{bm2}_{{{t}}} = {coeff}\, d{t}"),
        slots={
            "bm1": S(_BM_POOL),
            "bm2": X(_BM_POOL, ("bm1",)),
            "t": S(_TIME_POOL),
            "coeff": S(_COEFF_POOL),
        },
    ),  # 6×5×7×11 = 2,310
]

# ---------------------------------------------------------------------------
# Part B1 — Itô Calculus (8 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B1: list[Template] = [
    Template(
        name="ito_isometry",
        latex=(
            r"\mathbb{{E}}\!\left[\left(\int_0^{{{t}}} {func}({s})\,"
            r" d{bm}_{{{s}}}\right)^{{\!2}}\right]"
            r" = \mathbb{{E}}\!\left[\int_0^{{{t}}} {func}({s})^2\, d{s}\right]"
        ),
        slots={
            "func": S(_FUNC_POOL),
            "bm": S(_BM_POOL),
            "t": S(_TIME_POOL),
            "s": X(_TIME_POOL, ("t",)),
        },
    ),  # 8×6×7×6 = 2,016
    Template(
        name="ito_formula_bm",
        latex=(
            r"d{func}({bm}_{{{t}}})"
            r" = {func}'({bm}_{{{t}}})\, d{bm}_{{{t}}}"
            r" + \tfrac{{1}}{{2}} {func}''({bm}_{{{t}}})\, d{t}"
        ),
        slots={
            "func": S(_FUNC_POOL),
            "bm": S(_BM_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 8×6×7 = 336
    Template(
        name="ito_formula_general",
        latex=(
            r"d{func}({t}, {proc}_{{{t}}})"
            r" = \partial_{{{t}}} {func}\, d{t}"
            r" + \partial_x {func}\, d{proc}_{{{t}}}"
            r" + \tfrac{{1}}{{2}} \partial_{{xx}} {func}\,"
            r" (d{proc}_{{{t}}})^2"
        ),
        slots={
            "func": S(_FUNC_POOL),
            "proc": S(_PROC_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 8×9×7 = 504
    Template(
        name="ito_product_rule",
        latex=(
            r"d({proc1}_{{{t}}} {proc2}_{{{t}}})"
            r" = {proc1}_{{{t}}}\, d{proc2}_{{{t}}}"
            r" + {proc2}_{{{t}}}\, d{proc1}_{{{t}}}"
            r" + d[{proc1}, {proc2}]_{{{t}}}"
        ),
        slots={
            "proc1": S(_PROC_POOL),
            "proc2": X(_PROC_POOL, ("proc1",)),
            "t": S(_TIME_POOL),
        },
    ),  # 9×8×7 = 504
    Template(
        name="quadratic_covariation_def",
        latex=(
            r"[{proc1}, {proc2}]_{{{t}}}"
            r" = {proc1}_{{{t}}} {proc2}_{{{t}}}"
            r" - \int_0^{{{t}}} {proc1}_{{{s}}}\, d{proc2}_{{{s}}}"
            r" - \int_0^{{{t}}} {proc2}_{{{s}}}\, d{proc1}_{{{s}}}"
        ),
        slots={
            "proc1": S(_PROC_POOL),
            "proc2": X(_PROC_POOL, ("proc1",)),
            "t": S(_TIME_POOL),
            "s": X(_TIME_POOL, ("t",)),
        },
    ),  # 9×8×7×6 = 3,024
    Template(
        name="stratonovich_ito_relation",
        latex=(
            r"\int_0^{{{t}}} {func}({s})\, \circ\, d{bm}_{{{s}}}"
            r" = \int_0^{{{t}}} {func}({s})\, d{bm}_{{{s}}}"
            r" + \tfrac{{1}}{{2}} [{func}, {bm}]_{{{t}}}"
        ),
        slots={
            "func": S(_FUNC_POOL),
            "bm": S(_BM_POOL),
            "t": S(_TIME_POOL),
            "s": X(_TIME_POOL, ("t",)),
        },
    ),  # 8×6×7×6 = 2,016
    Template(
        name="martingale_representation",
        latex=(
            r"{proc}_{{{t}}} = {proc}_0"
            r" + \int_0^{{{t}}} {func}({s})\, d{bm}_{{{s}}}"
        ),
        slots={
            "proc": S(_PROC_POOL),
            "func": S(_FUNC_POOL),
            "bm": S(_BM_POOL),
            "t": S(_TIME_POOL),
            "s": X(_TIME_POOL, ("t",)),
        },
    ),  # 9×8×6×7×6 = 18,144
    Template(
        name="ito_formula_two_var",
        latex=(
            r"d{func}({proc1}_{{{t}}}, {proc2}_{{{t}}})"
            r" = \partial_1 {func}\, d{proc1}_{{{t}}}"
            r" + \partial_2 {func}\, d{proc2}_{{{t}}}"
            r" + \tfrac{{1}}{{2}}\bigl("
            r"\partial_{{11}} {func}\, d[{proc1}]_{{{t}}}"
            r" + 2\,\partial_{{12}} {func}\, d[{proc1},{proc2}]_{{{t}}}"
            r" + \partial_{{22}} {func}\, d[{proc2}]_{{{t}}}"
            r"\bigr)"
        ),
        slots={
            "func": S(_FUNC_POOL),
            "proc1": S(_PROC_POOL),
            "proc2": X(_PROC_POOL, ("proc1",)),
            "t": S(_TIME_POOL),
        },
    ),  # 8×9×8×7 = 4,032
]

# ---------------------------------------------------------------------------
# Part B2 — Stochastic Differential Equations (8 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B2: list[Template] = [
    Template(
        name="sde_general_form",
        latex=(
            r"d{proc}_{{{t}}} = {drift}({proc}_{{{t}}}, {t})\, d{t}"
            r" + {diff}({proc}_{{{t}}}, {t})\, d{bm}_{{{t}}}"
        ),
        slots={
            "proc": S(_PROC_POOL),
            "drift": S(_FUNC_POOL),
            "diff": X(_FUNC_POOL, ("drift",)),
            "bm": S(_BM_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 9×8×7×6×7 = 21,168
    Template(
        name="geometric_brownian_motion_sde",
        latex=(
            r"d{proc}_{{{t}}} = {drift}\, {proc}_{{{t}}}\, d{t}"
            r" + {diff}\, {proc}_{{{t}}}\, d{bm}_{{{t}}}"
        ),
        slots={
            "proc": S(_PROC_POOL),
            "drift": S(_COEFF_POOL),
            "diff": S(_DIFF_POOL),
            "bm": S(_BM_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 9×11×6×6×7 = 24,948
    Template(
        name="gbm_solution",
        latex=(
            r"{proc}_{{{t}}} = {proc}_0 \exp\!\left("
            r"\left({drift} - \tfrac{{{diff}^2}}{{2}}\right){t}"
            r" + {diff}\, {bm}_{{{t}}}\right)"
        ),
        slots={
            "proc": S(_PROC_POOL),
            "drift": S(_COEFF_POOL),
            "diff": S(_DIFF_POOL),
            "bm": S(_BM_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 9×11×6×6×7 = 24,948
    Template(
        name="ornstein_uhlenbeck_sde",
        latex=(
            r"d{proc}_{{{t}}} = {kappa}({theta} - {proc}_{{{t}}})\, d{t}"
            r" + {diff}\, d{bm}_{{{t}}}"
        ),
        slots={
            "proc": S(_PROC_POOL),
            "kappa": S(_COEFF_POOL),
            "theta": X(_COEFF_POOL, ("kappa",)),
            "diff": S(_DIFF_POOL),
            "bm": S(_BM_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 9×11×10×6×6×7 = 249,480
    Template(
        name="sde_lipschitz_condition",
        latex=(
            r"\lvert {drift}(x) - {drift}(y) \rvert"
            r" + \lvert {diff}(x) - {diff}(y) \rvert"
            r" \le {coeff} \lvert x - y \rvert"
        ),
        slots={
            "drift": S(_FUNC_POOL),
            "diff": X(_FUNC_POOL, ("drift",)),
            "coeff": S(_COEFF_POOL),
        },
    ),  # 8×7×11 = 616
    Template(
        name="sde_linear_solution",
        latex=(
            r"{proc}_{{{t}}} = \Phi_{{{t}}}\!\left("
            r"{proc}_0"
            r" + \int_0^{{{t}}} \Phi_{{{s}}}^{{-1}} {drift}({s})\, d{s}"
            r" + \int_0^{{{t}}} \Phi_{{{s}}}^{{-1}} {diff}({s})\, d{bm}_{{{s}}}"
            r"\right)"
        ),
        slots={
            "proc": S(_PROC_POOL),
            "drift": S(_FUNC_POOL),
            "diff": X(_FUNC_POOL, ("drift",)),
            "bm": S(_BM_POOL),
            "t": S(_TIME_POOL),
            "s": X(_TIME_POOL, ("t",)),
        },
    ),  # 9×8×7×6×7×6 = 127,008
    Template(
        name="sde_stopped_process",
        latex=r"{proc}^{{{stop}}}_{{{t}}} = {proc}_{{{t} \wedge {stop}}}",
        slots={
            "proc": S(_PROC_POOL),
            "stop": S(_STOP_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 9×6×7 = 378
    Template(
        name="sde_pathwise_solution",
        latex=(
            r"\mathbb{{E}}\!\left["
            r"\int_0^{{{t}}}\!\left("
            r"{drift}({proc}_{{{s}}})^2"
            r" + {diff}({proc}_{{{s}}})^2"
            r"\right)d{s}\right] < \infty"
        ),
        slots={
            "proc": S(_PROC_POOL),
            "drift": S(_FUNC_POOL),
            "diff": X(_FUNC_POOL, ("drift",)),
            "t": S(_TIME_POOL),
            "s": X(_TIME_POOL, ("t",)),
        },
    ),  # 9×8×7×7×6 = 21,168
]

# ---------------------------------------------------------------------------
# Part B3 — Continuous-Time Martingale Theory (8 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B3: list[Template] = [
    Template(
        name="martingale_def",
        latex=(
            r"\mathbb{{E}}\!\left[{proc}_{{{t2}}} \mid {filt}_{{{t1}}}\right]"
            r" = {proc}_{{{t1}}}"
        ),
        slots={
            "proc": S(_PROC_POOL),
            "filt": S(_FILT_POOL),
            "t1": S(_TIME_POOL),
            "t2": X(_TIME_POOL, ("t1",)),
        },
    ),  # 9×5×7×6 = 1,890
    Template(
        name="supermartingale_def",
        latex=(
            r"\mathbb{{E}}\!\left[{proc}_{{{t2}}} \mid {filt}_{{{t1}}}\right]"
            r" \le {proc}_{{{t1}}}"
        ),
        slots={
            "proc": S(_PROC_POOL),
            "filt": S(_FILT_POOL),
            "t1": S(_TIME_POOL),
            "t2": X(_TIME_POOL, ("t1",)),
        },
    ),  # 9×5×7×6 = 1,890
    Template(
        name="doob_optional_stopping",
        latex=(
            r"\mathbb{{E}}\!\left[{proc}_{{{stop}}}\right]"
            r" = \mathbb{{E}}\!\left[{proc}_0\right]"
        ),
        slots={
            "proc": S(_PROC_POOL),
            "stop": S(_STOP_POOL),
        },
    ),  # 9×6 = 54
    Template(
        name="doob_maximal_inequality",
        latex=(
            r"\mathbb{{P}}\!\left(\sup_{{0 \le {t1} \le {t2}}}"
            r" \lvert {proc}_{{{t1}}} \rvert \ge {coeff}\right)"
            r" \le \frac{{\mathbb{{E}}\!\left[\lvert {proc}_{{{t2}}} \rvert\right]}}"
            r"{{{coeff}}}"
        ),
        slots={
            "proc": S(_PROC_POOL),
            "coeff": S(_COEFF_POOL),
            "t1": S(_TIME_POOL),
            "t2": X(_TIME_POOL, ("t1",)),
        },
    ),  # 9×11×7×6 = 4,158
    Template(
        name="doob_lp_inequality",
        latex=(
            r"\mathbb{{E}}\!\left[\sup_{{0 \le {t} \le {T}}}"
            r" \lvert {proc}_{{{t}}} \rvert^{{{idx}}}\right]^{{1/{idx}}}"
            r" \le \frac{{{idx}}}{{{idx}-1}}"
            r"\,\mathbb{{E}}\!\left[\lvert {proc}_{{{T}}} \rvert^{{{idx}}}"
            r"\right]^{{1/{idx}}}"
        ),
        slots={
            "proc": S(_PROC_POOL),
            "t": S(_TIME_POOL),
            "T": X(_TIME_POOL, ("t",)),
            "idx": S(_IDX_POOL),
        },
    ),  # 9×7×6×6 = 2,268
    Template(
        name="local_martingale_def",
        latex=(
            r"\exists\, {stop}_{{{idx}}} \uparrow \infty"
            r" \text{{ s.t. }} {proc}^{{{{{stop}}}_{{{idx}}}}}"
            r" \text{{ is a martingale}}"
        ),
        slots={
            "proc": S(_PROC_POOL),
            "stop": S(_STOP_POOL),
            "idx": S(_IDX_POOL),
        },
    ),  # 9×6×6 = 324
    Template(
        name="girsanov_radon_nikodym",
        latex=(
            r"\frac{{d{meas2}}}{{d{meas1}}}\bigg|_{{{filt}_{{{t}}}}}"
            r" = \exp\!\left(\int_0^{{{t}}} {coeff}_{{{s}}}\, d{bm}_{{{s}}}"
            r" - \tfrac{{1}}{{2}} \int_0^{{{t}}} {coeff}_{{{s}}}^2\, d{s}\right)"
        ),
        slots={
            "meas1": S(_MEAS_POOL),
            "meas2": X(_MEAS_POOL, ("meas1",)),
            "filt": S(_FILT_POOL),
            "coeff": S(_COEFF_POOL),
            "bm": S(_BM_POOL),
            "t": S(_TIME_POOL),
            "s": X(_TIME_POOL, ("t",)),
        },
    ),  # 4×3×5×11×6×7×6 = 166,320
    Template(
        name="girsanov_new_bm",
        latex=(
            r"{bm2}_{{{t}}} = {bm1}_{{{t}}}"
            r" - \int_0^{{{t}}} {coeff}_{{{s}}}\, d{s}"
            r" \text{{ is a }}{meas}\text{{-Brownian motion}}"
        ),
        slots={
            "bm1": S(_BM_POOL),
            "bm2": X(_BM_POOL, ("bm1",)),
            "coeff": S(_COEFF_POOL),
            "meas": S(_MEAS_POOL),
            "t": S(_TIME_POOL),
            "s": X(_TIME_POOL, ("t",)),
        },
    ),  # 6×5×11×4×7×6 = 55,440
]

# ---------------------------------------------------------------------------
# Part B4 — Markov Processes & Generators (8 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B4: list[Template] = [
    Template(
        name="markov_semigroup_def",
        latex=(
            r"P_{{{t}}} {func}(x)"
            r" = \mathbb{{E}}^x\!\left[{func}({proc}_{{{t}}})\right]"
        ),
        slots={
            "func": S(_FUNC_POOL),
            "proc": S(_PROC_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 8×9×7 = 504
    Template(
        name="generator_def",
        latex=(
            r"\mathcal{{L}}{func}(x)"
            r" = \lim_{{{t} \downarrow 0}}"
            r" \frac{{P_{{{t}}} {func}(x) - {func}(x)}}{{{t}}}"
        ),
        slots={
            "func": S(_FUNC_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 8×7 = 56
    Template(
        name="diffusion_generator_1d",
        latex=(
            r"\mathcal{{L}} = {drift}(x) \frac{{d}}{{dx}}"
            r" + \tfrac{{1}}{{2}} {diff}(x)^2 \frac{{d^2}}{{dx^2}}"
        ),
        slots={
            "drift": S(_FUNC_POOL),
            "diff": X(_FUNC_POOL, ("drift",)),
        },
    ),  # 8×7 = 56
    Template(
        name="fokker_planck_1d",
        latex=(
            r"\frac{{\partial {func}}}{{\partial {t}}}"
            r" = -\frac{{\partial}}{{\partial x}}"
            r"\!\left({drift}(x)\, {func}\right)"
            r" + \tfrac{{1}}{{2}} \frac{{\partial^2}}{{\partial x^2}}"
            r"\!\left({diff}(x)^2\, {func}\right)"
        ),
        slots={
            "func": S(_FUNC_POOL),
            "drift": X(_FUNC_POOL, ("func",)),
            "diff": X(_FUNC_POOL, ("func", "drift")),
            "t": S(_TIME_POOL),
        },
    ),  # 8×7×6×7 = 2,352
    Template(
        name="kolmogorov_backward",
        latex=(
            r"\frac{{\partial {func}}}{{\partial {t}}}"
            r" + \mathcal{{L}}{func} = 0,"
            r"\quad {func}(x, {T}) = {func2}(x)"
        ),
        slots={
            "func": S(_FUNC_POOL),
            "func2": X(_FUNC_POOL, ("func",)),
            "t": S(_TIME_POOL),
            "T": X(_TIME_POOL, ("t",)),
        },
    ),  # 8×7×7×6 = 2,352
    Template(
        name="feynman_kac",
        latex=(
            r"{func}(x, {t})"
            r" = \mathbb{{E}}^x\!\left["
            r"e^{{-\int{lim_mod}_{{{t}}}^{{{T}}} {coeff}({proc}_{{{s}}})\, d{s}}}"
            r"\, {func2}({proc}_{{{T}}})"
            r"\right]"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "func": S(_FUNC_POOL),
            "func2": X(_FUNC_POOL, ("func",)),
            "proc": S(_PROC_POOL),
            "coeff": S(_COEFF_POOL),
            "t": S(_TIME_POOL),
            "T": X(_TIME_POOL, ("t",)),
            "s": X(_TIME_POOL, ("t", "T")),
        },
    ),  # 8×7×9×11×7×6×5 = 1,481,760
    Template(
        name="dynkin_formula",
        latex=(
            r"\mathbb{{E}}^x\!\left[{func}({proc}_{{{stop}}})\right]"
            r" = {func}(x)"
            r" + \mathbb{{E}}^x\!\left["
            r"\int_0^{{{stop}}} \mathcal{{L}}{func}({proc}_{{{t}}})\, d{t}"
            r"\right]"
        ),
        slots={
            "func": S(_FUNC_POOL),
            "proc": S(_PROC_POOL),
            "stop": S(_STOP_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 8×9×6×7 = 3,024
    Template(
        name="invariant_measure",
        latex=(
            r"\int P_{{{t}}} {func}(x)\, {meas}(dx)"
            r" = \int {func}(x)\, {meas}(dx)"
            r"\quad \forall\, {t} > 0"
        ),
        slots={
            "func": S(_FUNC_POOL),
            "meas": S(_MEAS_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 8×4×7 = 224
]

# ---------------------------------------------------------------------------
# Part B5 — Poisson Processes & Jump Processes (8 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B5: list[Template] = [
    Template(
        name="poisson_process_increment",
        latex=(
            r"{proc}_{{{t}}} - {proc}_{{{s}}}"
            r" \sim \mathrm{{Poisson}}\!\left({coeff}({t} - {s})\right)"
        ),
        slots={
            "proc": S(_PROC_POOL),
            "coeff": S(_COEFF_POOL),
            "t": S(_TIME_POOL),
            "s": X(_TIME_POOL, ("t",)),
        },
    ),  # 9×11×7×6 = 4,158
    Template(
        name="compound_poisson_process",
        latex=(r"{proc1}_{{{t}}} = \sum{lim_mod}_{{k=1}}^{{{proc2}_{{{t}}}}} {func}_k"),
        slots={
            "lim_mod": _LIM_MOD,
            "proc1": S(_PROC_POOL),
            "proc2": X(_PROC_POOL, ("proc1",)),
            "func": S(_FUNC_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 9×8×8×7 = 4,032
    Template(
        name="compensated_poisson_martingale",
        latex=(
            r"\tilde{{{proc}}}_{{{t}}} = {proc}_{{{t}}}"
            r" - {coeff}\, {t}"
            r"\text{{ is a martingale}}"
        ),
        slots={
            "proc": S(_PROC_POOL),
            "coeff": S(_COEFF_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 9×11×7 = 693
    Template(
        name="jump_sde_form",
        latex=(
            r"d{proc}_{{{t}}} = {drift}({proc}_{{{t}}})\, d{t}"
            r" + {diff}({proc}_{{{t}}})\, d{bm}_{{{t}}}"
            r" + {coeff}({proc}_{{{t}-}})\, d{jump}_{{{t}}}"
        ),
        slots={
            "proc": S(_PROC_POOL),
            "drift": S(_FUNC_POOL),
            "diff": X(_FUNC_POOL, ("drift",)),
            "coeff": X(_FUNC_POOL, ("drift", "diff")),
            "bm": S(_BM_POOL),
            "jump": X(_PROC_POOL, ("proc",)),
            "t": S(_TIME_POOL),
        },
    ),  # 9×8×7×6×6×8×7 = 1,354,752
    Template(
        name="levy_khintchine_formula",
        latex=(
            r"\Psi({coeff}) = i{drift}\,{coeff}"
            r" - \tfrac{{1}}{{2}} {diff}^2 {coeff}^2"
            r" + \int \!\left(e^{{i{coeff} z}} - 1 - i{coeff} z\right) \nu(dz)"
        ),
        slots={
            "coeff": S(_COEFF_POOL),
            "drift": X(_COEFF_POOL, ("coeff",)),
            "diff": S(_DIFF_POOL),
        },
    ),  # 11×10×6 = 660
    Template(
        name="poisson_intensity",
        latex=(r"\mathbb{{E}}\!\left[{proc}_{{{t}}}\right] = {coeff}\, {t}"),
        slots={
            "proc": S(_PROC_POOL),
            "coeff": S(_COEFF_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 9×11×7 = 693
    Template(
        name="inter_arrival_exponential",
        latex=(
            r"{stop}_{{{idx}}} \sim \mathrm{{Exp}}({coeff})"
            r"\text{{ i.i.d.}},\quad "
            r"N_{{{t}}} = \max\left\{{k : {stop}_1 + \cdots"
            r" + {stop}_k \le {t}\right\}}"
        ),
        slots={
            "stop": S(_STOP_POOL),
            "coeff": S(_COEFF_POOL),
            "idx": S(_IDX_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 6×11×6×7 = 2,772
    Template(
        name="levy_ito_decomposition",
        latex=(
            r"{proc}_{{{t}}} = {drift}\, {t}"
            r" + {diff}\, {bm}_{{{t}}}"
            r" + \int{lim_mod}_{{|z|<1}} z\, \tilde{{N}}({t},\, dz)"
            r" + \int{lim_mod}_{{|z|\ge 1}} z\, N({t},\, dz)"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "proc": S(_PROC_POOL),
            "drift": S(_COEFF_POOL),
            "diff": S(_DIFF_POOL),
            "bm": S(_BM_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 9×11×6×6×7 = 24,948
]

# ---------------------------------------------------------------------------
# Part B6 — Stochastic Control & Filtering (8 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B6: list[Template] = [
    Template(
        name="hjb_equation",
        latex=(
            r"\frac{{\partial {func}}}{{\partial {t}}}"
            r" + \max_{{u}}\left["
            r"{drift}(x, u) \cdot \nabla_x {func}"
            r" + \tfrac{{1}}{{2}} \mathrm{{tr}}"
            r"(\sigma\sigma^T D^2 {func})\right] = 0"
        ),
        slots={
            "func": S(_FUNC_POOL),
            "drift": X(_FUNC_POOL, ("func",)),
            "t": S(_TIME_POOL),
        },
    ),  # 8×7×7 = 392
    Template(
        name="value_function_bellman",
        latex=(
            r"{func}(x, {t}) = \sup_u \mathbb{{E}}\!\left["
            r"\int{lim_mod}_{{{t}}}^{{{T}}} {coeff}({proc}_{{{s}}}, u_{{{s}}})\, d{s}"
            r" + {func2}({proc}_{{{T}}})"
            r"\right]"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "func": S(_FUNC_POOL),
            "func2": X(_FUNC_POOL, ("func",)),
            "proc": S(_PROC_POOL),
            "coeff": S(_COEFF_POOL),
            "t": S(_TIME_POOL),
            "T": X(_TIME_POOL, ("t",)),
            "s": X(_TIME_POOL, ("t", "T")),
        },
    ),  # 8×7×9×11×7×6×5 = 1,481,760
    Template(
        name="kalman_filter_update",
        latex=(
            r"\hat{{{proc}}}_{{{t}|{t}}}"
            r" = \hat{{{proc}}}_{{{t}|{s}}}"
            r" + K_{{{t}}}\!\left(Y_{{{t}}}"
            r" - H\, \hat{{{proc}}}_{{{t}|{s}}}\right)"
        ),
        slots={
            "proc": S(_PROC_POOL),
            "t": S(_TIME_POOL),
            "s": X(_TIME_POOL, ("t",)),
        },
    ),  # 9×7×6 = 378
    Template(
        name="kalman_gain_matrix",
        latex=(
            r"K_{{{t}}} = P_{{{t}|{s}}} {H}^T"
            r"\!\left({H} P_{{{t}|{s}}} {H}^T + {R}\right)^{{-1}}"
        ),
        slots={
            "H": S(_FUNC_POOL),
            "R": X(_FUNC_POOL, ("H",)),
            "t": S(_TIME_POOL),
            "s": X(_TIME_POOL, ("t",)),
        },
    ),  # 8×7×7×6 = 2,352
    Template(
        name="riccati_equation_ode",
        latex=(
            r"\dot{{P}}_{{{t}}} = {A} P_{{{t}}} + P_{{{t}}} {A}^T"
            r" - P_{{{t}}} {H}^T {R}^{{-1}} {H} P_{{{t}}} + {Q}"
        ),
        slots={
            "A": S(_FUNC_POOL),
            "H": X(_FUNC_POOL, ("A",)),
            "R": X(_FUNC_POOL, ("A", "H")),
            "Q": X(_FUNC_POOL, ("A", "H", "R")),
            "t": S(_TIME_POOL),
        },
    ),  # 8×7×6×5×7 = 11,760
    Template(
        name="lqr_cost_functional",
        latex=(
            r"J({proc}) = \mathbb{{E}}\!\left["
            r"\int_0^{{{T}}} \left({proc}_{{{t}}}^T Q\, {proc}_{{{t}}}"
            r" + u_{{{t}}}^T R\, u_{{{t}}}\right) d{t}"
            r" + {proc}_{{{T}}}^T P\, {proc}_{{{T}}}"
            r"\right]"
        ),
        slots={
            "proc": S(_PROC_POOL),
            "t": S(_TIME_POOL),
            "T": X(_TIME_POOL, ("t",)),
        },
    ),  # 9×7×6 = 378
    Template(
        name="zakai_equation",
        latex=(
            r"d\sigma_{{{t}}}({func})"
            r" = \sigma_{{{t}}}(\mathcal{{L}}{func})\, d{t}"
            r" + \sigma_{{{t}}}({func2}\, {func})\, d{bm}_{{{t}}}"
        ),
        slots={
            "func": S(_FUNC_POOL),
            "func2": X(_FUNC_POOL, ("func",)),
            "bm": S(_BM_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 8×7×6×7 = 2,352
    Template(
        name="kushner_stratonovich_filter",
        latex=(
            r"d\pi_{{{t}}}({func})"
            r" = \pi_{{{t}}}(\mathcal{{L}}{func})\, d{t}"
            r" + \left(\pi_{{{t}}}({func2}\, {func})"
            r" - \pi_{{{t}}}({func2})\,\pi_{{{t}}}({func})\right)"
            r" \circ d{bm}_{{{t}}}"
        ),
        slots={
            "func": S(_FUNC_POOL),
            "func2": X(_FUNC_POOL, ("func",)),
            "bm": S(_BM_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 8×7×6×7 = 2,352
]

# ---------------------------------------------------------------------------
# Part C — High-n_eff function-pair templates (6 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_C: list[Template] = [
    Template(
        name="fn_ito_formula_composition",
        latex=(
            r"{fn1}\!\left(d{func}({bm}_{{{t}}})\right)"
            r" = {fn2}\!\left("
            r"{func}'({bm}_{{{t}}})\, d{bm}_{{{t}}}"
            r" + \tfrac{{1}}{{2}} {func}''({bm}_{{{t}}})\, d{t}"
            r"\right)"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "func": S(_FUNC_POOL),
            "bm": S(_BM_POOL),
            "t": S(_TIME_POOL),
        },
    ),  # 100²×8×6×7 = 33,600,000
    Template(
        name="fn_sde_solution_form",
        latex=(
            r"{fn1}({proc}_{{{t}}})"
            r" = {fn2}\!\left("
            r"{proc}_0"
            r" + \int_0^{{{t}}} {drift}({proc}_{{{s}}})\, d{s}"
            r" + \int_0^{{{t}}} {diff}({proc}_{{{s}}})\, d{bm}_{{{s}}}"
            r"\right)"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "proc": S(_PROC_POOL),
            "drift": S(_FUNC_POOL),
            "diff": X(_FUNC_POOL, ("drift",)),
            "bm": S(_BM_POOL),
            "t": S(_TIME_POOL),
            "s": X(_TIME_POOL, ("t",)),
        },
    ),  # 100²×9×8×7×6×7×6 ≈ 12.7B
    Template(
        name="fn_martingale_transform",
        latex=(
            r"{fn1}\!\left("
            r"\mathbb{{E}}\!\left[{proc}_{{{t2}}} \mid {filt}_{{{t1}}}\right]"
            r"\right)"
            r" = {fn2}({proc}_{{{t1}}})"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "proc": S(_PROC_POOL),
            "filt": S(_FILT_POOL),
            "t1": S(_TIME_POOL),
            "t2": X(_TIME_POOL, ("t1",)),
        },
    ),  # 100²×9×5×7×6 ≈ 1.89B
    Template(
        name="fn_fokker_planck_operator",
        latex=(
            r"{fn1}\!\left(\frac{{\partial {func}}}{{\partial {t}}}\right)"
            r" = {fn2}\!\left("
            r"-\frac{{\partial}}{{\partial x}}"
            r"\!\left({drift}(x)\, {func}\right)"
            r" + \tfrac{{1}}{{2}} \frac{{\partial^2}}{{\partial x^2}}"
            r"\!\left({diff}(x)^2\, {func}\right)\right)"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "func": S(_FUNC_POOL),
            "drift": X(_FUNC_POOL, ("func",)),
            "diff": X(_FUNC_POOL, ("func", "drift")),
            "t": S(_TIME_POOL),
        },
    ),  # 100²×8×7×6×7 ≈ 235B
    Template(
        name="fn_feynman_kac_composition",
        latex=(
            r"{fn1}({func}(x, {t}))"
            r" = {fn2}\!\left("
            r"\mathbb{{E}}^x\!\left["
            r"e^{{-{coeff}\, {T}}} {func2}({proc}_{{{T}}})"
            r"\right]\right)"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "func": S(_FUNC_POOL),
            "func2": X(_FUNC_POOL, ("func",)),
            "proc": S(_PROC_POOL),
            "coeff": S(_COEFF_POOL),
            "t": S(_TIME_POOL),
            "T": X(_TIME_POOL, ("t",)),
        },
    ),  # 100²×8×7×9×11×7×6 ≈ 2.33T
    Template(
        name="fn_girsanov_density",
        latex=(
            r"{fn1}\!\left(\frac{{d{meas2}}}{{d{meas1}}}\right)"
            r" = {fn2}\!\left(\exp\!\left("
            r"\int_0^{{{t}}} {coeff}_{{{s}}}\, d{bm}_{{{s}}}"
            r" - \tfrac{{1}}{{2}} \int_0^{{{t}}} {coeff}_{{{s}}}^2\, d{s}"
            r"\right)\right)"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "meas1": S(_MEAS_POOL),
            "meas2": X(_MEAS_POOL, ("meas1",)),
            "coeff": S(_COEFF_POOL),
            "bm": S(_BM_POOL),
            "t": S(_TIME_POOL),
            "s": X(_TIME_POOL, ("t",)),
        },
    ),  # 100²×4×3×11×6×7×6 ≈ 66.5B
]

# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

_TEMPLATES_D: list[Template] = [
    Template(
        name="communicating_states",
        latex=r"i \rightleftharpoons j \iff \exists\, n:\; P^n(i,j)>0 \;\wedge\; P^m(j,i)>0",
        slots={},
    ),
    Template(
        name="detailed_balance_reversibility",
        latex=r"\pi_i \, {proc}(i,j) \rightleftharpoons \pi_j \, {proc}(j,i) \;\forall\; i,j \in S",
        slots={"proc": S(_PROC_POOL)},
    ),
    Template(
        name="monotone_process_uparrow",
        latex=r"0 \leq {proc}_1 \leq {proc}_2 \leq \cdots \uparrow {proc}_\infty \text{{a.s.}}",
        slots={"proc": S(_PROC_POOL)},
    ),
]

_SPROC_TEMPLATES: list[Template] = (
    _TEMPLATES_A
    + _TEMPLATES_B1
    + _TEMPLATES_B2
    + _TEMPLATES_B3
    + _TEMPLATES_B4
    + _TEMPLATES_B5
    + _TEMPLATES_B6
    + _TEMPLATES_C
    + _TEMPLATES_D
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("stochastic_processes", _SPROC_TEMPLATES)
