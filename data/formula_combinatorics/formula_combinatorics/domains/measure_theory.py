"""Measure theory domain generators."""

from __future__ import annotations

from .._template_dsl import _FN_SLOT, _LIM_MOD, S, Template, X
from ._config import register_domain

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_MU_POOL = (r"\mu", r"\nu", r"\lambda", r"\rho", r"\sigma", r"\tau", r"\kappa", r"\pi")
_SPACE_POOL = ("X", "Y", "Z", r"\Omega", "E", "S")
_SET_POOL = ("A", "B", "C", "E", "F", "G", "H", "K")
_SIGALG_POOL = (
    r"\mathcal{F}",
    r"\mathcal{G}",
    r"\mathcal{A}",
    r"\mathcal{B}",
    r"\mathcal{E}",
    r"\mathcal{M}",
)
_FUNC_POOL = ("f", "g", "h", "F", "G", r"\phi", r"\psi", r"\varphi")
_VAR_POOL = ("x", "y", "t", "s", "u", "v", r"\omega")
_EXP_POOL = ("p", "q", "r", "2", "1", r"\infty")
_SCALAR_POOL = ("a", "b", "c", r"\alpha", r"\beta", r"\lambda", r"\epsilon", r"\delta")
_INT_POOL = ("n", "m", "k", "N", "M")

# ---------------------------------------------------------------------------
# Part A: reparameterized originals (14)
# ---------------------------------------------------------------------------

_TEMPLATES_A: list[Template] = [
    Template(
        name="sigma_additivity",
        latex=(
            r"{mu}(\emptyset) = 0, \quad {mu}\!\left(\bigsqcup_n {AA}_n\right)"
            r" = \sum_n {mu}({AA}_n)"
        ),
        slots={"mu": S(_MU_POOL), "AA": S(_SET_POOL)},
    ),
    Template(
        name="nonnegativity_integral",
        latex=r"\int {ff} \, d{mu}({xx}) \geq 0 \text{{ for }} {ff} \geq 0",
        slots={"ff": S(_FUNC_POOL), "mu": S(_MU_POOL), "xx": S(_VAR_POOL)},
    ),
    Template(
        name="radon_nikodym",
        latex=(
            r"\frac{{d{mu}}}{{d{nu}}} \geq 0, \quad {mu}({AA}) = "
            r"\int{lim_mod}_{{{AA}}} \frac{{d{mu}}}{{d{nu}}} \, d{nu}({xx})"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "mu": S(_MU_POOL),
            "nu": X(_MU_POOL, ("mu",)),
            "AA": S(_SET_POOL),
            "xx": S(_VAR_POOL),
        },
    ),
    Template(
        name="l1_convergence",
        latex=r"\int |{ff}_{{{nn}}} - {ff}| \, d{mu} \to 0",
        slots={"ff": S(_FUNC_POOL), "nn": S(_INT_POOL), "mu": S(_MU_POOL)},
    ),
    Template(
        name="fatou_lemma",
        latex=(r"\int \liminf_n {ff}_n \, d{mu} \leq \liminf_n \int {ff}_n \, d{mu}"),
        slots={"ff": S(_FUNC_POOL), "mu": S(_MU_POOL)},
    ),
    Template(
        name="fubini_tonelli",
        latex=(
            r"\int{lim_mod}_{{X \times Y}} {ff} \, d({mu} \otimes {nu}) = "
            r"\int_X \int_Y {ff}({xx},{yy}) \, d{nu}({yy}) \, d{mu}({xx})"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "ff": S(_FUNC_POOL),
            "mu": S(_MU_POOL),
            "nu": X(_MU_POOL, ("mu",)),
            "xx": S(_VAR_POOL),
            "yy": X(_VAR_POOL, ("xx",)),
        },
    ),
    Template(
        name="lp_norm",
        latex=(
            r"\|{ff}\|_{{L^{{{pp}}}({mu})}} = "
            r"\left(\int |{ff}|^{{{pp}}} \, d{mu}\right)^{{1/{pp}}}"
        ),
        slots={"ff": S(_FUNC_POOL), "pp": S(_EXP_POOL), "mu": S(_MU_POOL)},
    ),
    Template(
        name="absolute_continuity",
        latex=(
            r"{mu} \ll {nu} \iff {mu}({AA}) = 0 "
            r"\text{{ whenever }} {nu}({AA}) = 0"
        ),
        slots={
            "mu": S(_MU_POOL),
            "nu": X(_MU_POOL, ("mu",)),
            "AA": S(_SET_POOL),
        },
    ),
    Template(
        name="lebesgue_decomposition",
        latex=r"{mu} = {mu}_{{{nu}\text{{-ac}}}} + {mu}_{{{nu}\text{{-sing}}}}",
        slots={"mu": S(_MU_POOL), "nu": X(_MU_POOL, ("mu",))},
    ),
    Template(
        name="dominated_convergence",
        latex=(
            r"|{ff}_n| \leq {gg},\; \int {gg} \, d{mu} < \infty "
            r"\implies \int {ff}_n \, d{mu} \to \int {ff} \, d{mu}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "mu": S(_MU_POOL),
        },
    ),
    Template(
        name="sigma_algebra_generated",
        latex=(
            r"\sigma({CC}) = \bigcap\left\{{{FF} : {CC} \subseteq {FF},\;"
            r"{FF} \text{{ is a }} \sigma\text{{-algebra}}\right\}}"
        ),
        slots={"CC": S(_SIGALG_POOL), "FF": X(_SIGALG_POOL, ("CC",))},
    ),
    Template(
        name="markov_inequality",
        latex=(
            r"{mu}(\{{|{ff}| \geq {tt}\}}) \leq "
            r"\frac{{1}}{{{tt}}} \int |{ff}| \, d{mu}"
        ),
        slots={"ff": S(_FUNC_POOL), "tt": S(_SCALAR_POOL), "mu": S(_MU_POOL)},
    ),
    Template(
        name="jensens_inequality",
        latex=(
            r"{phi}\!\left(\int {ff} \, d{mu}\right) \leq "
            r"\int {phi}({ff}) \, d{mu}"
        ),
        slots={
            "phi": S(_FUNC_POOL),
            "ff": X(_FUNC_POOL, ("phi",)),
            "mu": S(_MU_POOL),
        },
    ),
    Template(
        name="product_measure",
        latex=(r"({mu} \otimes {nu})({AA} \times {BB}) = {mu}({AA}) \cdot {nu}({BB})"),
        slots={
            "mu": S(_MU_POOL),
            "nu": X(_MU_POOL, ("mu",)),
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B1: σ-algebras & measurability (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B1: list[Template] = [
    Template(
        name="borel_sigma_algebra",
        latex=r"\mathcal{{B}}({XX}) = \sigma(\text{{open sets in }} {XX})",
        slots={"XX": S(_SPACE_POOL)},
    ),
    Template(
        name="measurable_preimage",
        latex=(r"{ff}^{{-1}}({AA}) \in {FF} \text{{ for every }} {AA} \in {GG}"),
        slots={
            "ff": S(_FUNC_POOL),
            "AA": S(_SET_POOL),
            "FF": S(_SIGALG_POOL),
            "GG": X(_SIGALG_POOL, ("FF",)),
        },
    ),
    Template(
        name="sigma_algebra_intersection",
        latex=r"{FF} \cap {GG} \text{{ is a }} \sigma\text{{-algebra}}",
        slots={"FF": S(_SIGALG_POOL), "GG": X(_SIGALG_POOL, ("FF",))},
    ),
    Template(
        name="measurable_composition",
        latex=(
            r"{gg} \circ {ff} : ({XX}, {FF}) \to ({YY}, {HH}) "
            r"\text{{ is measurable}}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "XX": S(_SPACE_POOL),
            "YY": X(_SPACE_POOL, ("XX",)),
            "FF": S(_SIGALG_POOL),
            "HH": X(_SIGALG_POOL, ("FF",)),
        },
    ),
    Template(
        name="indicator_function",
        latex=(
            r"\mathbf{{1}}_{{{AA}}}({xx}) = "
            r"\begin{{cases}} 1 & {xx} \in {AA} \\ 0 & {xx} \notin {AA} \end{{cases}}"
        ),
        slots={"AA": S(_SET_POOL), "xx": S(_VAR_POOL)},
    ),
    Template(
        name="simple_function",
        latex=(
            r"{ff} = \sum{lim_mod}_{{k=1}}^{{{nn}}} {aa}_k \, "
            r"\mathbf{{1}}_{{{AA}_k}}"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "ff": S(_FUNC_POOL),
            "nn": S(_INT_POOL),
            "aa": S(_SCALAR_POOL),
            "AA": S(_SET_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B2: outer measures & Carathéodory (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B2: list[Template] = [
    Template(
        name="outer_measure_subadditivity",
        latex=(r"{mu}^*\!\left(\bigcup_n {AA}_n\right) \leq \sum_n {mu}^*({AA}_n)"),
        slots={"mu": S(_MU_POOL), "AA": S(_SET_POOL)},
    ),
    Template(
        name="caratheodory_condition",
        latex=(
            r"{mu}^*({EE}) = {mu}^*({EE} \cap {AA}) + {mu}^*({EE} \cap {AA}^c)"
            r" \quad \forall\, {EE}"
        ),
        slots={
            "mu": S(_MU_POOL),
            "AA": S(_SET_POOL),
            "EE": X(_SET_POOL, ("AA",)),
        },
    ),
    Template(
        name="premeasure_extension",
        latex=(
            r"{mu} : \mathcal{{A}} \to [0, \infty] \text{{ premeasure}}"
            r" \Rightarrow \exists\, \bar{{{mu}}} \text{{ on }} \sigma(\mathcal{{A}})"
        ),
        slots={"mu": S(_MU_POOL)},
    ),
    Template(
        name="regularity_inner",
        latex=(
            r"{mu}({AA}) = \sup\{{{mu}({KK}) : {KK} \subseteq {AA},"
            r" {KK} \text{{ compact}}\}}"
        ),
        slots={
            "mu": S(_MU_POOL),
            "AA": S(_SET_POOL),
            "KK": X(_SET_POOL, ("AA",)),
        },
    ),
    Template(
        name="regularity_outer",
        latex=(
            r"{mu}({AA}) = \inf\{{{mu}({UU}) : {AA} \subseteq {UU},"
            r" {UU} \text{{ open}}\}}"
        ),
        slots={
            "mu": S(_MU_POOL),
            "AA": S(_SET_POOL),
            "UU": X(_SET_POOL, ("AA",)),
        },
    ),
    Template(
        name="covering_measure",
        latex=(
            r"{mu}({AA}) = \inf\!\left\{{\sum_k \ell(I_k) :"
            r" {AA} \subseteq \bigcup_k I_k\right\}}"
        ),
        slots={"mu": S(_MU_POOL), "AA": S(_SET_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B3: convergence theorems (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B3: list[Template] = [
    Template(
        name="monotone_convergence",
        latex=(
            r"0 \leq {ff}_n \nearrow {ff} \Rightarrow "
            r"\int {ff}_n \, d{mu} \nearrow \int {ff} \, d{mu}"
        ),
        slots={"ff": S(_FUNC_POOL), "mu": S(_MU_POOL)},
    ),
    Template(
        name="vitali_convergence",
        latex=(
            r"{ff}_n \to {ff} \text{{ in }} L^{{{pp}}} \iff "
            r"\text{{u.i. and }} {ff}_n \xrightarrow{{{mu}}} {ff}"
        ),
        slots={"ff": S(_FUNC_POOL), "pp": S(_EXP_POOL), "mu": S(_MU_POOL)},
    ),
    Template(
        name="egorov_theorem",
        latex=(
            r"{ff}_n \to {ff} \text{{ a.e.}} \Rightarrow \forall \epsilon > 0,"
            r" \exists\, {AA} : {mu}({AA}^c) < \epsilon,"
            r" {ff}_n \rightrightarrows {ff} \text{{ on }} {AA}"
        ),
        slots={"ff": S(_FUNC_POOL), "mu": S(_MU_POOL), "AA": S(_SET_POOL)},
    ),
    Template(
        name="lusin_theorem",
        latex=(
            r"{ff} \text{{ measurable}} \Rightarrow \forall \epsilon > 0,"
            r" \exists\, {KK} \text{{ closed}}, {mu}({KK}^c) < \epsilon,"
            r" {ff}|_{{{KK}}} \text{{ continuous}}"
        ),
        slots={"ff": S(_FUNC_POOL), "mu": S(_MU_POOL), "KK": S(_SET_POOL)},
    ),
    Template(
        name="convergence_in_measure",
        latex=(
            r"{mu}(\{{|{ff}_n - {ff}| > \epsilon\}}) \to 0 "
            r"\quad \forall\, \epsilon > 0"
        ),
        slots={"ff": S(_FUNC_POOL), "mu": S(_MU_POOL)},
    ),
    Template(
        name="cauchy_in_measure",
        latex=(
            r"{ff}_n \to {gg} \text{{ and }} {ff}_n \to {hh} \text{{ in measure}}"
            r" \Rightarrow {gg} = {hh} \text{{ a.e.}}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "hh": X(_FUNC_POOL, ("ff", "gg")),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B4: Lp spaces (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B4: list[Template] = [
    Template(
        name="holder_inequality",
        latex=(
            r"\int |{ff} {gg}| \, d{mu} \leq \|{ff}\|_{{{pp}}} \|{gg}\|_{{{qq}}},"
            r" \quad \tfrac{{1}}{{{pp}}} + \tfrac{{1}}{{{qq}}} = 1"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "pp": S(_EXP_POOL),
            "qq": X(_EXP_POOL, ("pp",)),
            "mu": S(_MU_POOL),
        },
    ),
    Template(
        name="minkowski_inequality",
        latex=(r"\|{ff} + {gg}\|_{{{pp}}} \leq \|{ff}\|_{{{pp}}} + \|{gg}\|_{{{pp}}}"),
        slots={
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "pp": S(_EXP_POOL),
            "mu": S(_MU_POOL),
        },
    ),
    Template(
        name="lp_completeness",
        latex=r"L^{{{pp}}}({XX}, {mu}) \text{{ is a Banach space}}",
        slots={"pp": S(_EXP_POOL), "XX": S(_SPACE_POOL), "mu": S(_MU_POOL)},
    ),
    Template(
        name="l2_inner_product",
        latex=(
            r"\langle {ff}, {gg} \rangle_{{L^2}} = "
            r"\int {ff} \, \overline{{{gg}}} \, d{mu}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "mu": S(_MU_POOL),
        },
    ),
    Template(
        name="lp_inclusion",
        latex=(
            r"{mu}({XX}) < \infty \Rightarrow "
            r"L^{{{pp}}}({XX}) \subseteq L^{{{qq}}}({XX}) \text{{ for }} {pp} > {qq}"
        ),
        slots={
            "pp": S(_EXP_POOL),
            "qq": X(_EXP_POOL, ("pp",)),
            "XX": S(_SPACE_POOL),
            "mu": S(_MU_POOL),
        },
    ),
    Template(
        name="lp_dual",
        latex=(
            r"\bigl(L^{{{pp}}}({XX}, {mu})\bigr)^* \cong L^{{{qq}}},"
            r" \quad \tfrac{{1}}{{{pp}}} + \tfrac{{1}}{{{qq}}} = 1"
        ),
        slots={
            "pp": S(_EXP_POOL),
            "qq": X(_EXP_POOL, ("pp",)),
            "XX": S(_SPACE_POOL),
            "mu": S(_MU_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B5: differentiation of measures (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B5: list[Template] = [
    Template(
        name="lebesgue_diff_theorem",
        latex=(
            r"\lim_{{r \to 0}} \frac{{1}}{{{mu}(B_r({xx}))}}"
            r"\int{lim_mod}_{{B_r({xx})}} {ff} \, d{mu} = {ff}({xx}) \text{{ a.e.}}"
        ),
        slots={"lim_mod": _LIM_MOD, "ff": S(_FUNC_POOL), "mu": S(_MU_POOL), "xx": S(_VAR_POOL)},
    ),
    Template(
        name="hardy_littlewood",
        latex=(
            r"{mu}(\{{M{ff} > {aa}\}}) \leq "
            r"\frac{{C}}{{{aa}}} \|{ff}\|_{{L^1}}"
        ),
        slots={"ff": S(_FUNC_POOL), "mu": S(_MU_POOL), "aa": S(_SCALAR_POOL)},
    ),
    Template(
        name="rn_chain_rule",
        latex=(
            r"\frac{{d{mu}}}{{d{rho}}} = "
            r"\frac{{d{mu}}}{{d{nu}}} \cdot \frac{{d{nu}}}{{d{rho}}}"
        ),
        slots={
            "mu": S(_MU_POOL),
            "nu": X(_MU_POOL, ("mu",)),
            "rho": X(_MU_POOL, ("mu", "nu")),
        },
    ),
    Template(
        name="total_variation",
        latex=(r"\|{mu}\| = |{mu}|({XX}) = \sup \sum_k |{mu}({AA}_k)|"),
        slots={"mu": S(_MU_POOL), "XX": S(_SPACE_POOL), "AA": S(_SET_POOL)},
    ),
    Template(
        name="hahn_decomposition",
        latex=(
            r"{XX} = {PP} \cup {NN},\; {PP} \cap {NN} = \emptyset,\;"
            r"{nu} \geq 0 \text{{ on }} {PP},\; {nu} \leq 0 \text{{ on }} {NN}"
        ),
        slots={
            "XX": S(_SPACE_POOL),
            "nu": S(_MU_POOL),
            "PP": S(_SET_POOL),
            "NN": X(_SET_POOL, ("PP",)),
        },
    ),
    Template(
        name="jordan_decomposition",
        latex=r"{nu} = {nu}^+ - {nu}^-, \quad {nu}^+ \perp {nu}^-",
        slots={"nu": S(_MU_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B6: product measures & pushforward (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B6: list[Template] = [
    Template(
        name="three_fold_product",
        latex=(
            r"({mu} \otimes {nu} \otimes {rho})"
            r"({AA} \times {BB} \times {CC})"
            r" = {mu}({AA})\, {nu}({BB})\, {rho}({CC})"
        ),
        slots={
            "mu": S(_MU_POOL),
            "nu": X(_MU_POOL, ("mu",)),
            "rho": X(_MU_POOL, ("mu", "nu")),
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
            "CC": X(_SET_POOL, ("AA", "BB")),
        },
    ),
    Template(
        name="change_of_variables",
        latex=(
            r"\int{lim_mod}_{{f({AA})}} {gg}({yy}) \, d{nu}({yy}) = "
            r"\int{lim_mod}_{{{AA}}} {gg}({ff}({xx})) |\det D{ff}({xx})| \, d{mu}({xx})"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "mu": S(_MU_POOL),
            "nu": X(_MU_POOL, ("mu",)),
            "AA": S(_SET_POOL),
            "xx": S(_VAR_POOL),
            "yy": X(_VAR_POOL, ("xx",)),
        },
    ),
    Template(
        name="disintegration",
        latex=(
            r"{mu} = \int{lim_mod}_{{{YY}}} {mu}^{{{yy}}} \, d{nu}({yy})"
            r" \quad \text{{(disintegration over }} {nu}\text{{)}}"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "mu": S(_MU_POOL),
            "nu": X(_MU_POOL, ("mu",)),
            "YY": S(_SPACE_POOL),
            "yy": S(_VAR_POOL),
        },
    ),
    Template(
        name="pushforward_measure",
        latex=(
            r"{nu} = {ff}_* {mu}, \quad "
            r"{nu}({BB}) = {mu}({ff}^{{-1}}({BB}))"
        ),
        slots={
            "mu": S(_MU_POOL),
            "nu": X(_MU_POOL, ("mu",)),
            "ff": S(_FUNC_POOL),
            "BB": S(_SET_POOL),
        },
    ),
    Template(
        name="image_measure_integral",
        latex=(r"\int {gg} \, d({ff}_* {mu}) = \int {gg} \circ {ff} \, d{mu}"),
        slots={
            "gg": S(_FUNC_POOL),
            "ff": X(_FUNC_POOL, ("gg",)),
            "mu": S(_MU_POOL),
        },
    ),
    Template(
        name="singular_measures",
        latex=(
            r"{mu} \perp {nu} : \exists\, {AA}, {BB} \text{{ disjoint}},"
            r" {mu}({BB}) = 0,\; {nu}({AA}) = 0"
        ),
        slots={
            "mu": S(_MU_POOL),
            "nu": X(_MU_POOL, ("mu",)),
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B7: probability measures (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B7: list[Template] = [
    Template(
        name="probability_axioms",
        latex=(
            r"\mathbb{{P}}(\Omega) = 1, \quad "
            r"\mathbb{{P}}\!\left(\bigsqcup_n {AA}_n\right) = "
            r"\sum_n \mathbb{{P}}({AA}_n)"
        ),
        slots={"AA": S(_SET_POOL)},
    ),
    Template(
        name="conditional_probability",
        latex=(
            r"\mathbb{{P}}({AA} \mid {BB}) = "
            r"\frac{{\mathbb{{P}}({AA} \cap {BB})}}{{\mathbb{{P}}({BB})}}"
        ),
        slots={"AA": S(_SET_POOL), "BB": X(_SET_POOL, ("AA",))},
    ),
    Template(
        name="independence",
        latex=(
            r"\mathbb{{P}}({AA} \cap {BB}) = "
            r"\mathbb{{P}}({AA}) \cdot \mathbb{{P}}({BB})"
            r" \quad ({AA}, {BB} \text{{ independent}})"
        ),
        slots={"AA": S(_SET_POOL), "BB": X(_SET_POOL, ("AA",))},
    ),
    Template(
        name="expectation_def",
        latex=r"\mathbb{{E}}[{ff}] = \int_\Omega {ff} \, d\mathbb{{P}}",
        slots={"ff": S(_FUNC_POOL)},
    ),
    Template(
        name="variance_def",
        latex=(
            r"\operatorname{{Var}}({ff}) = "
            r"\mathbb{{E}}[({ff} - \mathbb{{E}}[{ff}])^2] = "
            r"\mathbb{{E}}[{ff}^2] - (\mathbb{{E}}[{ff}])^2"
        ),
        slots={"ff": S(_FUNC_POOL)},
    ),
    Template(
        name="chebyshev_inequality",
        latex=(
            r"\mathbb{{P}}(|{ff} - \mathbb{{E}}[{ff}]| \geq {tt}) \leq "
            r"\frac{{\operatorname{{Var}}({ff})}}{{{tt}^2}}"
        ),
        slots={"ff": S(_FUNC_POOL), "tt": S(_SCALAR_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part C: high-n_eff function-pair templates (6)
# ---------------------------------------------------------------------------

_TEMPLATES_C: list[Template] = [
    Template(
        name="integrand_product_fn",
        latex=r"\int{lim_mod}_{{{AA}}} {fn1}({xx}) {fn2}({xx}) \, d{mu}({xx})",
        slots={
            "lim_mod": _LIM_MOD,
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "AA": S(_SET_POOL),
            "xx": S(_VAR_POOL),
            "mu": S(_MU_POOL),
        },
    ),
    Template(
        name="lp_fn_sum",
        latex=r"\|{fn1} + {fn2}\|_{{L^{{{pp}}}({mu})}}",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "pp": S(_EXP_POOL),
            "mu": S(_MU_POOL),
        },
    ),
    Template(
        name="radon_nikodym_fn",
        latex=(
            r"{mu}({AA}) = \int{lim_mod}_{{{AA}}} {fn1}({xx}) \, d{nu}({xx})"
            r" \quad ({fn1} = \tfrac{{d{mu}}}{{d{nu}}})"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "fn1": _FN_SLOT,
            "AA": S(_SET_POOL),
            "xx": S(_VAR_POOL),
            "mu": S(_MU_POOL),
            "nu": X(_MU_POOL, ("mu",)),
        },
    ),
    Template(
        name="conditional_expectation_fn",
        latex=(r"\mathbb{{E}}[{fn1} \mid {FF}] \text{{ is }} {FF}\text{{-measurable}}"),
        slots={
            "fn1": _FN_SLOT,
            "FF": S(_SIGALG_POOL),
        },
    ),
    Template(
        name="fn_pair_measure_integral",
        latex=(
            r"\int {fn1}({xx}) \, d({fn2}_* {mu})({xx}) = "
            r"\int {fn1}({fn2}({xx})) \, d{mu}({xx})"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "xx": S(_VAR_POOL),
            "mu": S(_MU_POOL),
        },
    ),
    Template(
        name="fn_triple_integral",
        latex=(r"\int {fn1}({fn2}({xx})) \, {fn3}({xx}) \, d{mu}({xx})"),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "fn3": _FN_SLOT,
            "xx": S(_VAR_POOL),
            "mu": S(_MU_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

_MEASURE_TEMPLATES: list[Template] = (
    _TEMPLATES_A
    + _TEMPLATES_B1
    + _TEMPLATES_B2
    + _TEMPLATES_B3
    + _TEMPLATES_B4
    + _TEMPLATES_B5
    + _TEMPLATES_B6
    + _TEMPLATES_B7
    + _TEMPLATES_C
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("measure_theory", _MEASURE_TEMPLATES)
