"""Information theory domain generators."""

from __future__ import annotations

from ..engine._template_dsl import _FN_SLOT, _LIM_MOD, S, Template, X
from ..engine._vocab import _GENERIC_IDX as _IDX_POOL
from ._config import register_domain

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_RV_POOL = ("X", "Y", "Z", "U", "V", "W", "S", "T", "A", "B", "N", "M")  # 12
_DIST_POOL = ("P", "Q", "R", r"\mu", r"\nu", "S", "T", "U", r"\pi", r"\rho")  # 10
_LOG_POOL = (r"\log_2", r"\ln", r"\log", r"\log_{10}", r"\log_e", r"\log_q", r"\mathrm{ld}", r"\log_p")  # 8
_ALPHA_POOL = (
    r"\alpha",
    r"\beta",
    r"2",
    r"\frac{1}{2}",
    r"\frac{3}{2}",
    r"\gamma",
    r"\delta",
    r"\kappa",
    r"\lambda",
    r"\tau",
)  # 10
_EPS_POOL = (r"\epsilon", r"\delta", r"\varepsilon", r"\eta", r"\gamma", r"\kappa", r"\nu", r"\zeta")  # 8
_RATE_POOL = ("R", r"R_0", r"R_1", r"\mathcal{C}", "C")  # 5
_SET_POOL = (
    r"\mathcal{X}",
    r"\mathcal{Y}",
    r"\mathcal{Z}",
    r"\mathcal{A}",
    r"\mathcal{U}",
)  # 5
_PARAM_POOL = (r"\theta", r"\phi", r"\lambda", r"\mu", r"\sigma")  # 5
_NOISE_POOL = (r"\sigma^2", r"N_0", r"N_0/2", r"\sigma_n^2")  # 4
_SNR_POOL = (r"\mathrm{SNR}", r"\gamma", r"\rho", r"P/N_0")  # 4
_BW_POOL = ("W", "B", r"\Delta f", r"B_n")  # 4
_PWR_POOL = ("P", r"P_s", r"P_T", r"\mathcal{P}")  # 4
# Restricted pools for templates that append their own subscript (_k etc.) —
# entries already containing _ would produce invalid double-subscript LaTeX.
_PWR_BASE_POOL = tuple(v for v in _PWR_POOL if "_" not in v)
_NOISE_BASE_POOL = tuple(v for v in _NOISE_POOL if "_" not in v)

# ---------------------------------------------------------------------------
# Part A — reparameterized originals (14)
# ---------------------------------------------------------------------------

_TEMPLATES_A: list[Template] = [
    Template(
        name="entropy",
        latex=r"H({XX}) = -\sum{lim_mod}_{{x \in {Sx}}} p(x)\, {lb}\, p(x)",
        slots={"lim_mod": _LIM_MOD, "XX": S(_RV_POOL), "Sx": S(_SET_POOL), "lb": S(_LOG_POOL)},
    ),
    Template(
        name="joint_entropy",
        latex=r"H({XX}, {YY}) = -\sum{lim_mod}_{{x,y}} p(x,y)\, {lb}\, p(x,y)",
        slots={"lim_mod": _LIM_MOD, "XX": S(_RV_POOL), "YY": X(_RV_POOL, ("XX",)), "lb": S(_LOG_POOL)},
    ),
    Template(
        name="conditional_entropy",
        latex=r"H({YY} \mid {XX}) = H({XX}, {YY}) - H({XX})",
        slots={"XX": S(_RV_POOL), "YY": X(_RV_POOL, ("XX",))},
    ),
    Template(
        name="mutual_information",
        latex=r"I({XX}; {YY}) = H({XX}) + H({YY}) - H({XX}, {YY})",
        slots={"XX": S(_RV_POOL), "YY": X(_RV_POOL, ("XX",))},
    ),
    Template(
        name="kl_divergence",
        latex=r"D_{{KL}}({pp} \| {qq}) = \sum_x {pp}(x)\, {lb}\, \frac{{{pp}(x)}}{{{qq}(x)}}",
        slots={
            "pp": S(_DIST_POOL),
            "qq": X(_DIST_POOL, ("pp",)),
            "lb": S(_LOG_POOL),
        },
    ),
    Template(
        name="channel_capacity",
        latex=r"C = \max_{{p({XX})}} I({XX}; {YY})",
        slots={"XX": S(_RV_POOL), "YY": X(_RV_POOL, ("XX",))},
    ),
    Template(
        name="entropy_bound",
        latex=r"H({XX}) \leq {lb}\, |{Sx}|",
        slots={"XX": S(_RV_POOL), "Sx": S(_SET_POOL), "lb": S(_LOG_POOL)},
    ),
    Template(
        name="entropy_subadditivity",
        latex=r"H({XX}_1, \ldots, {XX}_{{{nn}}}) \leq \sum{lim_mod}_{{i=1}}^{{{nn}}} H({XX}_i)",
        slots={"lim_mod": _LIM_MOD, "XX": S(_RV_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="mutual_info_nonneg",
        latex=r"I({XX}; {YY}) \geq 0",
        slots={"XX": S(_RV_POOL), "YY": X(_RV_POOL, ("XX",))},
    ),
    Template(
        name="cross_entropy",
        latex=r"H({pp}, {qq}) = -\sum_x {pp}(x)\, {lb}\, {qq}(x)",
        slots={
            "pp": S(_DIST_POOL),
            "qq": X(_DIST_POOL, ("pp",)),
            "lb": S(_LOG_POOL),
        },
    ),
    Template(
        name="rate_distortion",
        latex=(
            r"R(D) = \min_{{{pp}(\hat{{{XX}}}|{XX})\,:\,"
            r"\mathbb{{E}}[d({XX},\hat{{{XX}}})] \leq D}}"
            r" I({XX}; \hat{{{XX}}})"
        ),
        slots={"XX": S(_RV_POOL), "pp": S(_DIST_POOL)},
    ),
    Template(
        name="fano_inequality",
        latex=(
            r"H({ee}) + {ee}\, {lb}(|{Sx}| - 1)"
            r" \geq H({XX} \mid \hat{{{XX}}})"
        ),
        slots={
            "XX": S(_RV_POOL),
            "Sx": S(_SET_POOL),
            "lb": S(_LOG_POOL),
            "ee": S(_EPS_POOL),
        },
    ),
    Template(
        name="data_processing_inequality",
        latex=r"{XX} \to {YY} \to {ZZ} \implies I({XX}; {ZZ}) \leq I({XX}; {YY})",
        slots={
            "XX": S(_RV_POOL),
            "YY": X(_RV_POOL, ("XX",)),
            "ZZ": X(_RV_POOL, ("XX", "YY")),
        },
    ),
    Template(
        name="chain_rule_entropy",
        latex=(
            r"H({XX}_1, \ldots, {XX}_{{{nn}}})"
            r" = \sum{lim_mod}_{{i=1}}^{{{nn}}} H({XX}_i \mid {XX}_1, \ldots, {XX}_{{i-1}})"
        ),
        slots={"lim_mod": _LIM_MOD, "XX": S(_RV_POOL), "nn": S(_IDX_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B — new flat templates by subfield (~50)
# ---------------------------------------------------------------------------

# Entropy variants (8)
_TEMPLATES_B_ENTROPY: list[Template] = [
    Template(
        name="renyi_entropy",
        latex=(
            r"H_{{{aa}}}({XX}) = \frac{{1}}{{1 - {aa}}}"
            r"\, {lb}\!\left(\sum{lim_mod}_{{x \in {Sx}}} p(x)^{{{aa}}}\right)"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "XX": S(_RV_POOL),
            "aa": S(_ALPHA_POOL),
            "Sx": S(_SET_POOL),
            "lb": S(_LOG_POOL),
        },
    ),
    Template(
        name="tsallis_entropy",
        latex=(
            r"S_{{{aa}}}({XX}) = \frac{{1}}{{{aa} - 1}}"
            r"\left(1 - \sum{lim_mod}_{{x}} p(x)^{{{aa}}}\right)"
        ),
        slots={"lim_mod": _LIM_MOD, "XX": S(_RV_POOL), "aa": S(_ALPHA_POOL)},
    ),
    Template(
        name="differential_entropy",
        latex=r"h({XX}) = -\int{lim_mod}_{{-\infty}}^{{\infty}} p(x)\, {lb}\, p(x)\, dx",
        slots={"lim_mod": _LIM_MOD, "XX": S(_RV_POOL), "lb": S(_LOG_POOL)},
    ),
    Template(
        name="conditional_differential_entropy",
        latex=r"h({YY} \mid {XX}) = -\iint p(x,y)\, {lb}\, p(y \mid x)\, dx\, dy",
        slots={
            "XX": S(_RV_POOL),
            "YY": X(_RV_POOL, ("XX",)),
            "lb": S(_LOG_POOL),
        },
    ),
    Template(
        name="entropy_gaussian",
        latex=(
            r"h\!\left(\mathcal{{N}}({mm}, {sv})\right)"
            r" = \tfrac{{1}}{{2}}\, {lb}(2\pi e\, {sv})"
        ),
        slots={"mm": S(_PARAM_POOL), "sv": S(_NOISE_POOL), "lb": S(_LOG_POOL)},
    ),
    Template(
        name="entropy_uniform",
        latex=r"H\!\left(\mathrm{{Uniform}}({Sx})\right) = {lb}\, |{Sx}|",
        slots={"Sx": S(_SET_POOL), "lb": S(_LOG_POOL)},
    ),
    Template(
        name="entropy_power_inequality",
        latex=(
            r"N\!\left(h({XX} + {YY})\right)"
            r" \geq N\!\left(h({XX})\right) + N\!\left(h({YY})\right)"
        ),
        slots={"XX": S(_RV_POOL), "YY": X(_RV_POOL, ("XX",))},
    ),
    Template(
        name="entropy_rate",
        latex=(
            r"\bar{{H}}({XX}) = \lim_{{{nn}\to\infty}}"
            r" \frac{{1}}{{{nn}}} H({XX}_1, \ldots, {XX}_{{{nn}}})"
        ),
        slots={"XX": S(_RV_POOL), "nn": S(_IDX_POOL)},
    ),
]

# Mutual information variants (6)
_TEMPLATES_B_MI: list[Template] = [
    Template(
        name="mi_kl_form",
        latex=(
            r"I({XX}; {YY}) = D_{{KL}}\!\left("
            r"p_{{{XX}{YY}}} \| p_{{{XX}}} \otimes p_{{{YY}}}\right)"
        ),
        slots={"XX": S(_RV_POOL), "YY": X(_RV_POOL, ("XX",))},
    ),
    Template(
        name="conditional_mi",
        latex=(
            r"I({XX}; {YY} \mid {ZZ})"
            r" = H({XX} \mid {ZZ}) - H({XX} \mid {YY}, {ZZ})"
        ),
        slots={
            "XX": S(_RV_POOL),
            "YY": X(_RV_POOL, ("XX",)),
            "ZZ": X(_RV_POOL, ("XX", "YY")),
        },
    ),
    Template(
        name="mi_chain_rule",
        latex=(
            r"I({XX}; {YY}_1, \ldots, {YY}_{{{nn}}})"
            r" = \sum{lim_mod}_{{i=1}}^{{{nn}}} I({XX}; {YY}_i \mid {YY}_1, \ldots, {YY}_{{i-1}})"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "XX": S(_RV_POOL),
            "YY": X(_RV_POOL, ("XX",)),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="interaction_information",
        latex=(
            r"I({XX}; {YY}; {ZZ})"
            r" = I({XX}; {YY}) - I({XX}; {YY} \mid {ZZ})"
        ),
        slots={
            "XX": S(_RV_POOL),
            "YY": X(_RV_POOL, ("XX",)),
            "ZZ": X(_RV_POOL, ("XX", "YY")),
        },
    ),
    Template(
        name="multivariate_mi",
        latex=(
            r"I({XX}_1;\, \ldots;\, {XX}_{{{nn}}})"
            r" = \sum{lim_mod}_{{S \subseteq [{nn}]}} (-1)^{{|S|+1}}"
            r" H\!\left(\{{{{XX}}_i\}}_{{i \in S}}\right)"
        ),
        slots={"lim_mod": _LIM_MOD, "XX": S(_RV_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="mi_integral_form",
        latex=(
            r"I({XX}; {YY}) = \int\!\int p(x, y)\, {lb}\,"
            r" \frac{{p(x,y)}}{{p(x)\, p(y)}}\, dx\, dy"
        ),
        slots={"XX": S(_RV_POOL), "YY": X(_RV_POOL, ("XX",)), "lb": S(_LOG_POOL)},
    ),
]

# Divergences (8)
_TEMPLATES_B_DIV: list[Template] = [
    Template(
        name="kl_continuous",
        latex=(
            r"D_{{KL}}({pp} \| {qq})"
            r" = \int p(x)\, {lb}\, \frac{{{pp}(x)}}{{{qq}(x)}}\, dx"
        ),
        slots={
            "pp": S(_DIST_POOL),
            "qq": X(_DIST_POOL, ("pp",)),
            "lb": S(_LOG_POOL),
        },
    ),
    Template(
        name="js_divergence",
        latex=(
            r"D_{{JS}}({pp} \| {qq})"
            r" = \tfrac{{1}}{{2}} D_{{KL}}\!\left({pp} \,\Big\|\, \tfrac{{{pp}+{qq}}}{{2}}\right)"
            r" + \tfrac{{1}}{{2}} D_{{KL}}\!\left({qq} \,\Big\|\, \tfrac{{{pp}+{qq}}}{{2}}\right)"
        ),
        slots={"pp": S(_DIST_POOL), "qq": X(_DIST_POOL, ("pp",))},
    ),
    Template(
        name="total_variation",
        latex=(
            r"\mathrm{{TV}}({pp}, {qq})"
            r" = \tfrac{{1}}{{2}} \sum_x |{pp}(x) - {qq}(x)|"
        ),
        slots={"pp": S(_DIST_POOL), "qq": X(_DIST_POOL, ("pp",))},
    ),
    Template(
        name="hellinger_distance",
        latex=(
            r"H^2({pp}, {qq})"
            r" = 1 - \sum_x \sqrt{{{pp}(x)\, {qq}(x)}}"
        ),
        slots={"pp": S(_DIST_POOL), "qq": X(_DIST_POOL, ("pp",))},
    ),
    Template(
        name="chi_squared_divergence",
        latex=(
            r"\chi^2({pp} \| {qq})"
            r" = \sum_x \frac{{({pp}(x) - {qq}(x))^2}}{{{qq}(x)}}"
        ),
        slots={"pp": S(_DIST_POOL), "qq": X(_DIST_POOL, ("pp",))},
    ),
    Template(
        name="f_divergence",
        latex=(
            r"D_f({pp} \| {qq})"
            r" = \sum_x {qq}(x)\, f\!\left(\frac{{{pp}(x)}}{{{qq}(x)}}\right)"
        ),
        slots={"pp": S(_DIST_POOL), "qq": X(_DIST_POOL, ("pp",))},
    ),
    Template(
        name="bhattacharyya_distance",
        latex=(
            r"D_B({pp}, {qq})"
            r" = -\ln\!\left(\sum_x \sqrt{{{pp}(x)\, {qq}(x)}}\right)"
        ),
        slots={"pp": S(_DIST_POOL), "qq": X(_DIST_POOL, ("pp",))},
    ),
    Template(
        name="alpha_renyi_divergence",
        latex=(
            r"D_{{{aa}}}({pp} \| {qq})"
            r" = \frac{{1}}{{{aa}-1}}\, {lb}\sum_x"
            r" {pp}(x)^{{{aa}}}\, {qq}(x)^{{1-{aa}}}"
        ),
        slots={
            "pp": S(_DIST_POOL),
            "qq": X(_DIST_POOL, ("pp",)),
            "aa": S(_ALPHA_POOL),
            "lb": S(_LOG_POOL),
        },
    ),
]

# Channel coding (6)
_TEMPLATES_B_CHAN: list[Template] = [
    Template(
        name="shannon_awgn",
        latex=r"C = \tfrac{{1}}{{2}}\, {lb}(1 + {snr})",
        slots={"lb": S(_LOG_POOL), "snr": S(_SNR_POOL)},
    ),
    Template(
        name="gaussian_channel_capacity",
        latex=(r"C = {bw}\, {lb}\!\left(1 + \frac{{{pw}}}{{{nz}\cdot {bw}}}\right)"),
        slots={
            "lb": S(_LOG_POOL),
            "bw": S(_BW_POOL),
            "pw": S(_PWR_POOL),
            "nz": S(_NOISE_POOL),
        },
    ),
    Template(
        name="bsc_capacity",
        latex=(
            r"C = 1 - H_b({ee})"
            r" = 1 + {ee}\, {lb}\, {ee} + (1-{ee})\, {lb}(1-{ee})"
        ),
        slots={"ee": S(_EPS_POOL), "lb": S(_LOG_POOL)},
    ),
    Template(
        name="parallel_gaussian_channels",
        latex=(
            r"C = \sum{lim_mod}_{{k=1}}^{{{nn}}}"
            r" \tfrac{{1}}{{2}}\, {lb}\!\left(1 + \frac{{{pw}_k}}{{{nz}_k}}\right)"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "nn": S(_IDX_POOL),
            "lb": S(_LOG_POOL),
            "pw": S(_PWR_BASE_POOL),
            "nz": S(_NOISE_BASE_POOL),
        },
    ),
    Template(
        name="rate_distortion_gaussian",
        latex=(
            r"R(D) = \max\!\left(0,\,"
            r" \tfrac{{1}}{{2}}\, {lb}\frac{{{sv}}}{{D}}\right)"
        ),
        slots={"lb": S(_LOG_POOL), "sv": S(_NOISE_POOL)},
    ),
    Template(
        name="error_exponent",
        latex=r"P_e^{{({nn})}} \leq e^{{-{nn}\, E({rr})}}",
        slots={"nn": S(_IDX_POOL), "rr": S(_RATE_POOL)},
    ),
]

# Source coding (4)
_TEMPLATES_B_SRC: list[Template] = [
    Template(
        name="source_coding_theorem",
        latex=(
            r"H_{{{lb}}}({XX}) \leq L^*_{{avg}}"
            r" < H_{{{lb}}}({XX}) + 1"
        ),
        slots={"XX": S(_RV_POOL), "lb": S(_LOG_POOL)},
    ),
    Template(
        name="huffman_bound",
        latex=(
            r"H({XX}) \leq \bar{{L}}_{{Huffman}} < H({XX}) + 1,"
            r"\quad H({XX}) = -\sum_x p(x)\, {lb}\, p(x)"
        ),
        slots={"XX": S(_RV_POOL), "lb": S(_LOG_POOL)},
    ),
    Template(
        name="kraft_inequality",
        latex=r"\sum{lim_mod}_{{k=1}}^{{{nn}}} D^{{-l_k}} \leq 1",
        slots={"lim_mod": _LIM_MOD, "nn": S(_IDX_POOL)},
    ),
    Template(
        name="lempel_ziv_rate",
        latex=(
            r"\lim_{{{nn} \to \infty}}"
            r" \frac{{c_{{LZ}}({XX}_1^{{{nn}}})}}{{{nn}}} = H({XX})"
        ),
        slots={"XX": S(_RV_POOL), "nn": S(_IDX_POOL)},
    ),
]

# AEP / Typicality (4)
_TEMPLATES_B_AEP: list[Template] = [
    Template(
        name="aep",
        latex=(
            r"-\frac{{1}}{{{nn}}}\, {lb}\, p({XX}_1, \ldots, {XX}_{{{nn}}})"
            r" \xrightarrow{{p}} H({XX})"
        ),
        slots={"XX": S(_RV_POOL), "nn": S(_IDX_POOL), "lb": S(_LOG_POOL)},
    ),
    Template(
        name="typical_set_prob",
        latex=(
            r"\Pr\!\left[{XX}^{{{nn}}} \in"
            r" \mathcal{{T}}_{{{ee}}}^{{{nn}}}\right] \geq 1 - {ee}"
        ),
        slots={"XX": S(_RV_POOL), "nn": S(_IDX_POOL), "ee": S(_EPS_POOL)},
    ),
    Template(
        name="typical_set_size",
        latex=(
            r"\left|\mathcal{{T}}_{{{ee}}}^{{{nn}}}\right|"
            r" \leq 2^{{{nn}(H({XX})+{ee})}}"
        ),
        slots={"XX": S(_RV_POOL), "nn": S(_IDX_POOL), "ee": S(_EPS_POOL)},
    ),
    Template(
        name="joint_typicality_lemma",
        latex=(
            r"\Pr\!\left[(x^{{{nn}}}, y^{{{nn}}})"
            r" \in \mathcal{{T}}_{{{ee}}}^{{{nn}}}({XX},{YY})\right]"
            r" \approx 2^{{-{nn}\, I({XX};{YY})}}"
        ),
        slots={
            "XX": S(_RV_POOL),
            "YY": X(_RV_POOL, ("XX",)),
            "nn": S(_IDX_POOL),
            "ee": S(_EPS_POOL),
        },
    ),
]

# Fisher information / Cramér-Rao (6)
_TEMPLATES_B_FISHER: list[Template] = [
    Template(
        name="fisher_information",
        latex=(
            r"\mathcal{{I}}_{{{pp}}}({pa})"
            r" = \mathbb{{E}}\!\left["
            r"\left(\frac{{\partial}}{{\partial {pa}}} \ln {pp}(x;\, {pa})\right)^2"
            r"\right]"
        ),
        slots={"pp": S(_DIST_POOL), "pa": S(_PARAM_POOL)},
    ),
    Template(
        name="cramer_rao_bound",
        latex=(
            r"\mathrm{{Var}}\!\left(\hat{{{pa}}}\right)"
            r" \geq \frac{{1}}{{{nn}\, \mathcal{{I}}({pa})}}"
        ),
        slots={"pa": S(_PARAM_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="fisher_additivity",
        latex=(
            r"\mathcal{{I}}_{{X^{{{nn}}}}}({pa})"
            r" = {nn}\, \mathcal{{I}}_X({pa})"
        ),
        slots={"pa": S(_PARAM_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="fisher_matrix",
        latex=(
            r"\left[\mathcal{{I}}({pa})\right]_{{ij}}"
            r" = \mathbb{{E}}\!\left["
            r"\frac{{\partial \ln p}}{{\partial {pa}_i}}"
            r"\cdot \frac{{\partial \ln p}}{{\partial {pa}_j}}"
            r"\right]"
        ),
        slots={"pa": S(_PARAM_POOL)},
    ),
    Template(
        name="jeffreys_prior",
        latex=r"\pi({pa}) \propto \sqrt{{\det \mathcal{{I}}({pa})}}",
        slots={"pa": S(_PARAM_POOL)},
    ),
    Template(
        name="cramer_rao_biased",
        latex=(
            r"\mathrm{{Var}}\!\left(\hat{{{pa}}}\right)"
            r" \geq \frac{{\left[1 + b'({pa})\right]^2}}{{\mathcal{{I}}({pa})}}"
        ),
        slots={"pa": S(_PARAM_POOL)},
    ),
]

# MDL / Kolmogorov complexity (4)
_TEMPLATES_B_MDL: list[Template] = [
    Template(
        name="kolmogorov_complexity",
        latex=r"K({XX}) = \min\{{|d| : \mathcal{{U}}(d) = {XX}\}}",
        slots={"XX": S(_RV_POOL)},
    ),
    Template(
        name="mdl_two_part",
        latex=(
            r"L({XX},\, \hat{{{XX}}})"
            r" = L(\hat{{{XX}}}) + L({XX} \mid \hat{{{XX}}})"
        ),
        slots={"XX": S(_RV_POOL)},
    ),
    Template(
        name="normalized_compression_distance",
        latex=(
            r"d({XX},{YY})"
            r" = \frac{{K({XX}{YY}) - \min(K({XX}),\, K({YY}))}}"
            r"{{\max(K({XX}),\, K({YY}))}}"
        ),
        slots={"XX": S(_RV_POOL), "YY": X(_RV_POOL, ("XX",))},
    ),
    Template(
        name="aic_criterion",
        latex=r"\mathrm{{AIC}} = 2{kk} - 2\ln\hat{{{ll}}}",
        slots={"kk": S(_IDX_POOL), "ll": S(_DIST_POOL)},
    ),
]

# Network information theory (4)
_TEMPLATES_B_NET: list[Template] = [
    Template(
        name="mac_capacity_region",
        latex=(
            r"R_{{1}} + R_{{2}} \leq I({XX}, {YY};\, {ZZ}),"
            r"\quad R_{{1}} \leq I({XX};\, {ZZ} \mid {YY})"
        ),
        slots={
            "XX": S(_RV_POOL),
            "YY": X(_RV_POOL, ("XX",)),
            "ZZ": X(_RV_POOL, ("XX", "YY")),
        },
    ),
    Template(
        name="slepian_wolf",
        latex=(
            r"R_{{{XX}}} + R_{{{YY}}} \geq H({XX}, {YY}),"
            r"\quad R_{{{XX}}} \geq H({XX} \mid {YY})"
        ),
        slots={"XX": S(_RV_POOL), "YY": X(_RV_POOL, ("XX",))},
    ),
    Template(
        name="wyner_ziv",
        latex=r"R({XX} \mid {YY}) \geq H({XX} \mid {YY})",
        slots={"XX": S(_RV_POOL), "YY": X(_RV_POOL, ("XX",))},
    ),
    Template(
        name="broadcast_degraded",
        latex=(
            r"R_1 \leq I({UU};\, {YY}),"
            r"\quad R_2 \leq I({XX};\, {ZZ} \mid {UU})"
        ),
        slots={
            "UU": S(_RV_POOL),
            "XX": X(_RV_POOL, ("UU",)),
            "YY": X(_RV_POOL, ("UU", "XX")),
            "ZZ": X(_RV_POOL, ("UU", "XX", "YY")),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part C — high-n_eff function-pair templates (12)
# ---------------------------------------------------------------------------

_TEMPLATES_C: list[Template] = [
    Template(
        name="kl_pair_continuous",
        latex=(
            r"D_{{KL}}({fn1} \| {fn2})"
            r" = \int {fn1}(x)\, \ln \frac{{{fn1}(x)}}{{{fn2}(x)}}\, dx"
        ),
        slots={"fn1": _FN_SLOT, "fn2": _FN_SLOT},
    ),
    Template(
        name="entropy_pair_joint",
        latex=(
            r"H({fn1}, {fn2})"
            r" = -\iint {fn1}(x)\,{fn2}(y)\,"
            r"\ln\!\left[{fn1}(x)\,{fn2}(y)\right]\, dx\, dy"
        ),
        slots={"fn1": _FN_SLOT, "fn2": _FN_SLOT},
    ),
    Template(
        name="mi_pair",
        latex=(
            r"I({fn1};\, {fn2})"
            r" = \iint p(x,y)\, \ln \frac{{p(x,y)}}{{{fn1}(x)\, {fn2}(y)}}\, dx\, dy"
        ),
        slots={"fn1": _FN_SLOT, "fn2": _FN_SLOT},
    ),
    Template(
        name="cross_entropy_pair",
        latex=(
            r"H({fn1},\, {fn2})"
            r" = -\int {fn1}(x)\, \ln {fn2}(x)\, dx"
        ),
        slots={"fn1": _FN_SLOT, "fn2": _FN_SLOT},
    ),
    Template(
        name="js_pair",
        latex=(
            r"D_{{JS}}({fn1} \| {fn2})"
            r" = \tfrac{{1}}{{2}} D_{{KL}}({fn1} \| M)"
            r" + \tfrac{{1}}{{2}} D_{{KL}}({fn2} \| M),"
            r"\quad M = \tfrac{{{fn1}+{fn2}}}{{2}}"
        ),
        slots={"fn1": _FN_SLOT, "fn2": _FN_SLOT},
    ),
    Template(
        name="total_variation_pair",
        latex=(
            r"\mathrm{{TV}}({fn1},\, {fn2})"
            r" = \sup_{{A}} \left|{fn1}(A) - {fn2}(A)\right|"
        ),
        slots={"fn1": _FN_SLOT, "fn2": _FN_SLOT},
    ),
    Template(
        name="renyi_divergence_pair",
        latex=(
            r"D_{{{aa}}}({fn1} \| {fn2})"
            r" = \frac{{1}}{{{aa}-1}}"
            r" \ln \int {fn1}(x)^{{{aa}}}\, {fn2}(x)^{{1-{aa}}}\, dx"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "aa": S(_ALPHA_POOL),
        },
    ),
    Template(
        name="fisher_pair",
        latex=(
            r"\mathcal{{I}}_{{{fn1}}}({pa})"
            r" = -\mathbb{{E}}_{{{fn1}}}\!\left["
            r"\frac{{\partial^2 \ln {fn1}(x;\,{pa})}}{{\partial {pa}^2}}"
            r"\right]"
        ),
        slots={"fn1": _FN_SLOT, "pa": S(_PARAM_POOL)},
    ),
    Template(
        name="mutual_info_three_pair",
        latex=(
            r"I({fn1};\, {fn2} \mid {fn3})"
            r" = H({fn1} \mid {fn3}) - H({fn1} \mid {fn2},\, {fn3})"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "fn3": _FN_SLOT,
        },
    ),
    Template(
        name="capacity_pair",
        latex=r"C = \max_{{{fn1}}} I({fn1};\, {fn2})",
        slots={"fn1": _FN_SLOT, "fn2": _FN_SLOT},
    ),
    Template(
        name="data_processing_pair",
        latex=(
            r"D_{{KL}}\!\left({fn1}({YY}) \| {fn2}({YY})\right)"
            r" \leq D_{{KL}}\!\left({fn1}({XX}) \| {fn2}({XX})\right)"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "XX": S(_RV_POOL),
            "YY": X(_RV_POOL, ("XX",)),
        },
    ),
    Template(
        name="entropy_power_pair",
        latex=(
            r"e^{{2h({fn1}+{fn2})}}"
            r" \geq e^{{2h({fn1})}} + e^{{2h({fn2})}}"
        ),
        slots={"fn1": _FN_SLOT, "fn2": _FN_SLOT},
    ),
]

# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

_INFOTH_TEMPLATES: list[Template] = (
    _TEMPLATES_A
    + _TEMPLATES_B_ENTROPY
    + _TEMPLATES_B_MI
    + _TEMPLATES_B_DIV
    + _TEMPLATES_B_CHAN
    + _TEMPLATES_B_SRC
    + _TEMPLATES_B_AEP
    + _TEMPLATES_B_FISHER
    + _TEMPLATES_B_MDL
    + _TEMPLATES_B_NET
    + _TEMPLATES_C
)

# Part C additions
_INFOTH_TEMPLATES += [
    Template(
        name="fn_channel_capacity_pair",
        latex=r"{fn1}(C) = {fn2}\!\left(\max_{{{pp}}} I({XX}; {YY})\right)",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "pp": S(_DIST_POOL),
            "XX": S(_RV_POOL),
            "YY": X(_RV_POOL, ("XX",)),
        },
    ),
    Template(
        name="fn_source_coding_pair",
        latex=r"{fn1}(L^*) = {fn2}\!\left(\sum{lim_mod}_{{x}} p(x)\, {lb}\, \frac{{1}}{{p(x)}}\right)",
        slots={
            "lim_mod": _LIM_MOD,
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "lb": S(_LOG_POOL),
        },
    ),
    Template(
        name="fn_rate_distortion_pair",
        latex=r"{fn1}(R(D)) = {fn2}\!\left(\min_{{p(\hat{{x}}|x):\,\mathbb{{E}}[d]\leq D}} I(X;\hat{{X}})\right)",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
        },
    ),
    Template(
        name="fn_entropy_bound_pair",
        latex=r"{fn1}(H({XX})) \leq {fn2}({lb}\,|\mathcal{{X}}|)",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "XX": S(_RV_POOL),
            "lb": S(_LOG_POOL),
        },
    ),
]

# \lg — binary logarithm (log base 2) templates
_INFOTH_TEMPLATES += [
    Template(
        name="entropy_lg",
        latex=r"H({XX}) = -\sum{lim_mod}_{{x}} p(x) \lg p(x)",
        slots={"lim_mod": _LIM_MOD, "XX": S(_RV_POOL)},
    ),
    Template(
        name="shannon_hartley_lg",
        latex=r"C = {bw} \lg\!\left(1 + \frac{{{pw}}}{{{nz}}}\right)",
        slots={
            "bw": S(_BW_POOL),
            "pw": S(_PWR_BASE_POOL),
            "nz": S(_NOISE_BASE_POOL),
        },
    ),
    Template(
        name="mutual_info_lg",
        latex=(
            r"I({XX}; {YY})"
            r" = \sum{lim_mod}_{{x,y}} p(x,y) \lg \frac{{p(x,y)}}{{p(x)\,p(y)}}"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "XX": S(_RV_POOL),
            "YY": X(_RV_POOL, ("XX",)),
        },
    ),
]

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("information_theory", _INFOTH_TEMPLATES)
