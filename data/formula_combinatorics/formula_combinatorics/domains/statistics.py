"""Statistics domain generators — descriptive statistics, estimation theory,
hypothesis testing, confidence intervals, linear regression, ANOVA, Bayesian
inference, nonparametric tests, order statistics, multiple testing, and
survival analysis."""

from __future__ import annotations

from .._template_dsl import _LIM_MOD, E, S, Template, X, register_domain
from .._vocab import (
    _EXP_OP,
    _fn_rich_nosub,
)
from .._vocab import (
    _LAM_STATS as _LAM_POOL,
)
from .._vocab import (
    _MU_STATS as _MU_POOL,
)
from .._vocab import (
    _PROB_OP_FULL as _PROB_OP,
)
from .._vocab import (
    _RV_BASE as _RV_POOL,
)
from .._vocab import (
    _SIG_STATS as _SIG_POOL,
)
from .._vocab import (
    _STATS_N as _N_POOL,
)

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_PARAM_POOL = (r"\theta", r"\mu", r"\sigma", r"\lambda", r"\beta", r"\alpha", r"\eta", r"\phi")  # 8
_K_POOL = ("k", "p", "q", "r")  # 4
_IDX_POOL = ("i", "j", "k", "t")  # 4
_TIME_POOL = ("t", "s", "T", r"t_0", r"t_1")  # 5
_MM_POOL = ("m", "M", r"\ell", "K")  # 4 — number of tests / hypotheses
_RESP_POOL = ("y", "z", "w", r"\mathbf{y}", r"\mathbf{z}")  # 5 — response vectors
_COEFF_VEC_POOL = (r"\beta", r"\theta", r"\gamma", r"\alpha")  # 4 — coefficient vectors
_P_POOL = ("p", "q", r"\pi", r"\rho")  # 4 — probability/quantile levels

# ---------------------------------------------------------------------------
# Part A — Descriptive Statistics (8 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_A: list[Template] = [
    Template(
        name="sample_mean",
        latex=(
            r"\bar{{{rv}}}_{{{nn}}} = \frac{{1}}{{{nn}}}"
            r"\sum{lim_mod}_{{i=1}}^{{{nn}}} {rv}_i"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "rv": S(_RV_POOL),
            "nn": S(_N_POOL),
        },
    ),
    Template(
        name="sample_variance",
        latex=(
            r"s^2 = \frac{{1}}{{{nn}-1}}"
            r"\sum{lim_mod}_{{i=1}}^{{{nn}}}({rv}_i - \bar{{{rv}}})^2"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "rv": S(_RV_POOL),
            "nn": S(_N_POOL),
        },
    ),
    Template(
        name="sample_std_dev",
        latex=(
            r"s = \sqrt{{\frac{{1}}{{{nn}-1}}"
            r"\sum{lim_mod}_{{i=1}}^{{{nn}}}({rv}_i - \bar{{{rv}}})^2}}"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "rv": S(_RV_POOL),
            "nn": S(_N_POOL),
        },
    ),
    Template(
        name="sample_covariance",
        latex=(
            r"s_{{{rv1}{rv2}}} = \frac{{1}}{{{nn}-1}}"
            r"\sum{lim_mod}_{{i=1}}^{{{nn}}}"
            r"({rv1}_i - \bar{{{rv1}}})({rv2}_i - \bar{{{rv2}}})"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "rv1": S(_RV_POOL),
            "rv2": X(_RV_POOL, ("rv1",)),
            "nn": S(_N_POOL),
        },
    ),
    Template(
        name="sample_correlation",
        latex=(r"r = \frac{{s_{{{rv1}{rv2}}}}}{{s_{{{rv1}}}\,s_{{{rv2}}}}}"),
        slots={
            "rv1": S(_RV_POOL),
            "rv2": X(_RV_POOL, ("rv1",)),
        },
    ),
    Template(
        name="empirical_cdf",
        latex=(
            r"\hat{{F}}_{{{nn}}}(x) = \frac{{1}}{{{nn}}}"
            r"\sum{lim_mod}_{{i=1}}^{{{nn}}} \mathbf{{1}}[{rv}_i \leq x]"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "rv": S(_RV_POOL),
            "nn": S(_N_POOL),
        },
    ),
    Template(
        name="sample_quantile",
        latex=r"\hat{{Q}}({pp}) = {rv}_{{(\lceil {nn}\,{pp} \rceil)}}",
        slots={
            "rv": S(_RV_POOL),
            "nn": S(_N_POOL),
            "pp": S(_P_POOL),
        },
    ),
    Template(
        name="sample_range",
        latex=r"R = {rv}_{{({nn})}} - {rv}_{{(1)}}",
        slots={
            "rv": S(_RV_POOL),
            "nn": S(_N_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B1 — Estimation Theory (8 templates; 5 migrated from probability.py)
# ---------------------------------------------------------------------------

_TEMPLATES_B1: list[Template] = [
    # --- migrated ---
    Template(
        name="mle_argmax",
        latex=(
            r"\hat{{{lam}}}"
            r" = \operatorname{{arg\,max}}_{{{lam}}} \mathcal{{L}}({lam})"
        ),
        slots={"lam": S(_LAM_POOL)},
    ),
    Template(
        name="score_function",
        latex=(
            r"s({lam}) = \frac{{\partial}}{{\partial {lam}}}"
            r" \log \mathcal{{L}}({lam})"
        ),
        slots={"lam": S(_LAM_POOL)},
    ),
    Template(
        name="fisher_information",
        latex=(
            r"I({lam}) = {op}\!\left["
            r"\left(\frac{{\partial}}{{\partial {lam}}}"
            r" \log f(x;\,{lam})\right)^{{\!2}}\right]"
        ),
        slots={"op": S(_EXP_OP), "lam": S(_LAM_POOL)},
    ),
    Template(
        name="cramer_rao",
        latex=(
            r"\operatorname{{Var}}(\hat{{{lam}}})"
            r" \geq \dfrac{{1}}{{I({lam})}}"
        ),
        slots={"lam": S(_LAM_POOL)},
    ),
    Template(
        name="delta_method",
        latex=(
            r"\sqrt{{{nn}}}\,\bigl({fn}(\bar{{{rv}}}_n) - {fn}({mu})\bigr)"
            r" \xrightarrow{{d}} \mathcal{{N}}\!\left(0,\,{fn}'({mu})^2\,{sig}^2\right)"
        ),
        slots={
            "fn": E(_fn_rich_nosub, n=100),
            "rv": S(_RV_POOL),
            "mu": S(_MU_POOL),
            "sig": S(_SIG_POOL),
            "nn": S(_N_POOL),
        },
    ),
    # --- new ---
    Template(
        name="bias_estimator",
        latex=(
            r"\mathrm{{Bias}}(\hat{{{par}}})"
            r" = {op}[\hat{{{par}}}] - {par}"
        ),
        slots={"op": S(_EXP_OP), "par": S(_PARAM_POOL)},
    ),
    Template(
        name="mse_decomp",
        latex=(
            r"\mathrm{{MSE}}(\hat{{{par}}})"
            r" = \mathrm{{Bias}}(\hat{{{par}}})^2 + \operatorname{{Var}}(\hat{{{par}}})"
        ),
        slots={"par": S(_PARAM_POOL)},
    ),
    Template(
        name="asymptotic_normality",
        latex=(
            r"\sqrt{{{nn}}}(\hat{{{par}}}_{{{nn}}} - {par})"
            r" \xrightarrow{{d}} \mathcal{{N}}(0,\,V({par}))"
        ),
        slots={"nn": S(_N_POOL), "par": S(_PARAM_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B2 — Sufficient Statistics & Exponential Families (5 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B2: list[Template] = [
    # --- migrated ---
    Template(
        name="log_likelihood_sum",
        latex=(
            r"\ell({fn1}) = \sum{lim_mod}_{{i=1}}^{{{nn}}}"
            r" \log {fn2}(x_i \mid {fn1})"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "nn": S(_N_POOL),
        },
    ),
    Template(
        name="rao_blackwell",
        latex=(
            r"{op}\!\left[\bigl({fn1}({rv1}) - {fn2}({rv2})\bigr)^2\right]"
            r" \geq {op}\!\left[\bigl(\hat{{\theta}} - {fn2}({rv2})\bigr)^2\right]"
        ),
        slots={
            "op": S(_EXP_OP),
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "rv1": S(_RV_POOL),
            "rv2": X(_RV_POOL, ("rv1",)),
        },
    ),
    # --- new ---
    Template(
        name="sufficient_stat_factorization",
        latex=(
            r"f(\mathbf{{x}} \mid {par})"
            r" = g(T(\mathbf{{x}}),\,{par})\,h(\mathbf{{x}})"
        ),
        slots={"par": S(_PARAM_POOL)},
    ),
    Template(
        name="exponential_family",
        latex=(
            r"f(x \mid {par})"
            r" = h(x)\exp\!\bigl(\eta({par})\,T(x) - A({par})\bigr)"
        ),
        slots={"par": S(_PARAM_POOL)},
    ),
    Template(
        name="complete_statistic",
        latex=(
            r"{op}_{{par}}[g(T)] = 0\;"
            r"\forall\,{par} \;\Longrightarrow\; g(T) = 0 \;\text{{a.s.}}"
        ),
        slots={"op": S(_EXP_OP), "par": S(_PARAM_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B3 — Hypothesis Testing (10 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B3: list[Template] = [
    Template(
        name="z_test_statistic",
        latex=(
            r"Z = \frac{{\bar{{{rv}}} - {mu}}}{{{sig}/\sqrt{{{nn}}}}}"
            r" \sim \mathcal{{N}}(0,1)"
        ),
        slots={
            "rv": S(_RV_POOL),
            "mu": S(_MU_POOL),
            "sig": S(_SIG_POOL),
            "nn": S(_N_POOL),
        },
    ),
    Template(
        name="one_sample_t_test",
        latex=(
            r"T = \frac{{\bar{{{rv}}} - {mu}}}{{s/\sqrt{{{nn}}}}}"
            r" \sim t_{{{nn}-1}}"
        ),
        slots={
            "rv": S(_RV_POOL),
            "mu": S(_MU_POOL),
            "nn": S(_N_POOL),
        },
    ),
    Template(
        name="two_sample_t_test",
        latex=(
            r"T = \frac{{\bar{{{rv1}}} - \bar{{{rv2}}}}}"
            r"{{s_p\sqrt{{1/{nn1}+1/{nn2}}}}}"
            r",\quad s_p^2 = \frac{{({nn1}-1)s_{{{rv1}}}^2+({nn2}-1)s_{{{rv2}}}^2}}"
            r"{{{nn1}+{nn2}-2}}"
        ),
        slots={
            "rv1": S(_RV_POOL),
            "rv2": X(_RV_POOL, ("rv1",)),
            "nn1": S(_N_POOL),
            "nn2": X(_N_POOL, ("nn1",)),
        },
    ),
    Template(
        name="pooled_variance",
        latex=(
            r"s_p^2 = \frac{{({nn1}-1)s_{{{rv1}}}^2+({nn2}-1)s_{{{rv2}}}^2}}"
            r"{{{nn1}+{nn2}-2}}"
        ),
        slots={
            "rv1": S(_RV_POOL),
            "rv2": X(_RV_POOL, ("rv1",)),
            "nn1": S(_N_POOL),
            "nn2": X(_N_POOL, ("nn1",)),
        },
    ),
    Template(
        name="chi_squared_gof",
        latex=(
            r"\chi^2 = \sum{lim_mod}_{{i=1}}^{{{kk}}}"
            r" \frac{{(O_i - E_i)^2}}{{E_i}}"
            r" \sim \chi^2_{{{kk}-1}}"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "kk": S(_K_POOL),
        },
    ),
    Template(
        name="f_statistic",
        latex=(
            r"F = \frac{{s_{{{rv1}}}^2}}{{s_{{{rv2}}}^2}}"
            r" \sim F_{{{nn1}-1,\,{nn2}-1}}"
        ),
        slots={
            "rv1": S(_RV_POOL),
            "rv2": X(_RV_POOL, ("rv1",)),
            "nn1": S(_N_POOL),
            "nn2": X(_N_POOL, ("nn1",)),
        },
    ),
    Template(
        name="likelihood_ratio_test",
        latex=(
            r"\Lambda = -2\log\frac{{L(\hat{{{par}}}_0)}}{{L(\hat{{{par}}})}}"
            r" \xrightarrow{{d}} \chi^2_{{{kk}}}"
        ),
        slots={
            "par": S(_PARAM_POOL),
            "kk": S(_K_POOL),
        },
    ),
    Template(
        name="neyman_pearson_lemma",
        latex=(
            r"\frac{{f(\mathbf{{x}} \mid {par1})}}{{f(\mathbf{{x}} \mid {par2})}}"
            r" \gtrless c"
        ),
        slots={
            "par1": S(_PARAM_POOL),
            "par2": X(_PARAM_POOL, ("par1",)),
        },
    ),
    Template(
        name="p_value_def",
        latex=r"p = {op}(T \geq t_{{\mathrm{{obs}}}} \mid H_0)",
        slots={"op": S(_PROB_OP)},
    ),
    Template(
        name="power_function",
        latex=r"\beta({par}) = P(\text{{reject }}\,H_0 \mid {par})",
        slots={"par": S(_PARAM_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B4 — Confidence Intervals (6 templates; 1 migrated)
# ---------------------------------------------------------------------------

_TEMPLATES_B4: list[Template] = [
    # --- migrated ---
    Template(
        name="sample_mean_normal_approx",
        latex=r"\bar{{{rv}}}_{{{nn}}} \approx \mathcal{{N}}\!\left({mu},\,\frac{{{sig}^2}}{{{nn}}}\right)",
        slots={"rv": S(_RV_POOL), "nn": S(_N_POOL), "mu": S(_MU_POOL), "sig": S(_SIG_POOL)},
    ),
    # --- new ---
    Template(
        name="ci_known_variance",
        latex=(r"\bar{{{rv}}} \pm z_{{\alpha/2}}\,\frac{{{sig}}}{{\sqrt{{{nn}}}}}"),
        slots={
            "rv": S(_RV_POOL),
            "sig": S(_SIG_POOL),
            "nn": S(_N_POOL),
        },
    ),
    Template(
        name="ci_unknown_variance",
        latex=(r"\bar{{{rv}}} \pm t_{{{nn}-1,\,\alpha/2}}\,\frac{{s}}{{\sqrt{{{nn}}}}}"),
        slots={
            "rv": S(_RV_POOL),
            "nn": S(_N_POOL),
        },
    ),
    Template(
        name="ci_proportion",
        latex=(
            r"\hat{{p}} \pm z_{{\alpha/2}}"
            r"\sqrt{{\frac{{\hat{{p}}(1-\hat{{p}})}}{{{nn}}}}}"
        ),
        slots={"nn": S(_N_POOL)},
    ),
    Template(
        name="ci_coverage",
        latex=(
            r"P\!\left(\hat{{{par}}}_L \leq {par} \leq \hat{{{par}}}_U\right)"
            r" = 1 - \alpha"
        ),
        slots={"par": S(_PARAM_POOL)},
    ),
    Template(
        name="chi_squared_variance_ci",
        latex=(
            r"\left[\frac{{({nn}-1)s^2}}{{\chi^2_{{{nn}-1,\,\alpha/2}}}},"
            r"\;\frac{{({nn}-1)s^2}}{{\chi^2_{{{nn}-1,\,1-\alpha/2}}}}\right]"
        ),
        slots={"nn": S(_N_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B5 — Linear Regression (8 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B5: list[Template] = [
    Template(
        name="ols_normal_equations",
        latex=(r"\hat{{{coeff}}} = (X^\top X)^{{-1}}X^\top {resp}"),
        slots={
            "coeff": S(_COEFF_VEC_POOL),
            "resp": S(_RESP_POOL),
        },
    ),
    Template(
        name="fitted_values",
        latex=r"\hat{{{resp}}} = X\hat{{{coeff}}} = H\,{resp}",
        slots={
            "coeff": S(_COEFF_VEC_POOL),
            "resp": S(_RESP_POOL),
        },
    ),
    Template(
        name="hat_matrix",
        latex=r"H = X(X^\top X)^{{-1}}X^\top,\quad H^2 = H,\quad H^\top = H",
        slots={},
    ),
    Template(
        name="rss_formula",
        latex=(
            r"\mathrm{{RSS}} = \|{resp} - X\hat{{{coeff}}}\|^2"
            r" = {resp}^\top(I - H){resp}"
        ),
        slots={
            "coeff": S(_COEFF_VEC_POOL),
            "resp": S(_RESP_POOL),
        },
    ),
    Template(
        name="r_squared",
        latex=(
            r"R^2 = 1 - \frac{{\|{resp} - X\hat{{{coeff}}}\|^2}}"
            r"{{\|{resp} - \bar{{{resp}}}\,\mathbf{{1}}\|^2}}"
        ),
        slots={
            "coeff": S(_COEFF_VEC_POOL),
            "resp": S(_RESP_POOL),
        },
    ),
    Template(
        name="regression_variance_est",
        latex=(
            r"\hat{{\sigma}}^2 = \frac{{\mathrm{{RSS}}}}{{{nn}-{kk}}}"
            r" = \frac{{\|{resp} - X\hat{{{coeff}}}\|^2}}{{{nn}-{kk}}}"
        ),
        slots={
            "coeff": S(_COEFF_VEC_POOL),
            "resp": S(_RESP_POOL),
            "nn": S(_N_POOL),
            "kk": S(_K_POOL),
        },
    ),
    Template(
        name="simple_regression_slope",
        latex=(
            r"\hat{{{coeff}}}_1 = \frac{{\sum_{{i=1}}^{{{nn}}}"
            r"(x_i - \bar{{x}})({resp}_i - \bar{{{resp}}})}}"
            r"{{\sum_{{i=1}}^{{{nn}}}(x_i - \bar{{x}})^2}}"
        ),
        slots={
            "coeff": S(_COEFF_VEC_POOL),
            "resp": S(_RESP_POOL),
            "nn": S(_N_POOL),
        },
    ),
    Template(
        name="gauss_markov",
        latex=(
            r"\operatorname{{Var}}(\tilde{{{coeff}}})"
            r" \geq \operatorname{{Var}}(\hat{{{coeff}}}_\mathrm{{OLS}})"
            r"\quad \forall\,\tilde{{{coeff}}}\text{{ linear unbiased}}"
        ),
        slots={"coeff": S(_COEFF_VEC_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B6 — ANOVA (4 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B6: list[Template] = [
    Template(
        name="anova_decomp",
        latex=(
            r"SS_T = SS_B + SS_W,"
            r"\quad df_B = {kk}-1,"
            r"\quad df_W = {nn}-{kk}"
        ),
        slots={"kk": S(_K_POOL), "nn": S(_N_POOL)},
    ),
    Template(
        name="anova_f_ratio",
        latex=(
            r"F = \frac{{SS_B/({kk}-1)}}{{SS_W/({nn}-{kk})}}"
            r" \sim F_{{{kk}-1,\,{nn}-{kk}}}"
        ),
        slots={"kk": S(_K_POOL), "nn": S(_N_POOL)},
    ),
    Template(
        name="anova_between",
        latex=(
            r"SS_B = \sum{lim_mod}_{{i=1}}^{{{kk}}}"
            r" {nn}_i(\bar{{X}}_{{i\cdot}} - \bar{{X}}_{{\cdot\cdot}})^2"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "kk": S(_K_POOL),
            "nn": S(_N_POOL),
        },
    ),
    Template(
        name="anova_within",
        latex=(
            r"SS_W = \sum_{{i=1}}^{{{kk}}}\sum_{{j=1}}^{{{nn}}_i}"
            r"(X_{{ij}} - \bar{{X}}_{{i\cdot}})^2"
        ),
        slots={"kk": S(_K_POOL), "nn": S(_N_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B7 — Bayesian Statistics (8 templates; 1 migrated)
# ---------------------------------------------------------------------------

_TEMPLATES_B7: list[Template] = [
    # --- migrated ---
    Template(
        name="bayes_posterior",
        latex=(
            r"p({lam} \mid \mathbf{{x}})"
            r" \propto p(\mathbf{{x}} \mid {lam})\,p({lam})"
        ),
        slots={"lam": S(_LAM_POOL)},
    ),
    # --- new ---
    Template(
        name="map_estimator",
        latex=(
            r"\hat{{{par}}}_\mathrm{{MAP}}"
            r" = \operatorname{{arg\,max}}_{{{par}}} p({par} \mid \mathbf{{x}})"
        ),
        slots={"par": S(_PARAM_POOL)},
    ),
    Template(
        name="posterior_mean",
        latex=(
            r"{op}[{par} \mid \mathbf{{x}}]"
            r" = \int {par}\, p({par} \mid \mathbf{{x}})\, d{par}"
        ),
        slots={"op": S(_EXP_OP), "par": S(_PARAM_POOL)},
    ),
    Template(
        name="credible_interval",
        latex=(r"P\!\left({par} \in C \mid \mathbf{{x}}\right) = 1 - \alpha"),
        slots={"par": S(_PARAM_POOL)},
    ),
    Template(
        name="posterior_predictive",
        latex=(
            r"p(\tilde{{{rv}}} \mid \mathbf{{x}})"
            r" = \int p(\tilde{{{rv}}} \mid {par})\,p({par} \mid \mathbf{{x}})\, d{par}"
        ),
        slots={"rv": S(_RV_POOL), "par": S(_PARAM_POOL)},
    ),
    Template(
        name="marginal_likelihood",
        latex=(r"p(\mathbf{{x}}) = \int p(\mathbf{{x}} \mid {par})\,p({par})\, d{par}"),
        slots={"par": S(_PARAM_POOL)},
    ),
    Template(
        name="jeffreys_prior",
        latex=r"p({par}) \propto \sqrt{{I({par})}}",
        slots={"par": S(_PARAM_POOL)},
    ),
    Template(
        name="conjugate_update_beta",
        latex=(
            r"\mathrm{{Beta}}({al},\,{bt})"
            r" \xrightarrow{{\text{{update}}}}"
            r" \mathrm{{Beta}}({al}+{kk},\,{bt}+{nn}-{kk})"
        ),
        slots={
            "al": S(_LAM_POOL),
            "bt": X(_LAM_POOL, ("al",)),
            "kk": S(_K_POOL),
            "nn": S(_N_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B8 — Nonparametric Statistics (6 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B8: list[Template] = [
    Template(
        name="ks_statistic",
        latex=r"D_{{{nn}}} = \sup_x \lvert \hat{{F}}_{{{nn}}}(x) - F(x) \rvert",
        slots={"nn": S(_N_POOL)},
    ),
    Template(
        name="glivenko_cantelli",
        latex=(
            r"\sup_x \lvert \hat{{F}}_{{{nn}}}(x) - F(x) \rvert"
            r" \xrightarrow{{\mathrm{{a.s.}}}} 0"
        ),
        slots={"nn": S(_N_POOL)},
    ),
    Template(
        name="spearman_correlation",
        latex=(
            r"r_s = 1 - \frac{{6\sum{lim_mod}_{{i=1}}^{{{nn}}} d_i^2}}"
            r"{{{nn}({nn}^2-1)}}"
        ),
        slots={"lim_mod": _LIM_MOD, "nn": S(_N_POOL)},
    ),
    Template(
        name="wilcoxon_signed_rank",
        latex=(
            r"W^+ = \sum{lim_mod}_{{i=1}}^{{{nn}}}"
            r" R_i\,\mathbf{{1}}[{rv}_i > {mu}]"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "rv": S(_RV_POOL),
            "mu": S(_MU_POOL),
            "nn": S(_N_POOL),
        },
    ),
    Template(
        name="mann_whitney_u",
        latex=(
            r"U = \sum_{{i=1}}^{{{nn1}}}\sum_{{j=1}}^{{{nn2}}}"
            r" \mathbf{{1}}[{rv1}_i > {rv2}_j]"
        ),
        slots={
            "rv1": S(_RV_POOL),
            "rv2": X(_RV_POOL, ("rv1",)),
            "nn1": S(_N_POOL),
            "nn2": X(_N_POOL, ("nn1",)),
        },
    ),
    Template(
        name="sign_test_statistic",
        latex=(
            r"B = \sum{lim_mod}_{{i=1}}^{{{nn}}}"
            r" \mathbf{{1}}[{rv}_i > {mu}]"
            r" \sim \mathrm{{Bin}}\!\left({nn},\tfrac{{1}}{{2}}\right)"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "rv": S(_RV_POOL),
            "mu": S(_MU_POOL),
            "nn": S(_N_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B9 — Order Statistics (4 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B9: list[Template] = [
    Template(
        name="order_stat_pdf",
        latex=(
            r"f_{{{rv}_{{({kk})}}}}"
            r"(x) = \frac{{{nn}!}}{{({kk}-1)!\,({nn}-{kk})!}}"
            r"[F(x)]^{{{kk}-1}}\,[1-F(x)]^{{{nn}-{kk}}}\,f(x)"
        ),
        slots={
            "rv": S(_RV_POOL),
            "kk": S(_K_POOL),
            "nn": S(_N_POOL),
        },
    ),
    Template(
        name="min_cdf",
        latex=(
            r"F_{{{rv}_{{(1)}}}}"
            r"(x) = 1 - [1-F(x)]^{{{nn}}}"
        ),
        slots={"rv": S(_RV_POOL), "nn": S(_N_POOL)},
    ),
    Template(
        name="max_cdf",
        latex=(
            r"F_{{{rv}_{{({nn})}}}}"
            r"(x) = [F(x)]^{{{nn}}}"
        ),
        slots={"rv": S(_RV_POOL), "nn": S(_N_POOL)},
    ),
    Template(
        name="sample_median_approx",
        latex=(
            r"\tilde{{{rv}}} \approx \mathcal{{N}}\!\left({mu},\,"
            r"\frac{{\pi\,{sig}^2}}{{2\,{nn}}}\right)"
            r"\quad ({nn} \text{{ large}})"
        ),
        slots={
            "rv": S(_RV_POOL),
            "mu": S(_MU_POOL),
            "sig": S(_SIG_POOL),
            "nn": S(_N_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B10 — Multiple Testing (3 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B10: list[Template] = [
    Template(
        name="bonferroni_correction",
        latex=r"\alpha_i \leq \frac{{\alpha}}{{{mm}}}",
        slots={"mm": S(_MM_POOL)},
    ),
    Template(
        name="fdr_bh_threshold",
        latex=(
            r"p_{{({kk})}} \leq \frac{{{kk}\,\alpha}}{{{mm}}}"
            r"\quad \text{{(Benjamini--Hochberg)}}"
        ),
        slots={"kk": S(_K_POOL), "mm": S(_MM_POOL)},
    ),
    Template(
        name="fwer_definition",
        latex=(
            r"\mathrm{{FWER}} = P\!\left("
            r"\bigcup_{{i \in \mathcal{{H}}_0}} \{{\text{{reject }}{par}_i\}}\right)"
        ),
        slots={"par": S(_PARAM_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B11 — Survival Analysis (5 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B11: list[Template] = [
    Template(
        name="survival_function",
        latex=r"S({tt}) = P({rv} > {tt}) = 1 - F({tt})",
        slots={"rv": S(_RV_POOL), "tt": S(_TIME_POOL)},
    ),
    Template(
        name="hazard_function",
        latex=r"h({tt}) = \frac{{f({tt})}}{{S({tt})}}",
        slots={"tt": S(_TIME_POOL)},
    ),
    Template(
        name="cumulative_hazard",
        latex=(
            r"H({tt}) = -\log S({tt})"
            r" = \int_0^{{{tt}}} h(s)\, ds"
        ),
        slots={"tt": S(_TIME_POOL)},
    ),
    Template(
        name="hazard_survival_link",
        latex=r"S({tt}) = \exp(-H({tt}))",
        slots={"tt": S(_TIME_POOL)},
    ),
    Template(
        name="kaplan_meier",
        latex=(
            r"\hat{{S}}({tt}) = \prod_{{t_i \leq {tt}}}"
            r"\!\left(1 - \frac{{d_i}}{{n_i}}\right)"
        ),
        slots={"tt": S(_TIME_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part C — High-n_eff function-pair templates (6 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_C: list[Template] = [
    Template(
        name="fn_likelihood_ratio",
        latex=(
            r"{fn1}\!\left(-2\log\frac{{L(\hat{{{par}}}_0)}}{{L(\hat{{{par}}})}}\right)"
            r" = {fn2}(\Lambda)"
        ),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "par": S(_PARAM_POOL),
        },
    ),
    Template(
        name="fn_posterior_transform",
        latex=(
            r"{fn1}\!\left(\int {par}\,p({par} \mid \mathbf{{x}})\,d{par}\right)"
            r" = {fn2}\!\left({op}[{par} \mid \mathbf{{x}}]\right)"
        ),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "op": S(_EXP_OP),
            "par": S(_PARAM_POOL),
        },
    ),
    Template(
        name="fn_ols_estimator",
        latex=(
            r"{fn1}(\hat{{{coeff}}})"
            r" = {fn2}\!\left((X^\top X)^{{-1}}X^\top {resp}\right)"
        ),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "coeff": S(_COEFF_VEC_POOL),
            "resp": S(_RESP_POOL),
        },
    ),
    Template(
        name="fn_test_statistic",
        latex=(
            r"{fn1}\!\left(\frac{{\bar{{{rv}}} - {mu}}}{{s/\sqrt{{{nn}}}}}\right)"
            r" = {fn2}(t_{{{nn}-1}})"
        ),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "rv": S(_RV_POOL),
            "mu": S(_MU_POOL),
            "nn": S(_N_POOL),
        },
    ),
    Template(
        name="fn_sufficient_stat",
        latex=(
            r"{fn1}(f(\mathbf{{x}} \mid {par}))"
            r" = {fn2}(g(T(\mathbf{{x}}),\,{par})\,h(\mathbf{{x}}))"
        ),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "par": S(_PARAM_POOL),
        },
    ),
    Template(
        name="fn_survival_hazard",
        latex=(
            r"{fn1}(S({tt}))"
            r" = {fn2}\!\left(\exp\!\left(-\int_0^{{{tt}}} h(s)\,ds\right)\right)"
        ),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "tt": S(_TIME_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

_STAT_TEMPLATES: list[Template] = (
    _TEMPLATES_A
    + _TEMPLATES_B1
    + _TEMPLATES_B2
    + _TEMPLATES_B3
    + _TEMPLATES_B4
    + _TEMPLATES_B5
    + _TEMPLATES_B6
    + _TEMPLATES_B7
    + _TEMPLATES_B8
    + _TEMPLATES_B9
    + _TEMPLATES_B10
    + _TEMPLATES_B11
    + _TEMPLATES_C
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("statistics", _STAT_TEMPLATES, 0.04)
