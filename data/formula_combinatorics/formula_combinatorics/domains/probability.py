"""Probability and statistics domain generators."""

from __future__ import annotations

from ..engine._template_dsl import _FN_SLOT, _LIM_MOD, S, Template, X
from ..engine._vocab import (
    _EXP_OP,
)
from ..engine._vocab import (
    _GENERIC_IDX as _IDX_POOL,
)
from ..engine._vocab import (
    _LAM_STATS as _LAM_POOL,
)
from ..engine._vocab import (
    _MU_STATS as _MU_POOL,
)
from ..engine._vocab import (
    _PROB_OP_FULL as _PROB_OP,
)
from ..engine._vocab import (
    _RV_BASE as _RV_POOL,
)
from ..engine._vocab import (
    _SIG_STATS as _SIG_POOL,
)
from ..engine._vocab import (
    _STATS_N as _N_POOL,
)
from ._config import register_domain

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_EVENT_POOL = ("A", "B", "C", "D", "E", "F", r"A_1", r"B_1")
_K_POOL = (r"k", r"\ell", "j", "r")
_P_POOL = ("p", "q", r"\theta", r"\pi", r"\rho")
_A_POOL = ("a", "b", "c", r"\varepsilon")
_AB_POOL = ("a", "b", r"\alpha", r"\beta")
_BERN_K = ("0", "1", "2", "3")

# ---------------------------------------------------------------------------
# Probability templates
# ---------------------------------------------------------------------------

_PROB_TEMPLATES: list[Template] = [
    # ------------------------------------------------------------------
    # Part A: reparameterized originals
    # ------------------------------------------------------------------
    Template(
        name="binomial_pmf",
        latex=(
            r"{op}({rv} = {kk}) = \binom{{{nn}}}{{{kk}}}"
            r" {pp}^{{{kk}}} (1-{pp})^{{{nn}-{kk}}}"
        ),
        slots={
            "op": S(_PROB_OP),
            "rv": S(_RV_POOL),
            "kk": S(_K_POOL),
            "nn": S(_N_POOL),
            "pp": S(_P_POOL),
        },
    ),
    Template(
        name="poisson_pmf",
        latex=r"{op}({rv} = {kk}) = \frac{{{lam}^{{{kk}}} e^{{-{lam}}}}}{{{kk}!}}",
        slots={
            "op": S(_PROB_OP),
            "rv": S(_RV_POOL),
            "kk": S(_K_POOL),
            "lam": S(_LAM_POOL),
        },
    ),
    Template(
        name="normal_pdf",
        latex=(
            r"f(x) = \frac{{1}}{{\sqrt{{2\pi}}\,{sig}}}"
            r" \exp\!\left(-\frac{{(x-{mu})^2}}{{2\,{sig}^2}}\right)"
        ),
        slots={"mu": S(_MU_POOL), "sig": S(_SIG_POOL)},
    ),
    Template(
        name="expected_value_discrete",
        latex=r"{op}[{rv}] = \sum{lim_mod}_{{k}} k \cdot P({rv} = k)",
        slots={"lim_mod": _LIM_MOD, "op": S(_EXP_OP), "rv": S(_RV_POOL)},
    ),
    Template(
        name="expected_value_continuous",
        latex=r"{op}[{rv}] = \int{lim_mod}_{{-\infty}}^{{\infty}} x\, f_{{{rv}}}(x)\, dx",
        slots={"lim_mod": _LIM_MOD, "op": S(_EXP_OP), "rv": S(_RV_POOL)},
    ),
    Template(
        name="variance",
        latex=(
            r"\operatorname{{Var}}({rv})"
            r" = {op}\!\left[{rv}^2\right] - \bigl({op}[{rv}]\bigr)^2"
        ),
        slots={"op": S(_EXP_OP), "rv": S(_RV_POOL)},
    ),
    Template(
        name="bayes_theorem",
        latex=(
            r"{op}({ev1} \mid {ev2})"
            r" = \frac{{{op}({ev2} \mid {ev1})\,{op}({ev1})}}{{{op}({ev2})}}"
        ),
        slots={
            "op": S(_PROB_OP),
            "ev1": S(_EVENT_POOL),
            "ev2": X(_EVENT_POOL, ("ev1",)),
        },
    ),
    Template(
        name="law_of_total_probability",
        latex=(
            r"{op}({ev}) = \sum{lim_mod}_{{i=1}}^{{{nn}}}"
            r" {op}({ev} \mid B_i)\,{op}(B_i)"
        ),
        slots={"lim_mod": _LIM_MOD, "op": S(_PROB_OP), "ev": S(_EVENT_POOL), "nn": S(_N_POOL)},
    ),
    Template(
        name="mgf",
        latex=(
            r"M_{{{rv}}}(t) = {op}\!\left[e^{{t\,{rv}}}\right]"
            r" = \sum{lim_mod}_{{k=0}}^{{\infty}} \frac{{{op}[{rv}^k]}}{{k!}}\,t^k"
        ),
        slots={"lim_mod": _LIM_MOD, "op": S(_EXP_OP), "rv": S(_RV_POOL)},
    ),
    Template(
        name="covariance",
        latex=(
            r"\operatorname{{Cov}}({rv1},{rv2})"
            r" = {op}[{rv1}\,{rv2}] - {op}[{rv1}]\,{op}[{rv2}]"
        ),
        slots={
            "op": S(_EXP_OP),
            "rv1": S(_RV_POOL),
            "rv2": X(_RV_POOL, ("rv1",)),
        },
    ),
    Template(
        name="cdf",
        latex=r"F_{{{rv}}}(x) = {op}({rv} \leq x) = \int{lim_mod}_{{-\infty}}^{{x}} f_{{{rv}}}(t)\, dt",
        slots={"lim_mod": _LIM_MOD, "op": S(_PROB_OP), "rv": S(_RV_POOL)},
    ),
    Template(
        name="geometric_pmf",
        latex=r"{op}({rv} = {kk}) = (1-{pp})^{{{kk}-1}}\,{pp}",
        slots={
            "op": S(_PROB_OP),
            "rv": S(_RV_POOL),
            "kk": S(_K_POOL),
            "pp": S(_P_POOL),
        },
    ),
    Template(
        name="jensens_inequality",
        latex=r"{fn}\!\bigl({op}[{rv}]\bigr) \leq {op}\!\bigl[{fn}({rv})\bigr]",
        slots={
            "fn": _FN_SLOT,
            "op": S(_EXP_OP),
            "rv": S(_RV_POOL),
        },
    ),
    Template(
        name="correlation",
        latex=(
            r"\rho_{{{rv1}{rv2}}} = "
            r"\frac{{\operatorname{{Cov}}({rv1},{rv2})}}"
            r"{{\sqrt{{\operatorname{{Var}}({rv1})\,\operatorname{{Var}}({rv2})}}}}"
        ),
        slots={
            "rv1": S(_RV_POOL),
            "rv2": X(_RV_POOL, ("rv1",)),
        },
    ),
    Template(
        name="central_limit_theorem",
        latex=(
            r"\frac{{\bar{{{rv}}}_n - {mu}}}{{{sig}/\sqrt{{{nn}}}}}"
            r" \xrightarrow{{d}} \mathcal{{N}}(0,1)"
        ),
        slots={
            "rv": S(_RV_POOL),
            "mu": S(_MU_POOL),
            "sig": S(_SIG_POOL),
            "nn": S(_N_POOL),
        },
    ),
    Template(
        name="law_of_large_numbers",
        latex=r"\bar{{{rv}}}_n \xrightarrow{{p}} \mu \text{{ as }} n \to \infty",
        slots={"rv": S(_RV_POOL)},
    ),
    Template(
        name="characteristic_function",
        latex=r"\varphi_{{{rv}}}(t) = {op}\!\left[e^{{it\,{rv}}}\right]",
        slots={"op": S(_EXP_OP), "rv": S(_RV_POOL)},
    ),
    Template(
        name="tower_property",
        latex=(r"{op}\!\Bigl[{op}[{rv} \mid \mathcal{{F}}]\Bigr] = {op}[{rv}]"),
        slots={"op": S(_EXP_OP), "rv": S(_RV_POOL)},
    ),
    # ------------------------------------------------------------------
    # Part B: new flat standalone templates
    # ------------------------------------------------------------------
    # Core probability rules
    Template(
        name="complement_rule",
        latex=r"{op}({ev}^c) = 1 - {op}({ev})",
        slots={"op": S(_PROB_OP), "ev": S(_EVENT_POOL)},
    ),
    Template(
        name="addition_rule",
        latex=(
            r"{op}({ev1} \cup {ev2})"
            r" = {op}({ev1}) + {op}({ev2}) - {op}({ev1} \cap {ev2})"
        ),
        slots={
            "op": S(_PROB_OP),
            "ev1": S(_EVENT_POOL),
            "ev2": X(_EVENT_POOL, ("ev1",)),
        },
    ),
    Template(
        name="conditional_prob",
        latex=(
            r"{op}({ev1} \mid {ev2})"
            r" = \dfrac{{{op}({ev1} \cap {ev2})}}{{{op}({ev2})}}"
        ),
        slots={
            "op": S(_PROB_OP),
            "ev1": S(_EVENT_POOL),
            "ev2": X(_EVENT_POOL, ("ev1",)),
        },
    ),
    Template(
        name="multiplication_rule",
        latex=(
            r"{op}({ev1} \cap {ev2})"
            r" = {op}({ev1} \mid {ev2})\,{op}({ev2})"
        ),
        slots={
            "op": S(_PROB_OP),
            "ev1": S(_EVENT_POOL),
            "ev2": X(_EVENT_POOL, ("ev1",)),
        },
    ),
    Template(
        name="independence_events",
        latex=r"{op}({ev1} \cap {ev2}) = {op}({ev1})\,{op}({ev2})",
        slots={
            "op": S(_PROB_OP),
            "ev1": S(_EVENT_POOL),
            "ev2": X(_EVENT_POOL, ("ev1",)),
        },
    ),
    Template(
        name="union_bound",
        latex=(
            r"{op}\!\Bigl(\bigcup_{{i=1}}^{{{nn}}} A_i\Bigr)"
            r" \leq \sum{lim_mod}_{{i=1}}^{{{nn}}} {op}(A_i)"
        ),
        slots={"lim_mod": _LIM_MOD, "op": S(_PROB_OP), "nn": S(_N_POOL)},
    ),
    # Continuous distributions
    Template(
        name="exponential_pdf",
        latex=r"f(x;\,{lam}) = {lam}\,e^{{-{lam}\,x}},\quad x \geq 0",
        slots={"lam": S(_LAM_POOL)},
    ),
    Template(
        name="exponential_mean",
        latex=r"{op}[{rv}] = \tfrac{{1}}{{{lam}}} \quad \bigl(\mathrm{{Exp}}({lam})\bigr)",
        slots={"op": S(_EXP_OP), "rv": S(_RV_POOL), "lam": S(_LAM_POOL)},
    ),
    Template(
        name="exponential_memoryless",
        latex=(r"{op}({rv} > s + t \mid {rv} > s) = {op}({rv} > t)"),
        slots={"op": S(_PROB_OP), "rv": S(_RV_POOL)},
    ),
    Template(
        name="gamma_pdf",
        latex=(
            r"f(x;\,{al},{bt}) ="
            r" \dfrac{{x^{{{al}-1}}\,e^{{-x/{bt}}}}}"
            r"{{{bt}^{{{al}}}\,\Gamma({al})}}"
        ),
        slots={"al": S(_LAM_POOL), "bt": S(_SIG_POOL)},
    ),
    Template(
        name="gamma_mean",
        latex=(
            r"{op}[{rv}] = {al}\,{bt}"
            r" \quad \bigl(\mathrm{{Gamma}}({al},{bt})\bigr)"
        ),
        slots={
            "op": S(_EXP_OP),
            "rv": S(_RV_POOL),
            "al": S(_LAM_POOL),
            "bt": S(_SIG_POOL),
        },
    ),
    Template(
        name="beta_pdf",
        latex=(
            r"f(x;\,{al},{bt}) ="
            r" \dfrac{{x^{{{al}-1}}(1-x)^{{{bt}-1}}}}{{B({al},{bt})}}"
        ),
        slots={"al": S(_LAM_POOL), "bt": X(_LAM_POOL, ("al",))},
    ),
    Template(
        name="beta_mean",
        latex=(
            r"{op}[{rv}] = \dfrac{{{al}}}{{{al}+{bt}}}"
            r" \quad \bigl(\mathrm{{Beta}}({al},{bt})\bigr)"
        ),
        slots={
            "op": S(_EXP_OP),
            "rv": S(_RV_POOL),
            "al": S(_LAM_POOL),
            "bt": X(_LAM_POOL, ("al",)),
        },
    ),
    Template(
        name="uniform_pdf",
        latex=(
            r"f(x;\,{aa},{bb}) = \dfrac{{1}}{{{bb}-{aa}}},"
            r"\quad {aa} \leq x \leq {bb}"
        ),
        slots={"aa": S(_AB_POOL), "bb": X(_AB_POOL, ("aa",))},
    ),
    Template(
        name="uniform_mean",
        latex=r"{op}[{rv}] = \dfrac{{{aa}+{bb}}}{{2}}",
        slots={
            "op": S(_EXP_OP),
            "rv": S(_RV_POOL),
            "aa": S(_AB_POOL),
            "bb": X(_AB_POOL, ("aa",)),
        },
    ),
    Template(
        name="lognormal_pdf",
        latex=(
            r"f(x;\,{mu},{sig}) = \dfrac{{1}}{{x\,{sig}\sqrt{{2\pi}}}}"
            r" \exp\!\left(-\dfrac{{(\ln x - {mu})^2}}{{2\,{sig}^2}}\right)"
        ),
        slots={"mu": S(_MU_POOL), "sig": S(_SIG_POOL)},
    ),
    Template(
        name="cauchy_pdf",
        latex=(
            r"f(x;\,{mu},{sig}) = \dfrac{{1}}{{\pi\,{sig}"
            r"\!\left[1+\!\left(\dfrac{{x-{mu}}}{{{sig}}}\right)^{{\!2}}\right]}}"
        ),
        slots={"mu": S(_MU_POOL), "sig": S(_SIG_POOL)},
    ),
    # Discrete distributions
    Template(
        name="negative_binomial_pmf",
        latex=(
            r"{op}({rv} = {kk}) = \binom{{{kk}+{rr}-1}}{{{kk}}}"
            r" {pp}^{{{rr}}} (1-{pp})^{{{kk}}}"
        ),
        slots={
            "op": S(_PROB_OP),
            "rv": S(_RV_POOL),
            "kk": S(_K_POOL),
            "rr": X(_K_POOL, ("kk",)),
            "pp": S(_P_POOL),
        },
    ),
    Template(
        name="hypergeometric_pmf",
        latex=(
            r"{op}({rv} = {kk}) ="
            r" \dfrac{{\dbinom{{K}}{{{kk}}}\dbinom{{N-K}}{{{nn}-{kk}}}}}"
            r"{{\dbinom{{N}}{{{nn}}}}}"
        ),
        slots={
            "op": S(_PROB_OP),
            "rv": S(_RV_POOL),
            "kk": S(_K_POOL),
            "nn": S(_N_POOL),
        },
    ),
    Template(
        name="bernoulli_pmf",
        latex=(r"{op}({rv} = {kk}) = {pp}^{{{kk}}}(1-{pp})^{{1-{kk}}}"),
        slots={
            "op": S(_PROB_OP),
            "rv": S(_RV_POOL),
            "kk": S(_BERN_K),
            "pp": S(_P_POOL),
        },
    ),
    Template(
        name="uniform_discrete_pmf",
        latex=(
            r"{op}({rv} = {kk}) = \dfrac{{1}}{{{nn}}},"
            r"\quad {kk} \in \{{1,2,\ldots,{nn}\}}"
        ),
        slots={
            "op": S(_PROB_OP),
            "rv": S(_RV_POOL),
            "kk": S(_K_POOL),
            "nn": S(_N_POOL),
        },
    ),
    # Moment and expectation identities
    Template(
        name="linearity_expectation",
        latex=(
            r"{op}[a\,{rv1} + b\,{rv2}]"
            r" = a\,{op}[{rv1}] + b\,{op}[{rv2}]"
        ),
        slots={
            "op": S(_EXP_OP),
            "rv1": S(_RV_POOL),
            "rv2": X(_RV_POOL, ("rv1",)),
        },
    ),
    Template(
        name="variance_linear",
        latex=r"\operatorname{{Var}}(a\,{rv} + b) = a^2\,\operatorname{{Var}}({rv})",
        slots={"rv": S(_RV_POOL)},
    ),
    Template(
        name="variance_sum",
        latex=(
            r"\operatorname{{Var}}({rv1} + {rv2})"
            r" = \operatorname{{Var}}({rv1}) + \operatorname{{Var}}({rv2})"
            r" + 2\operatorname{{Cov}}({rv1},{rv2})"
        ),
        slots={"rv1": S(_RV_POOL), "rv2": X(_RV_POOL, ("rv1",))},
    ),
    Template(
        name="second_moment_relation",
        latex=(
            r"{op}[{rv}^2]"
            r" = \operatorname{{Var}}({rv}) + \bigl({op}[{rv}]\bigr)^2"
        ),
        slots={"op": S(_EXP_OP), "rv": S(_RV_POOL)},
    ),
    Template(
        name="skewness_def",
        latex=(
            r"\gamma_1({rv}) ="
            r" \dfrac{{{op}\!\left[({rv}-\mu)^3\right]}}{{\sigma^3}}"
        ),
        slots={"op": S(_EXP_OP), "rv": S(_RV_POOL)},
    ),
    Template(
        name="kurtosis_def",
        latex=(
            r"\kappa({rv}) ="
            r" \dfrac{{{op}\!\left[({rv}-\mu)^4\right]}}{{\sigma^4}} - 3"
        ),
        slots={"op": S(_EXP_OP), "rv": S(_RV_POOL)},
    ),
    Template(
        name="variance_independence",
        latex=(
            r"\operatorname{{Var}}({rv1} + {rv2})"
            r" = \operatorname{{Var}}({rv1}) + \operatorname{{Var}}({rv2})"
            r" \quad ({rv1} \perp {rv2})"
        ),
        slots={"rv1": S(_RV_POOL), "rv2": X(_RV_POOL, ("rv1",))},
    ),
    Template(
        name="expectation_product_indep",
        latex=(
            r"{op}[{rv1}\,{rv2}] = {op}[{rv1}]\,{op}[{rv2}]"
            r" \quad ({rv1} \perp {rv2})"
        ),
        slots={
            "op": S(_EXP_OP),
            "rv1": S(_RV_POOL),
            "rv2": X(_RV_POOL, ("rv1",)),
        },
    ),
    # Probability inequalities
    Template(
        name="markov_inequality",
        latex=(
            r"{op}({rv} \geq {aa})"
            r" \leq \dfrac{{{op}[{rv}]}}{{{aa}}}"
        ),
        slots={"op": S(_PROB_OP), "rv": S(_RV_POOL), "aa": S(_A_POOL)},
    ),
    Template(
        name="chebyshev_inequality",
        latex=(
            r"{op}\!\left(|{rv} - {mu}| \geq {aa}\right)"
            r" \leq \dfrac{{\operatorname{{Var}}({rv})}}{{{aa}^2}}"
        ),
        slots={
            "op": S(_PROB_OP),
            "rv": S(_RV_POOL),
            "mu": S(_MU_POOL),
            "aa": S(_A_POOL),
        },
    ),
    Template(
        name="cauchy_schwarz_expectation",
        latex=(
            r"\bigl({op}[{rv1}\,{rv2}]\bigr)^2"
            r" \leq {op}[{rv1}^2]\,{op}[{rv2}^2]"
        ),
        slots={
            "op": S(_EXP_OP),
            "rv1": S(_RV_POOL),
            "rv2": X(_RV_POOL, ("rv1",)),
        },
    ),
    Template(
        name="chernoff_bound",
        latex=(
            r"{op}({rv} \geq {aa})"
            r" \leq \dfrac{{M_{{{rv}}}({lam})}}{{e^{{{lam}\,{aa}}}}}"
        ),
        slots={
            "op": S(_PROB_OP),
            "rv": S(_RV_POOL),
            "aa": S(_A_POOL),
            "lam": S(_LAM_POOL),
        },
    ),
    # Multivariate
    Template(
        name="marginal_pdf_cont",
        latex=(
            r"f_{{{rv1}}}(x)"
            r" = \int{lim_mod}_{{-\infty}}^{{\infty}} f_{{{rv1},{rv2}}}(x,y)\, dy"
        ),
        slots={"lim_mod": _LIM_MOD, "rv1": S(_RV_POOL), "rv2": X(_RV_POOL, ("rv1",))},
    ),
    Template(
        name="conditional_pdf_def",
        latex=(
            r"f_{{{rv2} \mid {rv1}}}(y \mid x)"
            r" = \dfrac{{f_{{{rv1},{rv2}}}(x,y)}}{{f_{{{rv1}}}(x)}}"
        ),
        slots={"rv1": S(_RV_POOL), "rv2": X(_RV_POOL, ("rv1",))},
    ),
    Template(
        name="joint_independence_pdf",
        latex=(r"f_{{{rv1},{rv2}}}(x,y) = f_{{{rv1}}}(x)\,f_{{{rv2}}}(y)"),
        slots={"rv1": S(_RV_POOL), "rv2": X(_RV_POOL, ("rv1",))},
    ),
    Template(
        name="covariance_matrix_entry",
        latex=(
            r"\Sigma_{{{ii}{jj}}}"
            r" = \operatorname{{Cov}}({rv}_{{{ii}}},{rv}_{{{jj}}})"
        ),
        slots={
            "rv": S(_RV_POOL),
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
        },
    ),
    Template(
        name="multivariate_normal",
        latex=(
            r"f(\mathbf{{x}}) = \frac{{1}}{{(2\pi)^{{d/2}}|\Sigma|^{{1/2}}}}"
            r" \exp\!\left(-\tfrac{{1}}{{2}}"
            r"(\mathbf{{x}}-{mu})^\top \Sigma^{{-1}}(\mathbf{{x}}-{mu})\right)"
        ),
        slots={"mu": S(_MU_POOL)},
    ),
    # Generating functions
    Template(
        name="pgf_def",
        latex=(
            r"G_{{{rv}}}(z) = {op}\!\left[z^{{{rv}}}\right]"
            r" = \sum{lim_mod}_{{k=0}}^{{\infty}} {op}({rv} = k)\,z^k"
        ),
        slots={"lim_mod": _LIM_MOD, "op": S(_EXP_OP), "rv": S(_RV_POOL)},
    ),
    Template(
        name="pgf_mean",
        latex=r"{op}[{rv}] = G'_{{{rv}}}(1)",
        slots={"op": S(_EXP_OP), "rv": S(_RV_POOL)},
    ),
    Template(
        name="mgf_derivative_moment",
        latex=r"{op}[{rv}^{{{nn}}}] = M^{{({nn})}}_{{{rv}}}(0)",
        slots={"op": S(_EXP_OP), "rv": S(_RV_POOL), "nn": S(_N_POOL)},
    ),
    Template(
        name="mgf_sum_independent",
        latex=(
            r"M_{{{rv1}+{rv2}}}(t)"
            r" = M_{{{rv1}}}(t)\,M_{{{rv2}}}(t)"
            r" \quad ({rv1} \perp {rv2})"
        ),
        slots={"rv1": S(_RV_POOL), "rv2": X(_RV_POOL, ("rv1",))},
    ),
    Template(
        name="characteristic_fn_inversion",
        latex=(
            r"f_{{{rv}}}(x)"
            r" = \frac{{1}}{{2\pi}}"
            r" \int{lim_mod}_{{-\infty}}^{{\infty}} e^{{-itx}}\,\varphi_{{{rv}}}(t)\, dt"
        ),
        slots={"lim_mod": _LIM_MOD, "rv": S(_RV_POOL)},
    ),
    # Convergence
    Template(
        name="convergence_in_probability",
        latex=r"{rv1}_n \xrightarrow{{p}} {rv2}",
        slots={"rv1": S(_RV_POOL), "rv2": X(_RV_POOL, ("rv1",))},
    ),
    Template(
        name="convergence_in_distribution",
        latex=r"{rv1}_n \xrightarrow{{d}} {rv2}",
        slots={"rv1": S(_RV_POOL), "rv2": X(_RV_POOL, ("rv1",))},
    ),
    Template(
        name="almost_sure_convergence",
        latex=r"{rv1}_n \xrightarrow{{\text{{a.s.}}}} {rv2}",
        slots={"rv1": S(_RV_POOL), "rv2": X(_RV_POOL, ("rv1",))},
    ),
    # ------------------------------------------------------------------
    # Part C: high-n_eff function-pair templates
    # ------------------------------------------------------------------
    Template(
        name="expectation_composition",
        latex=(
            r"{op}[{fn1}({rv})]"
            r" = \int{lim_mod}_{{-\infty}}^{{\infty}} {fn1}(x)\, f_{{{rv}}}(x)\, dx"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "op": S(_EXP_OP),
            "fn1": _FN_SLOT,
            "rv": S(_RV_POOL),
        },
    ),
    Template(
        name="conditional_expectation_product",
        latex=(
            r"{op}[{fn1}({rv1})\,{fn2}({rv2})]"
            r" = {op}[{fn1}({rv1})]\,{op}[{fn2}({rv2})]"
            r" \quad ({rv1} \perp {rv2})"
        ),
        slots={
            "op": S(_EXP_OP),
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "rv1": S(_RV_POOL),
            "rv2": X(_RV_POOL, ("rv1",)),
        },
    ),
    Template(
        name="mgf_composition",
        latex=(
            r"M_{{{fn1}({rv})}}(t)"
            r" = {op}\!\left[e^{{t\,{fn1}({rv})}}\right]"
        ),
        slots={
            "fn1": _FN_SLOT,
            "op": S(_EXP_OP),
            "rv": S(_RV_POOL),
        },
    ),
    Template(
        name="jacobian_change_of_variables",
        latex=(
            r"f_{{{rv2}}}(y)"
            r" = f_{{{rv1}}}\!\left({fn1}(y)\right)\,\bigl|{fn2}(y)\bigr|"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "rv1": S(_RV_POOL),
            "rv2": X(_RV_POOL, ("rv1",)),
        },
    ),
]

# Part C additions
_PROB_TEMPLATES += [
    Template(
        name="fn_cdf_pair",
        latex=r"{fn1}(F_{{{rv}}}({aa})) = {fn2}({op}({rv} \leq {aa}))",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "rv": S(_RV_POOL),
            "aa": S(_A_POOL),
            "op": S(_PROB_OP),
        },
    ),
    Template(
        name="fn_moment_pair",
        latex=r"{fn1}(\mathbb{{E}}[{rv}^{{{kk}}}]) = {fn2}\!\left(\int {aa}^{{{kk}}} f_{{{rv}}}({aa})\,d{aa}\right)",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "rv": S(_RV_POOL),
            "kk": S(_K_POOL),
            "aa": S(_A_POOL),
        },
    ),
    Template(
        name="fn_characteristic_fn_pair",
        latex=r"{fn1}(\varphi_{{{rv}}}({tt})) = {fn2}\!\left(\mathbb{{E}}[e^{{i{tt} {rv}}}]\right)",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "rv": S(_RV_POOL),
            "tt": S(_A_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Sampling weights
# ---------------------------------------------------------------------------

_PROB_TEMPLATES += [
    Template(
        name="poisson_binomial_approx",
        latex=r"\operatorname{{Bin}}({nn},\,{pp}) \approx \operatorname{{Poisson}}({nn}\,{pp}) \quad ({nn}\text{{ large, }}{pp}\text{{ small}})",
        slots={"nn": S(_N_POOL), "pp": S(_P_POOL)},
    ),
    Template(
        name="stirling_approx",
        latex=r"{nn}!\approx\sqrt{{2\pi\,{nn}}}\left(\frac{{{nn}}}{{e}}\right)^{{{nn}}}",
        slots={"nn": S(_N_POOL)},
    ),
    Template(
        name="normal_cdf_approx",
        latex=r"{op}\!\left(\frac{{{rv}-{mu}}}{{{sig}}} \leq {aa}\right) \approx \Phi({aa})",
        slots={"op": S(_PROB_OP), "rv": S(_RV_POOL), "mu": S(_MU_POOL), "sig": S(_SIG_POOL), "aa": S(_A_POOL)},
    ),
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("probability", _PROB_TEMPLATES)
