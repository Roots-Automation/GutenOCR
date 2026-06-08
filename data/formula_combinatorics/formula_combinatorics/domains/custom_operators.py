"""Custom-operator domain: templates featuring \\DeclareMathOperator-style operators.

Covers: \\operatorname{sgn}, \\operatorname{supp}, \\operatorname{diag},
\\operatorname{ess\\,sup}/\\operatorname{ess\\,inf}, \\operatorname*{colim},
and international/regional trig-name variants.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import E, S, Template, X, compute_weights, make_dispatcher
from .._vocab import (
    _CALLIGRAPHIC,
    _GEO_N,
    _SCALARS,
    _VARS,
    _fn_rich_nosub,
)

# ---------------------------------------------------------------------------
# Shared pools
# ---------------------------------------------------------------------------

_VAR_POOL: tuple[str, ...] = tuple(_VARS)
_COEFF: tuple[str, ...] = tuple(_SCALARS)
_POLY_POOL = ("p", "q", "f", "g", "h")
_MAT_POOL = ("A", "B", "M", "P", "Q", "S")
_N_POOL: tuple[str, ...] = tuple(_GEO_N)

# ---------------------------------------------------------------------------
# Section A — Sign function (\\operatorname{sgn})
# ---------------------------------------------------------------------------

_SGN_TEMPLATES: list[Template] = [
    Template(
        name="sgn_piecewise",
        latex=(
            r"\operatorname{{sgn}}({vv}) = "
            r"\begin{{cases}} 1 & {vv} > 0 \\ 0 & {vv} = 0 \\ -1 & {vv} < 0 \end{{cases}}"
        ),
        slots={"vv": S(_VAR_POOL)},
    ),
    Template(
        name="sgn_multiplicative",
        latex=(
            r"\operatorname{{sgn}}({aa}{bb}) = "
            r"\operatorname{{sgn}}({aa})\,\operatorname{{sgn}}({bb})"
        ),
        slots={
            "aa": S(_VAR_POOL, idx=0.3),
            "bb": X(_VAR_POOL, ("aa",), idx=0.3),
        },
    ),
    Template(
        name="sgn_abs_value",
        latex=r"{vv} = \operatorname{{sgn}}({vv})\,|{vv}|",
        slots={"vv": S(_VAR_POOL)},
    ),
    Template(
        name="sgn_heaviside",
        latex=r"\operatorname{{sgn}}({vv}) = 2H({vv}) - 1",
        slots={"vv": S(_VAR_POOL)},
    ),
    Template(
        name="sgn_distributional_deriv",
        latex=r"\frac{{d}}{{d{vv}}}\operatorname{{sgn}}({vv}) = 2\delta({vv})",
        slots={"vv": S(_VAR_POOL)},
    ),
    Template(
        name="sgn_squared",
        latex=r"\operatorname{{sgn}}({vv})^2 = 1 \quad ({vv} \neq 0)",
        slots={"vv": S(_VAR_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Section B — Support (\\operatorname{supp})
# ---------------------------------------------------------------------------

_SUPP_TEMPLATES: list[Template] = [
    Template(
        name="supp_definition",
        latex=r"\operatorname{{supp}}({ff}) = \overline{{\{{{vv} : {ff}({vv}) \neq 0\}}}}",
        slots={
            "ff": E(_fn_rich_nosub, n=100),
            "vv": S(_VAR_POOL),
        },
    ),
    Template(
        name="supp_compact",
        latex=r"\operatorname{{supp}}({ff}) \subset [{aa}, {bb}]",
        slots={
            "ff": E(_fn_rich_nosub, n=100),
            "aa": S(_COEFF, idx=0.2),
            "bb": X(_COEFF, ("aa",), idx=0.2),
        },
    ),
    Template(
        name="supp_intersection",
        latex=(
            r"\operatorname{{supp}}({ff} \cdot {gg}) "
            r"\subseteq \operatorname{{supp}}({ff}) \cap \operatorname{{supp}}({gg})"
        ),
        slots={
            "ff": E(_fn_rich_nosub, n=100),
            "gg": E(_fn_rich_nosub, n=100),
        },
    ),
    Template(
        name="supp_measure",
        latex=(
            r"\operatorname{{supp}}(\mu) = "
            r"\{{{vv} : \mu(U) > 0 \;\forall \text{{ open }} U \ni {vv}\}}"
        ),
        slots={"vv": S(_VAR_POOL)},
    ),
    Template(
        name="supp_union_bound",
        latex=(
            r"\operatorname{{supp}}({ff} + {gg}) "
            r"\subseteq \operatorname{{supp}}({ff}) \cup \operatorname{{supp}}({gg})"
        ),
        slots={
            "ff": E(_fn_rich_nosub, n=100),
            "gg": E(_fn_rich_nosub, n=100),
        },
    ),
]

# ---------------------------------------------------------------------------
# Section C — Diagonal operator (\\operatorname{diag})
# ---------------------------------------------------------------------------

_DIAG_TEMPLATES: list[Template] = [
    Template(
        name="diag_construction",
        latex=(
            r"\operatorname{{diag}}({aa}_1, \ldots, {aa}_{{{nn}}})"
            r" \in \mathbb{{R}}^{{{nn} \times {nn}}}"
        ),
        slots={
            "aa": S(_COEFF),
            "nn": S(_N_POOL),
        },
        distinct=[["aa", "nn"]],
    ),
    Template(
        name="diag_eigendecomp",
        latex=(
            r"{MM} = {PP}\,\operatorname{{diag}}"
            r"(\lambda_1, \ldots, \lambda_{{{nn}}})\,{PP}^{{-1}}"
        ),
        slots={
            "MM": S(_MAT_POOL),
            "PP": X(_MAT_POOL, ("MM",)),
            "nn": S(_N_POOL),
        },
    ),
    Template(
        name="diag_block",
        latex=r"\operatorname{{diag}}({AA}, {BB})",
        slots={
            "AA": S(_MAT_POOL),
            "BB": X(_MAT_POOL, ("AA",)),
        },
    ),
    Template(
        name="diag_trace",
        latex=(
            r"\operatorname{{tr}}({MM}) = "
            r"\operatorname{{tr}}\!\left(\operatorname{{diag}}"
            r"(\lambda_1, \ldots, \lambda_{{{nn}}})\right)"
        ),
        slots={
            "MM": S(_MAT_POOL),
            "nn": S(_N_POOL),
        },
        distinct=[["MM", "nn"]],
    ),
    Template(
        name="diag_det",
        latex=(
            r"\det\!\left(\operatorname{{diag}}"
            r"({aa}_1, \ldots, {aa}_{{{nn}}})\right)"
            r" = \prod_{{i=1}}^{{{nn}}} {aa}_i"
        ),
        slots={
            "aa": S(_COEFF),
            "nn": S(_N_POOL),
        },
        distinct=[["aa", "nn"]],
    ),
]

# ---------------------------------------------------------------------------
# Section D — Essential sup / inf (\\operatorname{ess\\,sup}, \\operatorname{ess\\,inf})
# ---------------------------------------------------------------------------

_ESS_TEMPLATES: list[Template] = [
    Template(
        name="esssup_definition",
        latex=(
            r"\operatorname{{ess\,sup}}_{{{vv} \in {DD}}}\, {ff}({vv})"
            r" = \inf\{{M : {ff}({vv}) \leq M \text{{ a.e.}}\}}"
        ),
        slots={
            "ff": E(_fn_rich_nosub, n=100),
            "vv": S(_VAR_POOL),
            "DD": S(tuple(_CALLIGRAPHIC)),
        },
    ),
    Template(
        name="esssup_Linfty",
        latex=r"\|{ff}\|_\infty = \operatorname{{ess\,sup}}_{{{vv}}}\,|{ff}({vv})|",
        slots={
            "ff": E(_fn_rich_nosub, n=100),
            "vv": S(_VAR_POOL),
        },
    ),
    Template(
        name="essinf_definition",
        latex=(
            r"\operatorname{{ess\,inf}}_{{{vv} \in {DD}}}\, {ff}({vv})"
            r" = \sup\{{m : {ff}({vv}) \geq m \text{{ a.e.}}\}}"
        ),
        slots={
            "ff": E(_fn_rich_nosub, n=100),
            "vv": S(_VAR_POOL),
            "DD": S(tuple(_CALLIGRAPHIC)),
        },
    ),
    Template(
        name="essinf_bound",
        latex=(
            r"\operatorname{{ess\,inf}}_{{{vv}}}\, {ff}({vv})"
            r" \leq \operatorname{{ess\,sup}}_{{{vv}}}\, {ff}({vv})"
        ),
        slots={
            "ff": E(_fn_rich_nosub, n=100),
            "vv": S(_VAR_POOL),
        },
    ),
    Template(
        name="esssup_set_form",
        latex=(
            r"\operatorname{{ess\,sup}}_{{{vv}}}\, {ff}({vv})"
            r" = \inf\{{M \geq 0 : \mu(\{{{vv} : {ff}({vv}) > M\}}) = 0\}}"
        ),
        slots={
            "ff": E(_fn_rich_nosub, n=100),
            "vv": S(_VAR_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Section E — Category-theory colimit (\\operatorname*{colim})
# ---------------------------------------------------------------------------

_COLIM_INDEX_POOL = (
    r"\mathcal{I}",
    r"\mathcal{J}",
    r"\mathcal{C}",
    r"\mathcal{D}",
    r"\mathbf{I}",
)

_COLIM_TEMPLATES: list[Template] = [
    Template(
        name="colim_basic",
        latex=r"\operatorname*{{colim}}_{{{ii} \in {II}}} {FF}({ii})",
        slots={
            "ii": S(("i", "j", "k", "\\alpha")),
            "II": S(_COLIM_INDEX_POOL),
            "FF": S(("F", "G", "H", r"\mathcal{F}", r"\mathcal{G}")),
        },
    ),
    Template(
        name="colim_filtered",
        latex=r"\operatorname*{{colim}}_{{n \geq 0}}\, {FF}_n",
        slots={
            "FF": S(("F", "G", "A", r"\mathcal{F}", r"\mathcal{A}")),
        },
    ),
    Template(
        name="colim_vs_lim",
        latex=(
            r"\operatorname*{{colim}}_{{n}}\, {FF}_n"
            r"\quad \text{{is dual to}} \quad"
            r"\varprojlim_{{n}}\, {FF}_n"
        ),
        slots={
            "FF": S(("F", "G", "A", r"\mathcal{F}")),
        },
    ),
    Template(
        name="colim_limits_variant",
        latex=r"\operatorname*{{colim}}\limits_{{{ii} \in {II}}} {FF}({ii})",
        slots={
            "ii": S(("i", "j", "k")),
            "II": S(_COLIM_INDEX_POOL),
            "FF": S(("F", "G", "H", r"\mathcal{F}")),
        },
    ),
]

# ---------------------------------------------------------------------------
# Section F — International / regional trig-name variants
# ---------------------------------------------------------------------------

_INTL_ARG_POOL: tuple[str, ...] = (
    r"\theta",
    r"\phi",
    r"\varphi",
    r"\alpha",
    r"\beta",
    r"\gamma",
    "x",
    "y",
    "t",
)

_INTL_TEMPLATES: list[Template] = [
    Template(
        name="intl_tg_equals_tan",
        latex=r"\operatorname{{tg}}({vv}) = \tan({vv})",
        slots={"vv": S(_INTL_ARG_POOL, idx=0.2)},
    ),
    Template(
        name="intl_arctg_equals_arctan",
        latex=r"\operatorname{{arctg}}({vv}) = \arctan({vv})",
        slots={"vv": S(_INTL_ARG_POOL, idx=0.2)},
    ),
    Template(
        name="intl_sh_equals_sinh",
        latex=r"\operatorname{{sh}}({vv}) = \sinh({vv})",
        slots={"vv": S(tuple(_VARS), idx=0.2)},
    ),
    Template(
        name="intl_ch_equals_cosh",
        latex=r"\operatorname{{ch}}({vv}) = \cosh({vv})",
        slots={"vv": S(tuple(_VARS), idx=0.2)},
    ),
    Template(
        name="intl_th_equals_tanh",
        latex=r"\operatorname{{th}}({vv}) = \tanh({vv})",
        slots={"vv": S(tuple(_VARS), idx=0.2)},
    ),
    Template(
        name="intl_arctg_value",
        latex=r"\operatorname{{arctg}}\frac{{\pi}}{{{nn}}} = {vv}",
        slots={
            "nn": S(("3", "4", "6")),
            "vv": S((r"\frac{\sqrt{3}}{3}", r"1", r"\frac{1}{\sqrt{3}}")),
        },
    ),
    Template(
        name="intl_ctg_equals_cot",
        latex=r"\operatorname{{ctg}}({vv}) = \cot({vv})",
        slots={"vv": S(_INTL_ARG_POOL, idx=0.2)},
    ),
    Template(
        name="intl_tg_formula",
        latex=(
            r"\operatorname{{tg}}({aa} + {bb}) = "
            r"\frac{{\operatorname{{tg}}{aa} + \operatorname{{tg}}{bb}}}"
            r"{{1 - \operatorname{{tg}}{aa}\,\operatorname{{tg}}{bb}}}"
        ),
        slots={
            "aa": S(_INTL_ARG_POOL, idx=0.3),
            "bb": X(_INTL_ARG_POOL, ("aa",), idx=0.3),
        },
    ),
]

# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

_CUSTOM_OP_TEMPLATES: list[Template] = (
    _SGN_TEMPLATES + _SUPP_TEMPLATES + _DIAG_TEMPLATES + _ESS_TEMPLATES + _COLIM_TEMPLATES + _INTL_TEMPLATES
)

_W = compute_weights(_CUSTOM_OP_TEMPLATES)

_custom_operators = make_dispatcher(_CUSTOM_OP_TEMPLATES, _W)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "custom_operators": _custom_operators,
}

WEIGHTS: dict[str, float] = {
    "custom_operators": 0.03,
}

TEMPLATES: dict[str, list[Template]] = {
    "custom_operators": _CUSTOM_OP_TEMPLATES,
}
