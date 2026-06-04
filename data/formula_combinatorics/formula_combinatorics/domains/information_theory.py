"""Information theory domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import S, Template, X, compute_weights, make_dispatcher

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_RV_POOL: list[str] = ["X", "Y", "Z"]

# ---------------------------------------------------------------------------
# Information theory templates
# ---------------------------------------------------------------------------

_INFOTH_TEMPLATES: list[Template] = [
    Template(
        name="entropy",
        latex=r"H({X}) = -\sum_x p(x) \log p(x)",
        slots={"X": S(_RV_POOL)},
    ),
    Template(
        name="joint_entropy",
        latex=r"H({X}, {Y}) = -\sum_{{x,y}} p(x,y) \log p(x,y)",
        slots={"X": S(_RV_POOL), "Y": X(_RV_POOL, ["X"])},
    ),
    Template(
        name="conditional_entropy",
        latex=r"H({Y} \mid {X}) = H({X}, {Y}) - H({X})",
        slots={"X": S(_RV_POOL), "Y": X(_RV_POOL, ["X"])},
    ),
    Template(
        name="mutual_information",
        latex=r"I({X}; {Y}) = H({X}) + H({Y}) - H({X}, {Y})",
        slots={"X": S(_RV_POOL), "Y": X(_RV_POOL, ["X"])},
    ),
    Template(
        name="kl_divergence",
        latex=r"D_{{KL}}(P \| Q) = \sum_x p(x) \log \frac{{p(x)}}{{q(x)}}",
        slots={},
    ),
    Template(
        name="channel_capacity",
        latex=r"C = \max_{{p(x)}} I({X}; {Y})",
        slots={"X": S(_RV_POOL), "Y": X(_RV_POOL, ["X"])},
    ),
    Template(
        name="entropy_bound",
        latex=r"H({X}) \leq \log |\mathcal{{{X}}}|",
        slots={"X": S(_RV_POOL)},
    ),
    Template(
        name="entropy_subadditivity",
        latex=r"H({X}_1, \ldots, {X}_n) \leq \sum_{{i=1}}^n H({X}_i)",
        slots={"X": S(_RV_POOL)},
    ),
    Template(
        name="mutual_info_nonneg",
        latex=r"I({X}; {Y}) \geq 0",
        slots={"X": S(_RV_POOL), "Y": X(_RV_POOL, ["X"])},
    ),
    Template(
        name="cross_entropy",
        latex=r"H(p, q) = -\sum_x p(x) \log q(x)",
        slots={},
    ),
    Template(
        name="rate_distortion",
        latex=r"R(D) = \min_{{p(\hat{{x}} \mid x) : E[d(X,\hat{{X}})] \leq D}} I({X}; \hat{{{X}}})",
        slots={"X": S(_RV_POOL)},
    ),
    Template(
        name="fano_inequality",
        latex=r"H(P_e) + P_e \log(|\mathcal{{{X}}}| - 1) \geq H({X} \mid \hat{{{X}}})",
        slots={"X": S(_RV_POOL)},
    ),
    Template(
        name="data_processing_inequality",
        latex=r"{X} \to {Y} \to Z \implies I({X}; Z) \leq I({X}; {Y})",
        slots={"X": S(_RV_POOL), "Y": X(_RV_POOL, ["X"])},
    ),
    Template(
        name="chain_rule_entropy",
        latex=r"H({X}_1, \ldots, {X}_n) = \sum_{{i=1}}^n H({X}_i \mid {X}_1, \ldots, {X}_{{i-1}})",
        slots={"X": S(_RV_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Sampling weights
# ---------------------------------------------------------------------------

_W_INFO: list[float] = compute_weights(_INFOTH_TEMPLATES)

# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

_information_theory = make_dispatcher(_INFOTH_TEMPLATES, _W_INFO)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "information_theory": _information_theory,
}

WEIGHTS: dict[str, float] = {
    "information_theory": 0.03,
}

TEMPLATES: dict[str, list[Template]] = {
    "information_theory": _INFOTH_TEMPLATES,
}
