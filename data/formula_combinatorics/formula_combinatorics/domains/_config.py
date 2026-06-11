"""Central configuration for domain sampling weights and n_eff caps.

All domain modules call ``register_domain(name, templates)`` without weight or
cap arguments; this module resolves both from ``DOMAIN_CONFIG``.  Edit weights
here to change the corpus distribution — no domain file needs to be touched.

Tags:
    foundational  — core undergraduate-level topics
    advanced      — graduate / research-level topics
    applied       — physics, chemistry, engineering mathematics
    structural    — notational / typographic coverage
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .._template_dsl import Template
from .._template_dsl import register_domain as _register_domain


@dataclass(frozen=True)
class DomainMeta:
    """Sampling configuration and metadata for one domain."""

    weight: float
    cap: float = 1_000_000
    tags: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""


# ---------------------------------------------------------------------------
# Per-domain configuration
# ---------------------------------------------------------------------------
# weight  — unnormalised sampling probability (registry normalises at load time)
# cap     — n_eff cap passed to compute_weights; default 1_000_000
# tags    — used at runtime for corpus.generate(tags=...) filtering
# ---------------------------------------------------------------------------

DOMAIN_CONFIG: dict[str, DomainMeta] = {
    # ── foundational ─────────────────────────────────────────────────────────
    "algebra": DomainMeta(weight=0.09, cap=75_000_000, tags=("foundational",)),
    "trigonometry": DomainMeta(weight=0.04, cap=75_000_000, tags=("foundational",)),
    "calculus": DomainMeta(weight=0.10, tags=("foundational",)),
    "linear_algebra": DomainMeta(weight=0.08, tags=("foundational",)),
    "geometry": DomainMeta(weight=0.07, tags=("foundational",)),
    "probability": DomainMeta(weight=0.07, tags=("foundational",)),
    # ── applied ──────────────────────────────────────────────────────────────
    "physics": DomainMeta(weight=0.06, tags=("applied",)),
    "chemistry": DomainMeta(weight=0.05, tags=("applied",)),
    "optimization": DomainMeta(weight=0.05, tags=("applied",)),
    # ── core mathematics ─────────────────────────────────────────────────────
    "set_theory": DomainMeta(weight=0.05, tags=("foundational",)),
    "logic": DomainMeta(weight=0.05, tags=("foundational",)),
    "group_theory": DomainMeta(weight=0.04, tags=("advanced",)),
    "analysis": DomainMeta(weight=0.04, tags=("advanced",)),
    "number_theory": DomainMeta(weight=0.04, tags=("advanced",)),
    "quantum_notation": DomainMeta(weight=0.04, tags=("applied",)),
    "statistics": DomainMeta(weight=0.04, tags=("foundational",)),
    "topology": DomainMeta(weight=0.04, tags=("advanced",)),
    "complex_analysis": DomainMeta(weight=0.03, tags=("advanced",)),
    "combinatorics": DomainMeta(weight=0.03, tags=("foundational",)),
    "custom_operators": DomainMeta(weight=0.03, tags=("structural",)),
    "differential_equations": DomainMeta(weight=0.03, tags=("foundational",)),
    "information_theory": DomainMeta(weight=0.03, tags=("applied",)),
    "math_fonts": DomainMeta(weight=0.03, tags=("structural",)),
    # ── advanced / specialised ────────────────────────────────────────────────
    "category_theory": DomainMeta(weight=0.02, tags=("advanced",)),
    "differential_geometry": DomainMeta(weight=0.02, tags=("advanced",)),
    "fourier": DomainMeta(weight=0.02, tags=("advanced",)),
    "graph_theory": DomainMeta(weight=0.02, tags=("advanced",)),
    "measure_theory": DomainMeta(weight=0.02, tags=("advanced",)),
    "p_adic": DomainMeta(weight=0.02, tags=("advanced",)),
    "stochastic_processes": DomainMeta(weight=0.02, tags=("advanced",)),
    "representation_theory": DomainMeta(weight=0.01, tags=("advanced",)),
    "ring_field_theory": DomainMeta(weight=0.01, tags=("advanced",)),
    # ── structural / formatting ───────────────────────────────────────────────
    "align": DomainMeta(weight=0.15, cap=5_000_000, tags=("structural",)),
}


# ---------------------------------------------------------------------------
# Domain-level register_domain wrapper
# ---------------------------------------------------------------------------


def register_domain(
    name: str,
    templates: list[Template],
) -> tuple[dict, dict, dict]:
    """Register a domain, pulling weight and cap from ``DOMAIN_CONFIG``.

    Drop-in replacement for ``_template_dsl.register_domain`` in domain
    modules — callers no longer pass weight or cap::

        GENERATORS, WEIGHTS, TEMPLATES = register_domain("algebra", _TEMPLATES)
    """
    cfg = DOMAIN_CONFIG[name]
    return _register_domain(name, templates, cfg.weight, cap=cfg.cap)
