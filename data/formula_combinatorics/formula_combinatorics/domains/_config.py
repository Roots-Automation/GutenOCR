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
    difficulty: str = "undergraduate"  # elementary | undergraduate | graduate | research


# ---------------------------------------------------------------------------
# Per-domain configuration
# ---------------------------------------------------------------------------
# weight  — unnormalised sampling probability (registry normalises at load time)
# cap     — n_eff cap passed to compute_weights; default 1_000_000
# tags    — used at runtime for corpus.generate(tags=...) filtering
# ---------------------------------------------------------------------------

DOMAIN_CONFIG: dict[str, DomainMeta] = {
    # ── foundational ─────────────────────────────────────────────────────────
    "algebra": DomainMeta(weight=0.09, cap=75_000_000, tags=("foundational",), difficulty="elementary"),
    "trigonometry": DomainMeta(weight=0.04, cap=75_000_000, tags=("foundational",), difficulty="elementary"),
    "calculus": DomainMeta(weight=0.10, tags=("foundational",), difficulty="undergraduate"),
    "linear_algebra": DomainMeta(weight=0.08, tags=("foundational",), difficulty="undergraduate"),
    "geometry": DomainMeta(weight=0.07, tags=("foundational",), difficulty="elementary"),
    "probability": DomainMeta(weight=0.07, tags=("foundational",), difficulty="undergraduate"),
    # ── applied ──────────────────────────────────────────────────────────────
    "physics": DomainMeta(weight=0.06, tags=("applied",), difficulty="undergraduate"),
    "chemistry": DomainMeta(weight=0.05, tags=("applied",), difficulty="undergraduate"),
    "optimization": DomainMeta(weight=0.05, tags=("applied",), difficulty="undergraduate"),
    # ── core mathematics ─────────────────────────────────────────────────────
    "set_theory": DomainMeta(weight=0.05, tags=("foundational",), difficulty="undergraduate"),
    "logic": DomainMeta(weight=0.05, tags=("foundational",), difficulty="undergraduate"),
    "group_theory": DomainMeta(weight=0.04, tags=("advanced",), difficulty="graduate"),
    "analysis": DomainMeta(weight=0.04, tags=("advanced",), difficulty="graduate"),
    "number_theory": DomainMeta(weight=0.04, tags=("advanced",), difficulty="graduate"),
    "quantum_notation": DomainMeta(weight=0.04, tags=("applied",), difficulty="graduate"),
    "statistics": DomainMeta(weight=0.04, tags=("foundational",), difficulty="undergraduate"),
    "topology": DomainMeta(weight=0.04, tags=("advanced",), difficulty="graduate"),
    "complex_analysis": DomainMeta(weight=0.03, tags=("advanced",), difficulty="graduate"),
    "combinatorics": DomainMeta(weight=0.03, tags=("foundational",), difficulty="undergraduate"),
    "custom_operators": DomainMeta(weight=0.03, tags=("structural",), difficulty="graduate"),
    "differential_equations": DomainMeta(weight=0.03, tags=("foundational",), difficulty="undergraduate"),
    "information_theory": DomainMeta(weight=0.03, tags=("applied",), difficulty="graduate"),
    "math_fonts": DomainMeta(weight=0.03, tags=("structural",), difficulty="undergraduate"),
    # ── advanced / specialised ────────────────────────────────────────────────
    "category_theory": DomainMeta(weight=0.02, tags=("advanced",), difficulty="research"),
    "differential_geometry": DomainMeta(weight=0.02, tags=("advanced",), difficulty="graduate"),
    "fourier": DomainMeta(weight=0.02, tags=("advanced",), difficulty="graduate"),
    "graph_theory": DomainMeta(weight=0.02, tags=("advanced",), difficulty="graduate"),
    "measure_theory": DomainMeta(weight=0.02, tags=("advanced",), difficulty="graduate"),
    "p_adic": DomainMeta(weight=0.02, tags=("advanced",), difficulty="research"),
    "stochastic_processes": DomainMeta(weight=0.02, tags=("advanced",), difficulty="graduate"),
    "representation_theory": DomainMeta(weight=0.01, tags=("advanced",), difficulty="research"),
    "ring_field_theory": DomainMeta(weight=0.01, tags=("advanced",), difficulty="graduate"),
    # ── structural / formatting ───────────────────────────────────────────────
    "align": DomainMeta(weight=0.15, cap=5_000_000, tags=("structural",), difficulty="undergraduate"),
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
