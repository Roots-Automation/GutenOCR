"""Central configuration for domain sampling weights and n_eff caps.

All domain modules call ``register_domain(name, templates)`` without weight or
cap arguments; this module resolves both from ``DOMAIN_CONFIG``.  Edit weights
here to change the corpus distribution — no domain file needs to be touched.

Tags (optional, informational):
    foundational  — core undergraduate-level topics
    advanced      — graduate / research-level topics
    applied       — physics, chemistry, engineering mathematics
    structural    — notational / typographic coverage
"""

from __future__ import annotations

from typing import Any

from .._template_dsl import Template
from .._template_dsl import register_domain as _register_domain

# ---------------------------------------------------------------------------
# Per-domain configuration
# ---------------------------------------------------------------------------
# weight  — unnormalised sampling probability (registry normalises at load time)
# cap     — n_eff cap passed to compute_weights; default 1_000_000
# tags    — informational only; not used at runtime
# ---------------------------------------------------------------------------

DOMAIN_CONFIG: dict[str, dict[str, Any]] = {
    # ── foundational ─────────────────────────────────────────────────────────
    "algebra": {"weight": 0.09, "cap": 75_000_000, "tags": ["foundational"]},
    "trigonometry": {"weight": 0.04, "cap": 75_000_000, "tags": ["foundational"]},
    "calculus": {"weight": 0.10, "tags": ["foundational"]},
    "linear_algebra": {"weight": 0.08, "tags": ["foundational"]},
    "geometry": {"weight": 0.07, "tags": ["foundational"]},
    "probability": {"weight": 0.07, "tags": ["foundational"]},
    # ── applied ──────────────────────────────────────────────────────────────
    "physics": {"weight": 0.06, "tags": ["applied"]},
    "chemistry": {"weight": 0.05, "tags": ["applied"]},
    "optimization": {"weight": 0.05, "tags": ["applied"]},
    # ── core mathematics ─────────────────────────────────────────────────────
    "set_theory": {"weight": 0.05, "tags": ["foundational"]},
    "logic": {"weight": 0.05, "tags": ["foundational"]},
    "group_theory": {"weight": 0.04, "tags": ["advanced"]},
    "analysis": {"weight": 0.04, "tags": ["advanced"]},
    "number_theory": {"weight": 0.04, "tags": ["advanced"]},
    "quantum_notation": {"weight": 0.04, "tags": ["applied"]},
    "statistics": {"weight": 0.04, "tags": ["foundational"]},
    "topology": {"weight": 0.04, "tags": ["advanced"]},
    "complex_analysis": {"weight": 0.03, "tags": ["advanced"]},
    "combinatorics": {"weight": 0.03, "tags": ["foundational"]},
    "custom_operators": {"weight": 0.03, "tags": ["structural"]},
    "differential_equations": {"weight": 0.03, "tags": ["foundational"]},
    "information_theory": {"weight": 0.03, "tags": ["applied"]},
    "math_fonts": {"weight": 0.03, "tags": ["structural"]},
    # ── advanced / specialised ────────────────────────────────────────────────
    "category_theory": {"weight": 0.02, "tags": ["advanced"]},
    "differential_geometry": {"weight": 0.02, "tags": ["advanced"]},
    "fourier": {"weight": 0.02, "tags": ["advanced"]},
    "graph_theory": {"weight": 0.02, "tags": ["advanced"]},
    "measure_theory": {"weight": 0.02, "tags": ["advanced"]},
    "p_adic": {"weight": 0.02, "tags": ["advanced"]},
    "stochastic_processes": {"weight": 0.02, "tags": ["advanced"]},
    "representation_theory": {"weight": 0.01, "tags": ["advanced"]},
    "ring_field_theory": {"weight": 0.01, "tags": ["advanced"]},
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
    weight: float = cfg["weight"]
    cap: float = cfg.get("cap", 1_000_000)
    return _register_domain(name, templates, weight, cap=cap)
