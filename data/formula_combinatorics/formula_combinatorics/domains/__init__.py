"""Domain registry: merges GENERATORS, WEIGHTS, and TEMPLATES from all domain modules."""

from __future__ import annotations

import importlib
import random
from collections.abc import Callable

from .._template_dsl import Template
from ._config import DOMAIN_CONFIG

_DOMAIN_NAMES: list[str] = [
    "algebra",
    "trigonometry",
    "calculus",
    "category_theory",
    "analysis",
    "differential_equations",
    "linear_algebra",
    "probability",
    "information_theory",
    "number_theory",
    "combinatorics",
    "graph_theory",
    "group_theory",
    "ring_field_theory",
    "representation_theory",
    "differential_geometry",
    "topology",
    "complex_analysis",
    "fourier",
    "physics",
    "chemistry",
    "quantum_notation",
    "measure_theory",
    "p_adic",
    "optimization",
    "set_theory",
    "logic",
    "statistics",
    "stochastic_processes",
    "custom_operators",
    "math_fonts",
    "geometry",
    "align",
]

GENERATORS: dict[str, Callable[[random.Random], str]] = {}
DEFAULT_WEIGHTS: dict[str, float] = {}
TEMPLATES: dict[str, list[Template]] = {}

for _name in _DOMAIN_NAMES:
    _mod = importlib.import_module(f".{_name}", package=__package__)
    GENERATORS.update(_mod.GENERATORS)
    DEFAULT_WEIGHTS.update(_mod.WEIGHTS)
    TEMPLATES.update(_mod.TEMPLATES)

assert set(DEFAULT_WEIGHTS) == set(GENERATORS), (
    f"Weight/generator mismatch: "
    f"extra weights={set(DEFAULT_WEIGHTS) - set(GENERATORS)}, "
    f"missing weights={set(GENERATORS) - set(DEFAULT_WEIGHTS)}"
)

# Tag index: domain name → list of tag strings (usable for corpus.generate(tags=...) filtering)
DOMAIN_TAGS: dict[str, list[str]] = {k: list(v.tags) for k, v in DOMAIN_CONFIG.items()}

__all__ = ["GENERATORS", "DEFAULT_WEIGHTS", "TEMPLATES", "DOMAIN_TAGS", "DOMAIN_CONFIG"]
