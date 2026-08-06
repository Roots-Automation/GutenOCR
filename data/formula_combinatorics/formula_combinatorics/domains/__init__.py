"""Domain registry: loads GENERATORS, WEIGHTS, and TEMPLATES from TOML packs.

Every domain must have a ``<domain>.toml`` pack file alongside this module.
The Python sub-module fallback has been removed; all domains are TOML-only.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

from ..engine._pack_loader import PackMeta, load_pack
from ..engine._template_dsl import Template
from ._config import DOMAIN_CONFIG, register_domain

_DOMAIN_NAMES: list[str] = [
    "algebra",
    "trigonometry",
    "calculus",
    "category_theory",
    "analysis",
    "asymptotics",
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
    "classical_mechanics",
    "electromagnetism",
    "statistical_mechanics",
    "quantum_mechanics",
    "field_theory",
    "chemistry",
    "quantum_notation",
    "measure_theory",
    "p_adic",
    "optimization",
    "set_theory",
    "logic",
    "proof_theory",
    "statistics",
    "stochastic_processes",
    "custom_operators",
    "math_fonts",
    "geometry",
    "align",
]

_DOMAINS_DIR = Path(__file__).parent

GENERATORS: dict[str, Callable[[random.Random], str]] = {}
DEFAULT_WEIGHTS: dict[str, float] = {}
TEMPLATES: dict[str, list[Template]] = {}

# Map domain name → pack SHA-256 (populated only for TOML-loaded domains)
PACK_HASHES: dict[str, str] = {}
# Map domain name → full PackMeta (populated only for TOML-loaded domains)
PACK_META: dict[str, PackMeta] = {}

for _name in _DOMAIN_NAMES:
    _toml_path = _DOMAINS_DIR / f"{_name}.toml"
    if not _toml_path.exists():
        raise FileNotFoundError(
            f"Domain '{_name}' has no TOML pack at {_toml_path}. All domains must be migrated to TOML."
        )
    _pack = load_pack(_toml_path)
    _gens, _wts, _tmpls = register_domain(_name, _pack.templates)
    PACK_HASHES[_name] = _pack.meta.sha256
    PACK_META[_name] = _pack.meta
    GENERATORS.update(_gens)
    DEFAULT_WEIGHTS.update(_wts)
    TEMPLATES.update(_tmpls)

assert set(DEFAULT_WEIGHTS) == set(GENERATORS), (
    f"Weight/generator mismatch: "
    f"extra weights={set(DEFAULT_WEIGHTS) - set(GENERATORS)}, "
    f"missing weights={set(GENERATORS) - set(DEFAULT_WEIGHTS)}"
)

# Tag index: domain name → list of tag strings (usable for corpus.generate(tags=...) filtering)
DOMAIN_TAGS: dict[str, list[str]] = {k: list(v.tags) for k, v in DOMAIN_CONFIG.items()}

# Difficulty index: domain name → difficulty level string
DOMAIN_DIFFICULTY: dict[str, str] = {k: v.difficulty for k, v in DOMAIN_CONFIG.items()}

__all__ = [
    "GENERATORS",
    "DEFAULT_WEIGHTS",
    "TEMPLATES",
    "DOMAIN_TAGS",
    "DOMAIN_DIFFICULTY",
    "DOMAIN_CONFIG",
    "PACK_HASHES",
    "PACK_META",
]
