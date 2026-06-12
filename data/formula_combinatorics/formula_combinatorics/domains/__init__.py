"""Domain registry: merges GENERATORS, WEIGHTS, and TEMPLATES from all domain modules.

Loading order for each domain:
1. If a ``<domain>.toml`` pack file exists alongside this module, load it via
   ``engine._pack_loader.load_pack()``.  The pack is the authoritative source
   for migrated domains.
2. Otherwise, fall back to the Python ``.<domain>`` sub-module (legacy path).
"""

from __future__ import annotations

import importlib
import random
from collections.abc import Callable
from pathlib import Path

from ..engine._pack_loader import load_pack
from ..engine._template_dsl import Template
from ._config import DOMAIN_CONFIG, register_domain

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

_DOMAINS_DIR = Path(__file__).parent

GENERATORS: dict[str, Callable[[random.Random], str]] = {}
DEFAULT_WEIGHTS: dict[str, float] = {}
TEMPLATES: dict[str, list[Template]] = {}

# Map domain name → pack SHA-256 (populated only for TOML-loaded domains)
PACK_HASHES: dict[str, str] = {}

for _name in _DOMAIN_NAMES:
    _toml_path = _DOMAINS_DIR / f"{_name}.toml"
    if _toml_path.exists():
        _pack = load_pack(_toml_path)
        _gens, _wts, _tmpls = register_domain(_name, _pack.templates)
        PACK_HASHES[_name] = _pack.meta.sha256
    else:
        _mod = importlib.import_module(f".{_name}", package=__package__)
        _gens, _wts, _tmpls = _mod.GENERATORS, _mod.WEIGHTS, _mod.TEMPLATES
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
]
