"""Domain registry: merges GENERATORS, WEIGHTS, and TEMPLATES from all domain modules."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import Template
from .algebra import GENERATORS as _G_ALGEBRA
from .algebra import TEMPLATES as _T_ALGEBRA
from .algebra import WEIGHTS as _W_ALGEBRA
from .analysis import GENERATORS as _G_ANALYSIS
from .analysis import TEMPLATES as _T_ANALYSIS
from .analysis import WEIGHTS as _W_ANALYSIS
from .calculus import GENERATORS as _G_CALCULUS
from .calculus import TEMPLATES as _T_CALCULUS
from .calculus import WEIGHTS as _W_CALCULUS
from .category_theory import GENERATORS as _G_CATTHY
from .category_theory import TEMPLATES as _T_CATTHY
from .category_theory import WEIGHTS as _W_CATTHY
from .combinatorics import GENERATORS as _G_COMB
from .combinatorics import TEMPLATES as _T_COMB
from .combinatorics import WEIGHTS as _W_COMB
from .complex_analysis import GENERATORS as _G_COMPLEX
from .complex_analysis import TEMPLATES as _T_COMPLEX
from .complex_analysis import WEIGHTS as _W_COMPLEX
from .custom_operators import GENERATORS as _G_CUSTOP
from .custom_operators import TEMPLATES as _T_CUSTOP
from .custom_operators import WEIGHTS as _W_CUSTOP
from .differential_equations import GENERATORS as _G_DIFFEQ
from .differential_equations import TEMPLATES as _T_DIFFEQ
from .differential_equations import WEIGHTS as _W_DIFFEQ
from .differential_geometry import GENERATORS as _G_DIFFGEOM
from .differential_geometry import TEMPLATES as _T_DIFFGEOM
from .differential_geometry import WEIGHTS as _W_DIFFGEOM
from .fourier import GENERATORS as _G_FOURIER
from .fourier import TEMPLATES as _T_FOURIER
from .fourier import WEIGHTS as _W_FOURIER
from .geometry import GENERATORS as _G_GEO
from .geometry import TEMPLATES as _T_GEO
from .geometry import WEIGHTS as _W_GEO
from .graph_theory import GENERATORS as _G_GT
from .graph_theory import TEMPLATES as _T_GT
from .graph_theory import WEIGHTS as _W_GT
from .group_theory import GENERATORS as _G_GROUP
from .group_theory import TEMPLATES as _T_GROUP
from .group_theory import WEIGHTS as _W_GROUP
from .information_theory import GENERATORS as _G_INFO
from .information_theory import TEMPLATES as _T_INFO
from .information_theory import WEIGHTS as _W_INFO
from .linear_algebra import GENERATORS as _G_LINEAR
from .linear_algebra import TEMPLATES as _T_LINEAR
from .linear_algebra import WEIGHTS as _W_LINEAR
from .logic import GENERATORS as _G_LOGIC
from .logic import TEMPLATES as _T_LOGIC
from .logic import WEIGHTS as _W_LOGIC
from .math_fonts import GENERATORS as _G_FONTS
from .math_fonts import TEMPLATES as _T_FONTS
from .math_fonts import WEIGHTS as _W_FONTS
from .measure_theory import GENERATORS as _G_MEASURE
from .measure_theory import TEMPLATES as _T_MEASURE
from .measure_theory import WEIGHTS as _W_MEASURE
from .number_theory import GENERATORS as _G_NT
from .number_theory import TEMPLATES as _T_NT
from .number_theory import WEIGHTS as _W_NT
from .optimization import GENERATORS as _G_OPT
from .optimization import TEMPLATES as _T_OPT
from .optimization import WEIGHTS as _W_OPT
from .p_adic import GENERATORS as _G_PADIC
from .p_adic import TEMPLATES as _T_PADIC
from .p_adic import WEIGHTS as _W_PADIC
from .physics import GENERATORS as _G_PHYSICS
from .physics import TEMPLATES as _T_PHYSICS
from .physics import WEIGHTS as _W_PHYSICS
from .probability import GENERATORS as _G_PROB
from .probability import TEMPLATES as _T_PROB
from .probability import WEIGHTS as _W_PROB
from .quantum_notation import GENERATORS as _G_QUANTUM
from .quantum_notation import TEMPLATES as _T_QUANTUM
from .quantum_notation import WEIGHTS as _W_QUANTUM
from .representation import GENERATORS as _G_REPR
from .representation import TEMPLATES as _T_REPR
from .representation import WEIGHTS as _W_REPR
from .ring_theory import GENERATORS as _G_RING
from .ring_theory import TEMPLATES as _T_RING
from .ring_theory import WEIGHTS as _W_RING
from .set_theory import GENERATORS as _G_SET
from .set_theory import TEMPLATES as _T_SET
from .set_theory import WEIGHTS as _W_SET
from .stochastic_processes import GENERATORS as _G_SPROC
from .stochastic_processes import TEMPLATES as _T_SPROC
from .stochastic_processes import WEIGHTS as _W_SPROC
from .topology import GENERATORS as _G_TOPOLOGY
from .topology import TEMPLATES as _T_TOPOLOGY
from .topology import WEIGHTS as _W_TOPOLOGY
from .trigonometry import GENERATORS as _G_TRIG
from .trigonometry import TEMPLATES as _T_TRIG
from .trigonometry import WEIGHTS as _W_TRIG

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    **_G_ALGEBRA,
    **_G_TRIG,
    **_G_CALCULUS,
    **_G_CATTHY,
    **_G_ANALYSIS,
    **_G_DIFFEQ,
    **_G_LINEAR,
    **_G_PROB,
    **_G_INFO,
    **_G_NT,
    **_G_COMB,
    **_G_GT,
    **_G_GROUP,
    **_G_RING,
    **_G_REPR,
    **_G_DIFFGEOM,
    **_G_TOPOLOGY,
    **_G_COMPLEX,
    **_G_FOURIER,
    **_G_PHYSICS,
    **_G_QUANTUM,
    **_G_MEASURE,
    **_G_PADIC,
    **_G_OPT,
    **_G_SET,
    **_G_LOGIC,
    **_G_SPROC,
    **_G_CUSTOP,
    **_G_FONTS,
    **_G_GEO,
}

DEFAULT_WEIGHTS: dict[str, float] = {
    **_W_ALGEBRA,
    **_W_TRIG,
    **_W_CALCULUS,
    **_W_CATTHY,
    **_W_ANALYSIS,
    **_W_DIFFEQ,
    **_W_LINEAR,
    **_W_PROB,
    **_W_INFO,
    **_W_NT,
    **_W_COMB,
    **_W_GT,
    **_W_GROUP,
    **_W_RING,
    **_W_REPR,
    **_W_DIFFGEOM,
    **_W_TOPOLOGY,
    **_W_COMPLEX,
    **_W_FOURIER,
    **_W_PHYSICS,
    **_W_QUANTUM,
    **_W_MEASURE,
    **_W_PADIC,
    **_W_OPT,
    **_W_SET,
    **_W_LOGIC,
    **_W_SPROC,
    **_W_CUSTOP,
    **_W_FONTS,
    **_W_GEO,
}

# Maps domain name → list of Template objects.
# Populated incrementally as domain files are ported to the template DSL.
# Domains not yet ported are absent from this dict.
TEMPLATES: dict[str, list[Template]] = {
    **_T_ALGEBRA,
    **_T_TRIG,
    **_T_CALCULUS,
    **_T_CATTHY,
    **_T_ANALYSIS,
    **_T_DIFFEQ,
    **_T_LINEAR,
    **_T_PROB,
    **_T_INFO,
    **_T_NT,
    **_T_COMB,
    **_T_GT,
    **_T_GROUP,
    **_T_RING,
    **_T_REPR,
    **_T_DIFFGEOM,
    **_T_TOPOLOGY,
    **_T_COMPLEX,
    **_T_FOURIER,
    **_T_PHYSICS,
    **_T_QUANTUM,
    **_T_MEASURE,
    **_T_PADIC,
    **_T_OPT,
    **_T_SET,
    **_T_LOGIC,
    **_T_SPROC,
    **_T_CUSTOP,
    **_T_FONTS,
    **_T_GEO,
}

assert set(DEFAULT_WEIGHTS) == set(GENERATORS), (
    f"Weight/generator mismatch: "
    f"extra weights={set(DEFAULT_WEIGHTS) - set(GENERATORS)}, "
    f"missing weights={set(GENERATORS) - set(DEFAULT_WEIGHTS)}"
)

__all__ = ["GENERATORS", "DEFAULT_WEIGHTS", "TEMPLATES"]
