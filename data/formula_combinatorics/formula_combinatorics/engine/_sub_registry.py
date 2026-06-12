"""Named sub-generator registry for TOML template packs.

Every callable that appears as ``E(...)``, ``P(...)``, or ``EP(...)`` in a domain
module must be registered here under a stable string name so that TOML template
packs can reference it by name rather than by embedding a callable.

Naming conventions:
- Atomic / expression generators:  _atom, _expr, _fn_rich, _fn_rich_nosub
- Polynomial generators:            _poly, _poly_mid_N (degree N middle terms)
- Matrix environment generators:    matrix_env_RxC_ENV (rows x cols, env name)
- Matrix-with-ellipsis generators:  matrix_ellipsis_ENV
- Smallmatrix generators:           smallmatrix_RxC (rows x cols)
"""

from __future__ import annotations

from collections.abc import Callable

from ._templates import _matrix_env, _matrix_with_ellipsis, _poly, _poly_mid_factory, _smallmatrix_inline
from ._vocab import _atom, _expr, _fn_rich, _fn_rich_nosub

SUB_GENERATORS: dict[str, Callable] = {
    # -----------------------------------------------------------------------
    # Atomic / expression generators (signature: gen(rng) -> str)
    # -----------------------------------------------------------------------
    "_atom": _atom,
    "_expr": _expr,
    "_fn_rich": _fn_rich,
    "_fn_rich_nosub": _fn_rich_nosub,
    # -----------------------------------------------------------------------
    # Polynomial generators (signature: gen(rng, var) -> str)
    # -----------------------------------------------------------------------
    "_poly": _poly,
    "_poly_mid_2": _poly_mid_factory(2),
    "_poly_mid_3": _poly_mid_factory(3),
    "_poly_mid_4": _poly_mid_factory(4),
    "_poly_mid_5": _poly_mid_factory(5),
    # -----------------------------------------------------------------------
    # Matrix environment generators — pmatrix (parentheses)
    # -----------------------------------------------------------------------
    "matrix_env_2x2_pmatrix": lambda rng: _matrix_env(rng, 2, 2, "pmatrix"),
    "matrix_env_3x3_pmatrix": lambda rng: _matrix_env(rng, 3, 3, "pmatrix"),
    "matrix_env_4x4_pmatrix": lambda rng: _matrix_env(rng, 4, 4, "pmatrix"),
    "matrix_env_3x1_pmatrix": lambda rng: _matrix_env(rng, 3, 1, "pmatrix"),
    "matrix_env_1x3_pmatrix": lambda rng: _matrix_env(rng, 1, 3, "pmatrix"),
    # --- bmatrix (square brackets)
    "matrix_env_2x2_bmatrix": lambda rng: _matrix_env(rng, 2, 2, "bmatrix"),
    "matrix_env_3x3_bmatrix": lambda rng: _matrix_env(rng, 3, 3, "bmatrix"),
    "matrix_env_3x1_bmatrix": lambda rng: _matrix_env(rng, 3, 1, "bmatrix"),
    "matrix_env_1x3_bmatrix": lambda rng: _matrix_env(rng, 1, 3, "bmatrix"),
    # --- vmatrix (single pipes = determinant)
    "matrix_env_2x2_vmatrix": lambda rng: _matrix_env(rng, 2, 2, "vmatrix"),
    "matrix_env_3x3_vmatrix": lambda rng: _matrix_env(rng, 3, 3, "vmatrix"),
    # --- Bmatrix (curly braces)
    "matrix_env_2x2_Bmatrix": lambda rng: _matrix_env(rng, 2, 2, "Bmatrix"),
    "matrix_env_3x3_Bmatrix": lambda rng: _matrix_env(rng, 3, 3, "Bmatrix"),
    # --- Vmatrix (double pipes)
    "matrix_env_2x2_Vmatrix": lambda rng: _matrix_env(rng, 2, 2, "Vmatrix"),
    "matrix_env_3x3_Vmatrix": lambda rng: _matrix_env(rng, 3, 3, "Vmatrix"),
    # --- plain matrix (no delimiters)
    "matrix_env_2x2_matrix": lambda rng: _matrix_env(rng, 2, 2, "matrix"),
    "matrix_env_3x3_matrix": lambda rng: _matrix_env(rng, 3, 3, "matrix"),
    # -----------------------------------------------------------------------
    # Matrix-with-ellipsis generators (signature: gen(rng) -> str)
    # -----------------------------------------------------------------------
    "matrix_ellipsis_pmatrix": lambda rng: _matrix_with_ellipsis(rng, "pmatrix"),
    "matrix_ellipsis_bmatrix": lambda rng: _matrix_with_ellipsis(rng, "bmatrix"),
    "matrix_ellipsis_vmatrix": lambda rng: _matrix_with_ellipsis(rng, "vmatrix"),
    # -----------------------------------------------------------------------
    # Smallmatrix inline generators (signature: gen(rng) -> str)
    # -----------------------------------------------------------------------
    "smallmatrix_2x2": lambda rng: _smallmatrix_inline(rng, 2, 2),
    "smallmatrix_3x2": lambda rng: _smallmatrix_inline(rng, 3, 2),
    "smallmatrix_2x3": lambda rng: _smallmatrix_inline(rng, 2, 3),
}
