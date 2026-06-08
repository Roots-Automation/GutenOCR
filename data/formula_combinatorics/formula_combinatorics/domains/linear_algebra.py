"""Linear algebra domain generator."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import E, S, Template, X, compute_weights, make_dispatcher
from .._templates import _matrix_env
from .._vocab import _MATRIX_NAMES, _atom

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_N_POOL = ["n", "m", "N"]
_P_NORM_POOL = ["1", "2", r"\infty", "F"]

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_LINEAR_ALGEBRA_TEMPLATES: list[Template] = [
    Template(
        name="2x2_pmatrix",
        latex=r"\begin{{equation*}}{mat}\end{{equation*}}",
        slots={"mat": E(lambda rng: _matrix_env(rng, 2, 2, "pmatrix"), n=50)},
    ),
    Template(
        name="3x3_vmatrix_det",
        latex=r"\det {mat}",
        slots={"mat": E(lambda rng: _matrix_env(rng, 3, 3, "vmatrix"), n=200)},
    ),
    Template(
        name="2x2_determinant_expansion",
        latex=(
            r"\begin{{equation*}}\begin{{vmatrix}} {e0} & {e1} \\ {e2} & {e3} \end{{vmatrix}} "
            r"= {e0} {e3} - {e1} {e2}\end{{equation*}}"
        ),
        slots={
            "e0": E(_atom, n=150),
            "e1": E(_atom, n=150),
            "e2": E(_atom, n=150),
            "e3": E(_atom, n=150),
        },
    ),
    Template(
        name="eigenvalue_equation",
        latex=r"{m} \mathbf{{v}} = \lambda \mathbf{{v}}",
        slots={"m": S(_MATRIX_NAMES)},
    ),
    Template(
        name="characteristic_polynomial",
        latex=r"\det\!\left({m} - \lambda I\right) = 0",
        slots={"m": S(_MATRIX_NAMES)},
    ),
    Template(
        name="inverse_product",
        latex=r"(AB)^{{-1}} = B^{{-1}} A^{{-1}}",
        slots={},
    ),
    Template(
        name="dot_product",
        latex=r"\mathbf{{u}} \cdot \mathbf{{v}} = \sum_{{i=1}}^{{{n}}} u_i v_i",
        slots={"n": S(_N_POOL)},
    ),
    Template(
        name="cross_product",
        latex=(
            r"\mathbf{{u}} \times \mathbf{{v}} = "
            r"\begin{{vmatrix}} \mathbf{{i}} & \mathbf{{j}} & \mathbf{{k}} \\ "
            r"u_1 & u_2 & u_3 \\ "
            r"v_1 & v_2 & v_3 \end{{vmatrix}}"
        ),
        slots={},
    ),
    Template(
        name="matrix_norm_triangle",
        latex=r"\|{m} + {m2}\|_{{{p}}} \leq \|{m}\|_{{{p}}} + \|{m2}\|_{{{p}}}",
        slots={"m": S(_MATRIX_NAMES), "m2": X(_MATRIX_NAMES, ["m"]), "p": S(_P_NORM_POOL)},
    ),
    Template(
        name="trace_det_eigenvalues",
        latex=r"\operatorname{{tr}}({m}) = \sum_{{i=1}}^{{{n}}} \lambda_i, \quad \det({m}) = \prod_{{i=1}}^{{{n}}} \lambda_i",
        slots={"m": S(_MATRIX_NAMES), "n": S(_N_POOL)},
    ),
    Template(
        name="svd",
        latex=r"{m} = U \Sigma V^\top",
        slots={"m": S(_MATRIX_NAMES)},
    ),
    Template(
        name="operator_norm_bound",
        latex=r"\|{m} \mathbf{{x}}\|_2 \leq \|{m}\|_2 \|\mathbf{{x}}\|_2",
        slots={"m": S(_MATRIX_NAMES)},
    ),
    Template(
        name="2d_dot_product",
        latex=(
            r"\begin{{equation*}}\begin{{pmatrix}} {e0} \\ {e1} \end{{pmatrix}} \cdot "
            r"\begin{{pmatrix}} {e2} \\ {e3} \end{{pmatrix}} = "
            r"{e0} {e2} + {e1} {e3}\end{{equation*}}"
        ),
        slots={
            "e0": E(_atom, n=150),
            "e1": E(_atom, n=150),
            "e2": E(_atom, n=150),
            "e3": E(_atom, n=150),
        },
    ),
    Template(
        name="2x2_bmatrix",
        latex=r"\begin{{equation*}}{mat}\end{{equation*}}",
        slots={"mat": E(lambda rng: _matrix_env(rng, 2, 2, "bmatrix"), n=50)},
    ),
    Template(
        name="rank_nullity",
        latex=r"\operatorname{{rank}}({m}) + \operatorname{{null}}({m}) = {n}",
        slots={"m": S(_MATRIX_NAMES), "n": S(_N_POOL)},
    ),
    Template(
        name="qr_decomposition",
        latex=r"{m} = QR",
        slots={"m": S(_MATRIX_NAMES)},
    ),
    Template(
        name="orthogonal_projection",
        latex=r"P_{{V}} \mathbf{{x}} = \frac{{\mathbf{{v}} \cdot \mathbf{{x}}}}{{\mathbf{{v}} \cdot \mathbf{{v}}}} \mathbf{{v}}",
        slots={},
    ),
    Template(
        name="spectral_decomposition",
        latex=r"{m} = \sum_{{i=1}}^{{{n}}} \lambda_i \mathbf{{u}}_i \mathbf{{u}}_i^\top",
        slots={"m": S(_MATRIX_NAMES), "n": S(_N_POOL)},
    ),
]

_PART_PERP: list[Template] = [
    Template(
        name="orthogonal_complement_def",
        latex=r"{m}^\perp = \{{\mathbf{{v}} \in \mathbb{{R}}^{{{n}}} \mid \mathbf{{v}} \cdot \mathbf{{u}} = 0 \;\forall\,\mathbf{{u}} \in {m}\}}",
        slots={"m": S(_MATRIX_NAMES), "n": S(_N_POOL)},
    ),
    Template(
        name="orthogonal_vectors_iff",
        latex=r"\mathbf{{u}} \perp \mathbf{{v}} \iff \mathbf{{u}} \cdot \mathbf{{v}} = 0",
        slots={},
    ),
    Template(
        name="gram_schmidt_orthogonality",
        latex=r"\mathbf{{e}}_i \perp \mathbf{{e}}_j \;\forall\; i \neq j \quad \text{{(Gram--Schmidt output)}}",
        slots={},
    ),
    Template(
        name="orthogonal_direct_sum",
        latex=r"\mathbb{{R}}^{{{n}}} = {m} \oplus {m}^\perp",
        slots={"m": S(_MATRIX_NAMES), "n": S(_N_POOL)},
    ),
]

_LINEAR_ALGEBRA_TEMPLATES += _PART_PERP

_PART_ARROWS: list[Template] = [
    Template(
        name="linear_map_rightarrow",
        latex=r"T: \mathbb{{R}}^{{{m}}} \rightarrow \mathbb{{R}}^{{{n}}}",
        slots={"m": S(_N_POOL), "n": S(_N_POOL)},
    ),
    Template(
        name="matrix_action_mapsto",
        latex=r"{mm}: \mathbf{{v}} \mapsto {mm}\mathbf{{v}}, \quad {mm} \in \mathbb{{R}}^{{{n} \times {n}}}",
        slots={"mm": S(_MATRIX_NAMES), "n": S(_N_POOL)},
    ),
    Template(
        name="dual_map_longmapsto",
        latex=r"T^*: \phi \longmapsto \phi \circ T, \quad \phi \in V^*",
        slots={},
    ),
]
_LINEAR_ALGEBRA_TEMPLATES += _PART_ARROWS

_W = compute_weights(_LINEAR_ALGEBRA_TEMPLATES)

_linear_algebra = make_dispatcher(_LINEAR_ALGEBRA_TEMPLATES, _W)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "linear_algebra": _linear_algebra,
}

WEIGHTS: dict[str, float] = {
    "linear_algebra": 0.08,
}

TEMPLATES: dict = {
    "linear_algebra": _LINEAR_ALGEBRA_TEMPLATES,
}
