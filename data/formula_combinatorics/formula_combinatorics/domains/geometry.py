"""Elementary geometry domain: angles, triangles, parallelism, proof notation."""

from __future__ import annotations

from .._template_dsl import E, S, Template
from ._config import register_domain

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_PTS = ("A", "B", "C", "D", "E", "F", "P", "Q", "R")
_TRI_VERTS = ("A", "B", "C", "D", "E", "F", "X", "Y", "Z")
_ANGLE_LETTERS = ("A", "B", "C", "D", "E", "F", "P", "Q", "R", r"\alpha", r"\beta", r"\gamma")
_DEG_VALS = ("30", "45", "60", "90", "120", "135", "150", "180")
_SCALAR_POOL = ("a", "b", "c", "d", "k", "m", "n", "r")
_CIRCLE_NUMS = ("1", "2", "3", "4", "5", "6")
_SIM_REL = (r"\sim", r"\cong")
_PROOF_BECAUSE_POOL = (
    r"\overline{{AB}} \parallel \overline{{CD}}",
    r"\overline{{AB}} \perp \overline{{CD}}",
    r"\triangle {P}{Q}{R} \cong \triangle {P2}{Q2}{R2}",
    r"\angle {A} = \angle {B}",
)

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_GEO_ANGLE = [
    Template(
        name="angle_eq_angle",
        latex=r"\angle {A}{B}{C} = \angle {D}{E}{F}",
        slots={"A": S(_PTS), "B": S(_PTS), "C": S(_PTS), "D": S(_PTS), "E": S(_PTS), "F": S(_PTS)},
    ),
    Template(
        name="angle_eq_deg",
        latex=r"\angle {A}{B}{C} = {d}^{{\circ}}",
        slots={"A": S(_PTS), "B": S(_PTS), "C": S(_PTS), "d": S(_DEG_VALS)},
    ),
    Template(
        name="angle_sum_180",
        latex=r"\angle {A} + \angle {B} + \angle {C} = 180^{{\circ}}",
        slots={"A": S(_ANGLE_LETTERS), "B": S(_ANGLE_LETTERS), "C": S(_ANGLE_LETTERS)},
    ),
    Template(
        name="angle_sum_90",
        latex=r"\angle {A} + \angle {B} = 90^{{\circ}}",
        slots={"A": S(_ANGLE_LETTERS), "B": S(_ANGLE_LETTERS)},
    ),
    Template(
        name="angle_letter_eq_val",
        latex=r"\angle {A} = {d}^{{\circ}}",
        slots={"A": S(_ANGLE_LETTERS), "d": S(_DEG_VALS)},
    ),
    Template(
        name="angle_ineq",
        latex=r"\angle {A}{B}{C} < \angle {D}{E}{F}",
        slots={"A": S(_PTS), "B": S(_PTS), "C": S(_PTS), "D": S(_PTS), "E": S(_PTS), "F": S(_PTS)},
    ),
]

_GEO_TRIANGLE = [
    Template(
        name="triangle_congruent",
        latex=r"\triangle {P}{Q}{R} \cong \triangle {P2}{Q2}{R2}",
        slots={
            "P": S(_TRI_VERTS),
            "Q": S(_TRI_VERTS),
            "R": S(_TRI_VERTS),
            "P2": S(_TRI_VERTS),
            "Q2": S(_TRI_VERTS),
            "R2": S(_TRI_VERTS),
        },
    ),
    Template(
        name="triangle_similar",
        latex=r"\triangle {P}{Q}{R} \sim \triangle {P2}{Q2}{R2}",
        slots={
            "P": S(_TRI_VERTS),
            "Q": S(_TRI_VERTS),
            "R": S(_TRI_VERTS),
            "P2": S(_TRI_VERTS),
            "Q2": S(_TRI_VERTS),
            "R2": S(_TRI_VERTS),
        },
    ),
    Template(
        name="triangle_area",
        latex=r"S_{{\triangle {P}{Q}{R}}} = \frac{{1}}{{2}} {a} {b} \sin \angle {P}",
        slots={"P": S(_TRI_VERTS), "Q": S(_TRI_VERTS), "R": S(_TRI_VERTS), "a": S(_SCALAR_POOL), "b": S(_SCALAR_POOL)},
    ),
    Template(
        name="triangle_angle_sum",
        latex=r"\angle {P} + \angle {Q} + \angle {R} = 180^{{\circ}}",
        slots={"P": S(_TRI_VERTS), "Q": S(_TRI_VERTS), "R": S(_TRI_VERTS)},
    ),
    Template(
        name="similar_ratio",
        latex=r"\frac{{\overline{{{P}{Q}}}}}{{\overline{{{P2}{Q2}}}}} = \frac{{\overline{{{Q}{R}}}}}{{\overline{{{Q2}{R2}}}}} = {k}",
        slots={
            "P": S(_TRI_VERTS),
            "Q": S(_TRI_VERTS),
            "R": S(_TRI_VERTS),
            "P2": S(_TRI_VERTS),
            "Q2": S(_TRI_VERTS),
            "R2": S(_TRI_VERTS),
            "k": S(_SCALAR_POOL),
        },
    ),
    Template(
        name="pythagorean",
        latex=r"{a}^2 + {b}^2 = {c}^2",
        slots={"a": S(_SCALAR_POOL), "b": S(_SCALAR_POOL), "c": S(_SCALAR_POOL)},
    ),
    Template(
        name="law_of_cosines",
        latex=r"{c}^2 = {a}^2 + {b}^2 - 2{a}{b} \cos \angle {C}",
        slots={"a": S(_SCALAR_POOL), "b": S(_SCALAR_POOL), "c": S(_SCALAR_POOL), "C": S(_TRI_VERTS)},
    ),
]

_GEO_PARALLEL_PERP = [
    Template(
        name="parallel_lines",
        latex=r"{A}{B} \parallel {C}{D}",
        slots={"A": S(_PTS), "B": S(_PTS), "C": S(_PTS), "D": S(_PTS)},
    ),
    Template(
        name="perp_lines",
        latex=r"{A}{B} \perp {C}{D}",
        slots={"A": S(_PTS), "B": S(_PTS), "C": S(_PTS), "D": S(_PTS)},
    ),
    Template(
        name="parallel_overline",
        latex=r"\overline{{{A}{B}}} \parallel \overline{{{C}{D}}}",
        slots={"A": S(_PTS), "B": S(_PTS), "C": S(_PTS), "D": S(_PTS)},
    ),
    Template(
        name="perp_overline",
        latex=r"\overline{{{A}{B}}} \perp \overline{{{C}{D}}}",
        slots={"A": S(_PTS), "B": S(_PTS), "C": S(_PTS), "D": S(_PTS)},
    ),
    Template(
        name="parallel_angle_eq",
        latex=r"\because {A}{B} \parallel {C}{D}, \quad \therefore \angle {E} = \angle {F}",
        slots={"A": S(_PTS), "B": S(_PTS), "C": S(_PTS), "D": S(_PTS), "E": S(_ANGLE_LETTERS), "F": S(_ANGLE_LETTERS)},
    ),
    Template(
        name="perp_angle_90",
        latex=r"{A}{B} \perp {C}{D}, \quad \angle {E}{O}{F} = 90^{{\circ}}",
        slots={"A": S(_PTS), "B": S(_PTS), "C": S(_PTS), "D": S(_PTS), "E": S(_PTS), "O": S(_PTS), "F": S(_PTS)},
    ),
]

_GEO_PROOF = [
    Template(
        name="therefore_angle_eq",
        latex=r"\therefore \angle {A} = \angle {B}",
        slots={"A": S(_ANGLE_LETTERS), "B": S(_ANGLE_LETTERS)},
    ),
    Template(
        name="because_therefore",
        latex=r"\because \angle {A} = \angle {B}, \quad \therefore \triangle {P}{Q}{R} \sim \triangle {P2}{Q2}{R2}",
        slots={
            "A": S(_ANGLE_LETTERS),
            "B": S(_ANGLE_LETTERS),
            "P": S(_TRI_VERTS),
            "Q": S(_TRI_VERTS),
            "R": S(_TRI_VERTS),
            "P2": S(_TRI_VERTS),
            "Q2": S(_TRI_VERTS),
            "R2": S(_TRI_VERTS),
        },
    ),
    Template(
        name="because_parallel_therefore_angle",
        latex=r"\because \overline{{{A}{B}}} \parallel \overline{{{C}{D}}}, \quad \therefore \angle {E} = \angle {F}",
        slots={"A": S(_PTS), "B": S(_PTS), "C": S(_PTS), "D": S(_PTS), "E": S(_ANGLE_LETTERS), "F": S(_ANGLE_LETTERS)},
    ),
    Template(
        name="therefore_congruent",
        latex=r"\therefore \triangle {P}{Q}{R} \cong \triangle {P2}{Q2}{R2}",
        slots={
            "P": S(_TRI_VERTS),
            "Q": S(_TRI_VERTS),
            "R": S(_TRI_VERTS),
            "P2": S(_TRI_VERTS),
            "Q2": S(_TRI_VERTS),
            "R2": S(_TRI_VERTS),
        },
    ),
    Template(
        name="because_perp_therefore",
        latex=r"\because {A}{B} \perp {C}{D}, \quad \therefore \angle {E} = 90^{{\circ}}",
        slots={"A": S(_PTS), "B": S(_PTS), "C": S(_PTS), "D": S(_PTS), "E": S(_ANGLE_LETTERS)},
    ),
]

_GEO_VECTORS = [
    Template(
        name="vec_addition",
        latex=r"\overrightarrow{{{A}{B}}} + \overrightarrow{{{B}{C}}} = \overrightarrow{{{A}{C}}}",
        slots={"A": S(_PTS), "B": S(_PTS), "C": S(_PTS)},
    ),
    Template(
        name="vec_zero",
        latex=r"\overrightarrow{{{A}{B}}} + \overrightarrow{{{B}{A}}} = \vec{{0}}",
        slots={"A": S(_PTS), "B": S(_PTS)},
    ),
    Template(
        name="vec_scalar",
        latex=r"\overrightarrow{{{A}{B}}} = {k} \overrightarrow{{{C}{D}}}",
        slots={"A": S(_PTS), "B": S(_PTS), "C": S(_PTS), "D": S(_PTS), "k": S(_SCALAR_POOL)},
    ),
    Template(
        name="vec_midpoint",
        latex=r"\overrightarrow{{{O}{M}}} = \frac{{1}}{{2}} \left( \overrightarrow{{{O}{A}}} + \overrightarrow{{{O}{B}}} \right)",
        slots={"O": S(_PTS), "M": S(_PTS), "A": S(_PTS), "B": S(_PTS)},
    ),
]

_GEO_ANNOTATION = [
    Template(
        name="circled_num_eq",
        latex=r"\textcircled{{{n}}} \quad {A}{B} = {a}",
        slots={"n": S(_CIRCLE_NUMS), "A": S(_PTS), "B": S(_PTS), "a": S(_SCALAR_POOL)},
    ),
    Template(
        name="circled_num_angle",
        latex=r"\textcircled{{{n}}} \quad \angle {A} = {d}^{{\circ}}",
        slots={"n": S(_CIRCLE_NUMS), "A": S(_ANGLE_LETTERS), "d": S(_DEG_VALS)},
    ),
    Template(
        name="circled_num_parallel",
        latex=r"\textcircled{{{n}}} \quad {A}{B} \parallel {C}{D}",
        slots={"n": S(_CIRCLE_NUMS), "A": S(_PTS), "B": S(_PTS), "C": S(_PTS), "D": S(_PTS)},
    ),
    Template(
        name="circled_num_congruent",
        latex=r"\textcircled{{{n}}} \quad \triangle {P}{Q}{R} \cong \triangle {P2}{Q2}{R2}",
        slots={
            "n": S(_CIRCLE_NUMS),
            "P": S(_TRI_VERTS),
            "Q": S(_TRI_VERTS),
            "R": S(_TRI_VERTS),
            "P2": S(_TRI_VERTS),
            "Q2": S(_TRI_VERTS),
            "R2": S(_TRI_VERTS),
        },
    ),
]

_GEO_MISC = [
    Template(
        name="segment_eq",
        latex=r"\overline{{{A}{B}}} = \overline{{{C}{D}}}",
        slots={"A": S(_PTS), "B": S(_PTS), "C": S(_PTS), "D": S(_PTS)},
    ),
    Template(
        name="segment_sum",
        latex=r"\overline{{{A}{B}}} + \overline{{{B}{C}}} = \overline{{{A}{C}}}",
        slots={"A": S(_PTS), "B": S(_PTS), "C": S(_PTS)},
    ),
    Template(
        name="bigtriangleup",
        latex=r"\bigtriangleup {P}{Q}{R}",
        slots={"P": S(_TRI_VERTS), "Q": S(_TRI_VERTS), "R": S(_TRI_VERTS)},
    ),
    Template(
        name="angle_bisector",
        latex=r"\angle {A}{O}{B} = \frac{{1}}{{2}} \angle {A}{P}{B}",
        slots={"A": S(_PTS), "O": S(_PTS), "B": S(_PTS), "P": S(_PTS)},
    ),
    Template(
        name="circle_arc",
        latex=r"\angle {A}{O}{B} = 2 \angle {A}{C}{B}",
        slots={"A": S(_PTS), "O": S(_PTS), "B": S(_PTS), "C": S(_PTS)},
    ),
]

_GEO_MORE_PROOF = [
    Template(
        name="therefore_parallel",
        latex=r"\therefore {A}{B} \parallel {C}{D}",
        slots={"A": S(_PTS), "B": S(_PTS), "C": S(_PTS), "D": S(_PTS)},
    ),
    Template(
        name="because_angle_sum",
        latex=r"\because \angle {ang1} + \angle {ang2} = 180^{{\circ}}, \quad \therefore {P1}{P2} \parallel {P3}{P4}",
        slots={
            "ang1": S(_ANGLE_LETTERS),
            "ang2": S(_ANGLE_LETTERS),
            "P1": S(_PTS),
            "P2": S(_PTS),
            "P3": S(_PTS),
            "P4": S(_PTS),
        },
    ),
    Template(
        name="therefore_angle_eq_2",
        latex=r"\therefore \angle {A}{B}{C} = \angle {D}{E}{F}",
        slots={"A": S(_PTS), "B": S(_PTS), "C": S(_PTS), "D": S(_PTS), "E": S(_PTS), "F": S(_PTS)},
    ),
    Template(
        name="because_congruent",
        latex=r"\because \triangle {P}{Q}{R} \cong \triangle {P2}{Q2}{R2}, \quad \therefore \angle {P} = \angle {P2}",
        slots={
            "P": S(_TRI_VERTS),
            "Q": S(_TRI_VERTS),
            "R": S(_TRI_VERTS),
            "P2": S(_TRI_VERTS),
            "Q2": S(_TRI_VERTS),
            "R2": S(_TRI_VERTS),
        },
    ),
    Template(
        name="therefore_segment_eq",
        latex=r"\therefore \overline{{{A}{B}}} = \overline{{{C}{D}}}",
        slots={"A": S(_PTS), "B": S(_PTS), "C": S(_PTS), "D": S(_PTS)},
    ),
]

_GEO_MORE_PROOF_RICH = [
    Template(
        name="therefore_parallel_rich",
        latex=r"\therefore \overline{{{P1}{P2}}} \parallel \overline{{{P3}{P4}}}",
        slots={
            "P1": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P2": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P3": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P4": E(lambda rng: rng.choice(_PTS), n=5_000_000),
        },
    ),
    Template(
        name="therefore_congruent_rich",
        latex=r"\therefore \triangle {P1}{Q1}{R1} \cong \triangle {P2}{Q2}{R2}",
        slots={
            "P1": E(lambda rng: rng.choice(_TRI_VERTS), n=5_000_000),
            "Q1": E(lambda rng: rng.choice(_TRI_VERTS), n=5_000_000),
            "R1": E(lambda rng: rng.choice(_TRI_VERTS), n=5_000_000),
            "P2": E(lambda rng: rng.choice(_TRI_VERTS), n=5_000_000),
            "Q2": E(lambda rng: rng.choice(_TRI_VERTS), n=5_000_000),
            "R2": E(lambda rng: rng.choice(_TRI_VERTS), n=5_000_000),
        },
    ),
]

_GEO_MORE_VECTORS = [
    Template(
        name="vec_dot_product",
        latex=r"\overrightarrow{{{A}{B}}} \cdot \overrightarrow{{{C}{D}}} = {a}",
        slots={"A": S(_PTS), "B": S(_PTS), "C": S(_PTS), "D": S(_PTS), "a": S(_SCALAR_POOL)},
    ),
    Template(
        name="vec_parallel",
        latex=r"\overrightarrow{{{A}{B}}} \parallel \overrightarrow{{{C}{D}}}",
        slots={"A": S(_PTS), "B": S(_PTS), "C": S(_PTS), "D": S(_PTS)},
    ),
    Template(
        name="vec_magnitude",
        latex=r"\left| \overrightarrow{{{A}{B}}} \right| = {a}",
        slots={"A": S(_PTS), "B": S(_PTS), "a": S(_SCALAR_POOL)},
    ),
    # High-n_eff versions using E() to ensure these fire at reasonable rate
    Template(
        name="vec_addition_rich",
        latex=r"\overrightarrow{{{P1}{P2}}} + \overrightarrow{{{P2}{P3}}} = \overrightarrow{{{P1}{P3}}}",
        slots={
            "P1": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P2": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P3": E(lambda rng: rng.choice(_PTS), n=5_000_000),
        },
    ),
    Template(
        name="vec_scalar_rich",
        latex=r"\overrightarrow{{{P1}{P2}}} = {k} \overrightarrow{{{P3}{P4}}}",
        slots={
            "P1": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P2": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P3": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P4": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "k": E(lambda rng: rng.choice(_SCALAR_POOL), n=8),
        },
    ),
    Template(
        name="vec_perp_rich",
        latex=r"\overrightarrow{{{P1}{P2}}} \perp \overrightarrow{{{P3}{P4}}}",
        slots={
            "P1": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P2": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P3": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P4": E(lambda rng: rng.choice(_PTS), n=5_000_000),
        },
    ),
]

_GEO_MORE_PARALLEL = [
    Template(
        name="parallel_rich",
        latex=r"{P1}{P2} \parallel {P3}{P4}",
        slots={
            "P1": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P2": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P3": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P4": E(lambda rng: rng.choice(_PTS), n=5_000_000),
        },
    ),
    Template(
        name="parallel_overline_rich",
        latex=r"\overline{{{P1}{P2}}} \parallel \overline{{{P3}{P4}}}",
        slots={
            "P1": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P2": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P3": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P4": E(lambda rng: rng.choice(_PTS), n=5_000_000),
        },
    ),
]

_GEO_MORE_ANNOTATION = [
    Template(
        name="circled_num_similar",
        latex=r"\textcircled{{{n}}} \quad \triangle {P}{Q}{R} \sim \triangle {P2}{Q2}{R2}",
        slots={
            "n": S(_CIRCLE_NUMS),
            "P": S(_TRI_VERTS),
            "Q": S(_TRI_VERTS),
            "R": S(_TRI_VERTS),
            "P2": S(_TRI_VERTS),
            "Q2": S(_TRI_VERTS),
            "R2": S(_TRI_VERTS),
        },
    ),
    Template(
        name="circled_num_perp",
        latex=r"\textcircled{{{n}}} \quad {A}{B} \perp {C}{D}",
        slots={"n": S(_CIRCLE_NUMS), "A": S(_PTS), "B": S(_PTS), "C": S(_PTS), "D": S(_PTS)},
    ),
    Template(
        name="circled_num_because",
        latex=r"\textcircled{{{n}}} \quad \because \angle {A} = \angle {B}",
        slots={"n": S(_CIRCLE_NUMS), "A": S(_ANGLE_LETTERS), "B": S(_ANGLE_LETTERS)},
    ),
    # High-n_eff versions using E() so they are selected at a meaningful rate
    Template(
        name="circled_num_parallel_rich",
        latex=r"\textcircled{{{n}}} \quad \overline{{{P1}{P2}}} \parallel \overline{{{P3}{P4}}}",
        slots={
            "n": E(lambda rng: rng.choice(_CIRCLE_NUMS), n=6),
            "P1": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P2": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P3": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P4": E(lambda rng: rng.choice(_PTS), n=5_000_000),
        },
    ),
    Template(
        name="circled_num_vec_rich",
        latex=r"\textcircled{{{n}}} \quad \overrightarrow{{{P1}{P2}}} = \overrightarrow{{{P3}{P4}}}",
        slots={
            "n": E(lambda rng: rng.choice(_CIRCLE_NUMS), n=6),
            "P1": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P2": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P3": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P4": E(lambda rng: rng.choice(_PTS), n=5_000_000),
        },
    ),
    Template(
        name="circled_num_angle_rich",
        latex=r"\textcircled{{{n}}} \quad \angle {P1}{P2}{P3} = {a}^\circ",
        slots={
            "n": E(lambda rng: rng.choice(_CIRCLE_NUMS), n=6),
            "P1": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P2": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "P3": E(lambda rng: rng.choice(_PTS), n=5_000_000),
            "a": E(lambda rng: rng.choice(_DEG_VALS), n=8),
        },
    ),
]

_GEO_TEMPLATES = (
    _GEO_ANGLE
    + _GEO_TRIANGLE
    + _GEO_PARALLEL_PERP
    + _GEO_PROOF
    + _GEO_MORE_PROOF
    + _GEO_MORE_PROOF_RICH
    + _GEO_VECTORS
    + _GEO_MORE_VECTORS
    + _GEO_MORE_PARALLEL
    + _GEO_ANNOTATION
    + _GEO_MORE_ANNOTATION
    + _GEO_MISC
)

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("geometry", _GEO_TEMPLATES)
