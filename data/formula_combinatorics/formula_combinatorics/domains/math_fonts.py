"""Math fonts domain — coverage for \\mathsf, \\mathtt, \\mathit, \\mathnormal
and full-expression font wrapping."""

from __future__ import annotations

from .._template_dsl import E, S, Template, X, register_domain
from .._vocab import _SCALARS, _VARS, _atom, _expr

# ---------------------------------------------------------------------------
# Shared pools
# ---------------------------------------------------------------------------

_CAT_POOL = ("Set", "Grp", "Top", "Ab", "Ring", "Vect", "Cat", "Mod", "Mon")
_GRAPH_POOL = ("G", "H", "K", "C", "P", "W", "T")
_ALG_POOL = ("ALG", "DFS", "BFS", "MST", "FFT", "GCD", "LCS", "SAT")
_TT_VAR_POOL = ("x", "y", "w", "u", "v", "s", "t")
_MULTICHAR_IT_POOL = ("dx", "df", "Re", "Im", "Id", "px", "py", "dt", "dp")
_SMALL_INT_POOL = ("n", "m", "k", "N", "M", "2", "3", "4", "8")
_FUNCTOR_POOL = ("F", "G", "H", "T", "U", "I", "J")
_FN_LETTER_POOL = ("f", "g", "h", "p", "q", "r")
_SET_LETTER_POOL = ("A", "B", "C", "D", "X", "Y", "Z")

# ---------------------------------------------------------------------------
# Section 1 — \mathsf sans-serif
# ---------------------------------------------------------------------------

_MATHSF_TEMPLATES: list[Template] = [
    # Category name in sans-serif — the most common real-world usage
    Template(
        name="mathsf_category_single",
        latex=r"\mathsf{{{cc}}}",
        slots={"cc": S(_CAT_POOL)},
    ),
    # Functor arrow between two distinct sans-serif categories
    Template(
        name="mathsf_functor_arrow",
        latex=r"{ff} : \mathsf{{{cc}}} \to \mathsf{{{dd}}}",
        slots={
            "ff": S(tuple(_FUNCTOR_POOL)),
            "cc": S(_CAT_POOL),
            "dd": X(_CAT_POOL, ("cc",)),
        },
    ),
    # Hom-set between two sans-serif categories
    Template(
        name="mathsf_hom_between",
        latex=r"\mathrm{{Hom}}(\mathsf{{{cc}}}, \mathsf{{{dd}}})",
        slots={
            "cc": S(_CAT_POOL),
            "dd": X(_CAT_POOL, ("cc",)),
        },
    ),
    # Chromatic number of a named graph in sans-serif
    Template(
        name="mathsf_graph_chromatic",
        latex=r"\chi(\mathsf{{{gg}}}) \leq {kk}",
        slots={
            "gg": S(_GRAPH_POOL),
            "kk": S(tuple(_SCALARS)),
        },
    ),
    # Algorithm complexity with sans-serif algorithm name
    Template(
        name="mathsf_algorithm_complexity",
        latex=r"\mathsf{{{alg}}}({nn}) = O({nn}^{{{kk}}})",
        slots={
            "alg": S(_ALG_POOL),
            "nn": S(_SMALL_INT_POOL),
            "kk": S(("2", "3", r"\log {nn}", "k")),
        },
    ),
]

# ---------------------------------------------------------------------------
# Section 2 — \mathtt typewriter
# ---------------------------------------------------------------------------

_MATHTT_TEMPLATES: list[Template] = [
    # Bit string variable
    Template(
        name="mathtt_bit_string",
        latex=r"\mathtt{{{xx}}} \in \{{0,1\}}^{{{nn}}}",
        slots={
            "xx": S(_TT_VAR_POOL),
            "nn": S(_SMALL_INT_POOL),
        },
    ),
    # XOR of three distinct typewriter variables
    Template(
        name="mathtt_xor",
        latex=r"\mathtt{{{xx}}} \oplus \mathtt{{{yy}}} = \mathtt{{{zz}}}",
        slots={
            "xx": S(_TT_VAR_POOL),
            "yy": X(_TT_VAR_POOL, ("xx",)),
            "zz": X(_TT_VAR_POOL, ("xx", "yy")),
        },
    ),
    # String / word length
    Template(
        name="mathtt_word_length",
        latex=r"|\mathtt{{{ww}}}| = {nn}",
        slots={
            "ww": S(_TT_VAR_POOL),
            "nn": S(_SMALL_INT_POOL),
        },
    ),
    # Pseudocode assignment
    Template(
        name="mathtt_assignment",
        latex=r"\mathtt{{{xx}}} \leftarrow \mathtt{{{yy}}}",
        slots={
            "xx": S(_TT_VAR_POOL),
            "yy": X(_TT_VAR_POOL, ("xx",)),
        },
    ),
]

# ---------------------------------------------------------------------------
# Section 3 — \mathit math italic
# ---------------------------------------------------------------------------

_MATHIT_TEMPLATES: list[Template] = [
    # Multi-character math-italic identifier (e.g. \mathit{dx}, \mathit{Re})
    Template(
        name="mathit_multichar_var",
        latex=r"\mathit{{{mv}}}",
        slots={"mv": S(_MULTICHAR_IT_POOL)},
    ),
    # Derivative ratio with italic numerator and denominator
    Template(
        name="mathit_derivative_ratio",
        latex=r"\dfrac{{d\mathit{{{ff}}}}}{{d\mathit{{{xx}}}}}",
        slots={
            "ff": S(tuple(_FN_LETTER_POOL)),
            "xx": S(tuple(_VARS)),
        },
    ),
    # Composed maps in italic
    Template(
        name="mathit_composed_map",
        latex=r"\mathit{{{ff}}} \circ \mathit{{{gg}}} : {AA} \to {CC}",
        slots={
            "ff": S(tuple(_FN_LETTER_POOL)),
            "gg": X(tuple(_FN_LETTER_POOL), ("ff",)),
            "AA": S(tuple(_SET_LETTER_POOL)),
            "CC": X(tuple(_SET_LETTER_POOL), ("AA",)),
        },
    ),
    # Sandwiched relation with italic bounds and variable
    Template(
        name="mathit_inline_relation",
        latex=r"\mathit{{{aa}}} \leq \mathit{{{xx}}} \leq \mathit{{{bb}}}",
        slots={
            "aa": S(tuple(_SCALARS)),
            "xx": S(tuple(_VARS)),
            "bb": X(tuple(_SCALARS), ("aa",)),
        },
    ),
]

# ---------------------------------------------------------------------------
# Section 4 — \mathnormal explicit normal
# ---------------------------------------------------------------------------

_MATHNORMAL_TEMPLATES: list[Template] = [
    # Single variable reset to default math italic via \mathnormal
    Template(
        name="mathnormal_single",
        latex=r"\mathnormal{{{ll}}}",
        slots={"ll": S(tuple(_VARS))},
    ),
    # Constraint pair with explicit \mathnormal
    Template(
        name="mathnormal_constraint",
        latex=r"\mathnormal{{{aa}}} \geq 0, \quad \mathnormal{{{bb}}} \leq 1",
        slots={
            "aa": S(tuple(_SCALARS)),
            "bb": X(tuple(_SCALARS), ("aa",)),
        },
    ),
]

# ---------------------------------------------------------------------------
# Section 5 — Full-expression font wrapping (structural gap)
# ---------------------------------------------------------------------------

_FONT_WRAP_TEMPLATES: list[Template] = [
    # Bold: entire linear system wrapped in \mathbf
    Template(
        name="fontbf_linear_relation",
        latex=r"\mathbf{{{AA} {xx} = {bb}}}",
        slots={
            "AA": S(tuple(_SET_LETTER_POOL)),
            "xx": S(tuple(_VARS)),
            "bb": X(tuple(_SCALARS), ()),
        },
    ),
    # Sans-serif: object–arrow–object chain entirely in \mathsf
    Template(
        name="fontsf_category_chain",
        latex=r"\mathsf{{{AA} \xrightarrow{{{ff}}} {BB} \xrightarrow{{{gg}}} {CC}}}",
        slots={
            "AA": S(tuple(_SET_LETTER_POOL)),
            "BB": X(tuple(_SET_LETTER_POOL), ("AA",)),
            "CC": X(tuple(_SET_LETTER_POOL), ("AA", "BB")),
            "ff": S(tuple(_FN_LETTER_POOL)),
            "gg": X(tuple(_FN_LETTER_POOL), ("ff",)),
        },
    ),
    # Italic: entire polynomial expression wrapped in \mathit
    Template(
        name="fontit_polynomial_expr",
        latex=r"\mathit{{{aa} x^2 + {bb} x + {cc} = 0}}",
        slots={
            "aa": S(tuple(_SCALARS)),
            "bb": X(tuple(_SCALARS), ("aa",)),
            "cc": X(tuple(_SCALARS), ("aa", "bb")),
        },
    ),
]

# ---------------------------------------------------------------------------
# Section 6 — \pmb (poor man's bold) and \boldsymbol variants
# ---------------------------------------------------------------------------

_BOLD_EXPR_POOL = (r"\alpha", r"\beta", r"\gamma", r"\delta", r"\lambda", r"\mu", r"\omega", r"\sigma")

_PMB_TEMPLATES: list[Template] = [
    Template(
        name="pmb_single",
        latex=r"\pmb{{{v}}}",
        slots={"v": E(_atom, n=5_000_000)},
    ),
    Template(
        name="pmb_relation",
        latex=r"\pmb{{{v}}} = {w}",
        slots={"v": E(_atom, n=5_000_000), "w": E(_expr, n=5_000_000)},
    ),
    Template(
        name="boldsymbol_greek",
        latex=r"\boldsymbol{{{g}}}",
        slots={"g": S(_BOLD_EXPR_POOL)},
    ),
    Template(
        name="boldsymbol_eq",
        latex=r"\boldsymbol{{{g}}} = {a} \mathbf{{{v}}}",
        slots={"g": S(_BOLD_EXPR_POOL), "a": S(tuple(_SCALARS)), "v": S(tuple(_SET_LETTER_POOL))},
    ),
    Template(
        name="bf_declaration",
        latex=r"{{\bf {v}}} = {a}",
        slots={"v": S(tuple(_SET_LETTER_POOL)), "a": S(tuple(_SCALARS))},
    ),
]

# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

_MATH_FONTS_TEMPLATES: list[Template] = (
    _MATHSF_TEMPLATES
    + _MATHTT_TEMPLATES
    + _MATHIT_TEMPLATES
    + _MATHNORMAL_TEMPLATES
    + _FONT_WRAP_TEMPLATES
    + _PMB_TEMPLATES
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("math_fonts", _MATH_FONTS_TEMPLATES, 0.03)
