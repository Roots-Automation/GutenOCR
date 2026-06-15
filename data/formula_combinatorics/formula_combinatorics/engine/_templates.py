"""Shared LaTeX fragment builders reused across domain modules."""

from __future__ import annotations

import random
from collections.abc import Callable

from ._vocab import (
    _BOUNDS,
    _FUNCS,
    _INDICES,
    _MATRIX_NAMES,
    _SCALARS,
    _VARS,
    _atom,
    _expr,
    _i,
    _lower,
    _s,
    _upper,
    _v,
)


def _limit(rng: random.Random, *, pt: str | None = None) -> str:
    """Return a \\lim expression: \\lim_{v \\to pt} expr."""
    var = _v(rng)
    point = pt if pt is not None else rng.choice(_BOUNDS + (r"\infty", "0", _s(rng)))
    return rf"\lim_{{{var} \to {point}}} {_expr(rng)}"


def _def_integral(
    rng: random.Random,
    lo: str | None = None,
    hi: str | None = None,
) -> str:
    """Return a definite integral \\int_lo^hi expr dv."""
    var = _v(rng)
    lower = lo if lo is not None else _lower(rng)
    upper = hi if hi is not None else _upper(rng)
    lim = r"\limits" if rng.random() < 0.5 else ""
    return rf"\int{lim}_{{{lower}}}^{{{upper}}} {_expr(rng)} \, d{var}"


def _indef_integral(rng: random.Random) -> str:
    """Return an indefinite integral \\int expr dv."""
    var = _v(rng)
    return rf"\int {_expr(rng)} \, d{var}"


def _partial_deriv(rng: random.Random, *, order: int = 1) -> str:
    """Return a partial derivative fragment \\frac{\\partial^n f}{\\partial v^n}."""
    f = rng.choice(["f", "u", "g", "h"])
    var = _v(rng)
    if order == 1:
        return rf"\frac{{\partial {f}}}{{\partial {var}}}"
    return rf"\frac{{\partial^{{{order}}} {f}}}{{\partial {var}^{{{order}}}}}"


def _mixed_partial(rng: random.Random) -> str:
    """Return a mixed second partial derivative \\frac{\\partial^2 f}{\\partial v \\partial w}."""
    f = rng.choice(["f", "u", "g", "h"])
    vars_ = rng.sample(["x", "y", "z", "t", "u", "v"], 2)
    return rf"\frac{{\partial^2 {f}}}{{\partial {vars_[0]} \, \partial {vars_[1]}}}"


def _matrix_env(
    rng: random.Random,
    rows: int,
    cols: int,
    env: str = "pmatrix",
) -> str:
    """Return a LaTeX matrix environment filled with random atoms."""
    entries = [_atom(rng) for _ in range(rows * cols)]
    row_strs = []
    for r in range(rows):
        row_strs.append(" & ".join(entries[r * cols : (r + 1) * cols]))
    body = r" \\ ".join(row_strs)
    return rf"\begin{{{env}}} {body} \end{{{env}}}"


_SMALLMATRIX_DELIMS: tuple[tuple[str, str], ...] = (
    (r"\bigl(", r"\bigr)"),
    (r"\bigl[", r"\bigr]"),
    (r"\bigl\{", r"\bigr\}"),
    (r"\bigl\langle", r"\bigr\rangle"),
)


def _smallmatrix_inline(
    rng: random.Random,
    rows: int,
    cols: int,
) -> str:
    """Return a smallmatrix wrapped in a randomly chosen \\big delimiter pair."""
    entries = [_atom(rng) for _ in range(rows * cols)]
    row_strs = []
    for r in range(rows):
        row_strs.append(" & ".join(entries[r * cols : (r + 1) * cols]))
    body = r" \\ ".join(row_strs)
    open_, close = rng.choice(_SMALLMATRIX_DELIMS)
    return rf"{open_}\begin{{smallmatrix}} {body} \end{{smallmatrix}}{close}"


def _sum_indexed(rng: random.Random, lo: str, hi: str) -> str:
    """Return a \\sum_{idx=lo}^{hi} expr."""
    idx = _i(rng)
    lim = r"\limits" if rng.random() < 0.5 else ""
    return rf"\sum{lim}_{{{idx}={lo}}}^{{{hi}}} {_expr(rng)}"


def _norm(rng: random.Random, p: str | None = None) -> str:
    """Return a norm \\|expr\\|_p."""
    inner = _atom(rng)
    sub = p if p is not None else rng.choice(["1", "2", r"\infty", "p", "F"])
    return rf"\|{inner}\|_{{{sub}}}"


def _func_apply(rng: random.Random) -> str:
    """Return fn\\!\\left(expr\\right) for a random function."""
    fn = rng.choice(_FUNCS)
    return rf"{fn}\!\left({_expr(rng)}\right)"


def _matrix_with_ellipsis(rng: random.Random, env: str = "pmatrix") -> str:
    """Return a matrix showing corner entries with \\vdots/\\cdots/\\ddots ellipsis rows."""
    mode = rng.choice(["corner", "block"])
    n = rng.choice(["n", "m", "N", "M"])
    name = rng.choice(_MATRIX_NAMES)
    a = name.lower()

    if mode == "corner":
        # Full corner pattern: a_{11} ... a_{1n} / vdots ddots vdots / a_{n1} ... a_{nn}
        top = rf"{a}_{{11}} & \cdots & {a}_{{1{n}}}"
        mid = r"\vdots & \ddots & \vdots"
        bot = rf"{a}_{{{n}1}} & \cdots & {a}_{{{n}{n}}}"
        body = rf"{top} \\ {mid} \\ {bot}"
    else:
        # Block pattern: concrete first row/col, ellipsis at end
        body = (
            rf"{a}_{{11}} & {a}_{{12}} & \cdots \\"
            rf" {a}_{{21}} & {a}_{{22}} & \cdots \\"
            rf" \vdots & \vdots & \ddots"
        )

    return rf"\begin{{{env}}} {body} \end{{{env}}}"


_SUBSTACK_CONDS = [
    lambda i, s, rng: rf"{i} \neq {rng.choice(_INDICES)}",
    lambda i, s, rng: rf"\gcd({i}, {s}) = 1",
    lambda i, s, rng: rf"{i} \geq 1",
    lambda i, s, rng: rf"{i} \text{{ prime}}",
    lambda i, s, rng: rf"{i} \nmid {s}",
]


def _substack_sum(rng: random.Random) -> str:
    """Return a \\sum with a \\substack multi-condition subscript."""
    idx = _i(rng)
    bound = _s(rng)
    hi = _upper(rng)
    cond_fn = rng.choice(_SUBSTACK_CONDS)
    cond = cond_fn(idx, bound, rng)
    lim = r"\limits" if rng.random() < 0.5 else ""
    return rf"\sum{lim}_{{\substack{{{idx}=1\\{cond}}}}}^{{{hi}}} {_expr(rng)}"


def _substack_prod(rng: random.Random) -> str:
    """Return a \\prod with a \\substack multi-condition subscript."""
    idx = _i(rng)
    bound = _s(rng)
    hi = _upper(rng)
    cond_fn = rng.choice(_SUBSTACK_CONDS)
    cond = cond_fn(idx, bound, rng)
    lim = r"\limits" if rng.random() < 0.5 else ""
    return rf"\prod{lim}_{{\substack{{{idx}=1\\{cond}}}}}^{{{hi}}} {_expr(rng)}"


def _interval(rng: random.Random) -> str:
    """Return an interval using \\lbrack/\\rbrack or ( ) delimiters."""
    a = rng.choice([_lower(rng), _s(rng), "-" + _s(rng)])
    b = rng.choice([_upper(rng), _s(rng)])
    left, right = rng.choice(
        [
            (r"\lbrack", r"\rbrack"),
            ("(", r"\rbrack"),
            (r"\lbrack", ")"),
            ("(", ")"),
        ]
    )
    return rf"{left} {a}, {b} {right}"


def _poly_mid_factory(max_exp: int) -> Callable[[random.Random, str], str]:
    """Factory for expanded-polynomial middle-term generators.

    Returns a ParamSub-compatible function that produces the inner terms
    ``c_{max_exp} v^{max_exp} + ... + c_2 v^2`` for use in fully-expanded
    polynomial templates (degree = max_exp + 1).
    """

    def _poly_mid(rng: random.Random, v: str) -> str:
        return " + ".join(rf"{_s(rng)} {v}^{{{i}}}" for i in range(max_exp, 1, -1))

    return _poly_mid


def _multinomial_full(rng: random.Random, n: str) -> str:
    """Return a full multinomial coefficient formula with two distinct scalars drawn from _SCALARS excluding n."""
    pool = [s for s in _SCALARS if s != n]
    a, b = rng.sample(pool, 2)
    return (
        rf"\binom{{{n}}}{{{a},\,{b},\,{n}-{a}-{b}}} "
        rf"= \frac{{{n}!}}{{{a}!\,{b}!\,({n}-{a}-{b})!}}"
    )


def _recurrence_rhs(rng: random.Random, n: str) -> str:
    """Return a master-theorem RHS string using the drawn n variable name."""
    return rng.choice([f"O({n})", "O(1)", f"O({n}^2)", rf"O(\log {n})", rf"O({n} \log {n})"])


def _poly(rng: random.Random, var: str, *, max_degree: int = 5) -> str:
    """Return a random polynomial in var with scalar coefficients.

    Degree 1–max_degree, sparse (intermediate terms included at ~60%),
    optional constant term (50%), signs between terms independently +/-.
    Always produces at least two terms.
    """
    degree = rng.randint(1, max_degree)

    # Collect (power, coeff) for each term that appears
    included: list[tuple[int, str]] = []
    for i in range(degree, 0, -1):
        if i == degree or rng.random() < 0.6:
            included.append((i, _s(rng)))
    if rng.random() < 0.5:
        included.append((0, _s(rng)))

    # Guarantee at least two terms
    if len(included) < 2:
        included.append((0, _s(rng)))

    # Assemble LaTeX with independent +/- between terms
    parts: list[str] = []
    for idx, (power, coeff) in enumerate(included):
        if power == 0:
            term = coeff
        elif power == 1:
            term = rf"{coeff} {var}"
        else:
            term = rf"{coeff} {var}^{{{power}}}"
        if idx == 0:
            parts.append(term)
        else:
            parts.append(rng.choice(["+", "-"]) + " " + term)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Align / multiline environment generators (moved from domains/align.py)
# ---------------------------------------------------------------------------

_ALIGN_DELIM_PAIRS = (("(", ")"), ("[", "]"))
_ALIGN_LHS_SIMPLE = ("f(x)", "g(t)", "y", "I", "S")
_ALIGN_LHS_VAR = ("y", "z", "w")
_ALIGN_LHS_CALC = ("f(x)", "g(t)", "y", "S", "I")


def _diff_op(rng: random.Random) -> str:
    v = rng.choice(_VARS)
    return rng.choice([rf"\frac{{d}}{{d{v}}}", rf"\frac{{d^2}}{{d{v}^2}}"])


def _piecewise_2(rng: random.Random) -> str:
    v = _v(rng)
    e1, e2 = _expr(rng, 1), _expr(rng, 1)
    cond = rng.choice(["0", _s(rng), r"\pi"])
    body = (
        rf"\begin{{cases}} "
        rf"{e1} & \text{{if }} {v} \geq {cond} \\"
        rf" {e2} & \text{{if }} {v} < {cond} "
        rf"\end{{cases}}"
    )
    return rf"\begin{{align*}}f({v}) &= {body}\end{{align*}}"


def _piecewise_3(rng: random.Random) -> str:
    v = _v(rng)
    e1, e2, e3 = _expr(rng, 1), _expr(rng, 1), _expr(rng, 1)
    body = (
        rf"\begin{{cases}} "
        rf"{e1} & \text{{if }} {v} > 0 \\"
        rf" {e2} & \text{{if }} {v} = 0 \\"
        rf" {e3} & \text{{if }} {v} < 0 "
        rf"\end{{cases}}"
    )
    return rf"\begin{{align*}}g({v}) &= {body}\end{{align*}}"


def _invis_bracket_split(rng: random.Random) -> str:
    lhs = rng.choice(("f(x)", "g(t)", "y", "S"))
    op, cl = rng.choice(_ALIGN_DELIM_PAIRS)
    e1, e2, e3, e4 = _expr(rng, 2), _expr(rng, 2), _expr(rng, 1), _expr(rng, 1)
    lines = [
        rf"{lhs} &= \left{op} {e1} + {e2} + \cdots \right.",
        rf"&\left. \quad + {e3} + {e4} \right{cl}",
    ]
    sep = r" \\"
    return rf"\begin{{align*}}{sep.join(lines)}\end{{align*}}"


def _integral_bounds_split(rng: random.Random) -> str:
    v = _v(rng)
    fn = rng.choice([rf"\frac{{d}}{{d{v}}}", rf"\frac{{d^2}}{{d{v}^2}}"])
    a, b = _s(rng), _s(rng)
    e1, e2, e3, e4 = _expr(rng, 2), _expr(rng, 1), _expr(rng, 1), _expr(rng, 1)
    lines = [
        rf"I &= \left[ {fn}\left( {e1} \right) + {e2} \right.",
        rf"&\left. \quad - {e3} \cdot {e4} \right]_{{{a}}}^{{{b}}}",
    ]
    sep = r" \\"
    return rf"\begin{{align*}}{sep.join(lines)}\end{{align*}}"


def _grouped_coeff(rng: random.Random) -> str:
    lhs = rng.choice(_ALIGN_LHS_VAR)
    c = _s(rng)
    e1, e2, e3, e4, e5, e6 = (
        _expr(rng, 2),
        _expr(rng, 2),
        _expr(rng, 1),
        _expr(rng, 1),
        _expr(rng, 1),
        _expr(rng, 1),
    )
    lines = [
        rf"{lhs} &= {c} \left( \frac{{{e1}}}{{{e2}}} + {e3} \right.",
        rf"&\left. \qquad + \frac{{{e4}}}{{{e5}}} \right) + {e6}",
    ]
    sep = r" \\"
    return rf"\begin{{align*}}{sep.join(lines)}\end{{align*}}"


def _multline_poly_body(rng: random.Random, env: str) -> str:
    v = _v(rng)
    poly = _poly(rng, v, max_degree=rng.randint(4, 6))
    terms = poly.split("+")
    mid = max(1, len(terms) // 2)
    first = "+".join(terms[:mid]).rstrip()
    rest = "+".join(terms[mid:]).lstrip()
    return rf"\begin{{{env}}}{first} \\ \quad + {rest}\end{{{env}}}"


def _multline_poly_starred(rng: random.Random) -> str:
    return _multline_poly_body(rng, "multline*")


def _multline_poly_numbered(rng: random.Random) -> str:
    return _multline_poly_body(rng, "multline")


def _gather_body(rng: random.Random, env: str) -> str:
    n_eqns = rng.randint(2, 3)
    eqn_lines = []
    for _ in range(n_eqns):
        lhs = rng.choice((_v(rng), "f(x)", "g(t)", "y"))
        eqn_lines.append(rf"{lhs} = {_expr(rng, 2)}")
    sep = r" \\"
    return rf"\begin{{{env}}}{sep.join(eqn_lines)}\end{{{env}}}"


def _gather_starred(rng: random.Random) -> str:
    return _gather_body(rng, "gather*")


def _gather_numbered(rng: random.Random) -> str:
    return _gather_body(rng, "gather")


def _multicolumn_scalars(rng: random.Random) -> str:
    rows = []
    for _ in range(3):
        v1, v2, v3 = _v(rng), _v(rng), _v(rng)
        c1, c2, c3 = _s(rng), _s(rng), _s(rng)
        rows.append(rf"{v1} &= {c1} && {v2} &= {c2} && {v3} &= {c3}")
    sep = r" \\"
    return rf"\begin{{align*}}{sep.join(rows)}\end{{align*}}"


def _multicolumn_mixed(rng: random.Random) -> str:
    rows = []
    for _ in range(2):
        v1, v2, v3 = _v(rng), _v(rng), _v(rng)
        e1, e2, e3 = _expr(rng, 1), _expr(rng, 1), _expr(rng, 1)
        rows.append(rf"{v1} &= {e1} && {v2} &= {e2} && {v3} &= {e3}")
    sep = r" \\"
    return rf"\begin{{align*}}{sep.join(rows)}\end{{align*}}"


def _multicolumn_inequalities(rng: random.Random) -> str:
    rel = rng.choice((r"\leq", r"\geq"))
    rows = []
    for _ in range(3):
        v1, v2 = _v(rng), _v(rng)
        e1, e2 = _expr(rng, 1), _expr(rng, 1)
        rows.append(rf"{v1} &{rel} {e1} && {v2} &{rel} {e2}")
    sep = r" \\"
    return rf"\begin{{align*}}{sep.join(rows)}\end{{align*}}"
