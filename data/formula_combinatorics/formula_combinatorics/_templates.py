"""Shared LaTeX fragment builders reused across domain modules."""

from __future__ import annotations

import random

from ._vocab import (
    _BOUNDS,
    _FUNCS,
    _INDICES,
    _MATRIX_NAMES,
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
    point = pt if pt is not None else rng.choice(_BOUNDS + [r"\infty", "0", _s(rng)])
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


_SMALLMATRIX_DELIMS = [
    (r"\bigl(", r"\bigr)"),
    (r"\bigl[", r"\bigr]"),
    (r"\bigl\{", r"\bigr\}"),
    (r"\bigl\langle", r"\bigr\rangle"),
]


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
