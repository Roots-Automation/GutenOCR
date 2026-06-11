"""Multi-line LaTeX align* and cases environment generator.

DEPRECATED: This module exists for backward compatibility only.
The _align() function has been replaced by the 'align' domain registered in
formula_combinatorics/domains/align.py, which is now sampled via the standard
domain weighting mechanism. corpus.generate() no longer calls _align() directly.
"""

from __future__ import annotations

import random

from ._templates import _poly
from ._vocab import _expr, _overbrace, _s, _underbrace, _v

_LABELS = ("eq1", "eq2", "main", "result", "key", "def", "prop")


def _align(rng: random.Random) -> str:
    """Generate a multi-line LaTeX block (align*, multline, gather, split, etc.)."""
    style = rng.randint(0, 26)
    env = "align*"  # default wrapper; overridden below
    use_split = False  # when True, body is wrapped in \begin{split}...\end{split}

    if style == 0:
        lhs = rng.choice(["f(x)", "g(t)", "y", "I", "S"])
        n_lines = rng.randint(2, 3)
        lines = [rf"{lhs} &= {_expr(rng, 2)}"]
        for _ in range(n_lines - 1):
            lines.append(rf"&= {_expr(rng, 2)}")

    elif style == 1:
        a, b, c, d = _s(rng), _s(rng), _s(rng), _s(rng)
        r1, r2 = _s(rng), _s(rng)
        lines = [
            rf"{a} x + {b} y &= {r1}",
            rf"{c} x + {d} y &= {r2}",
        ]

    elif style == 2:
        lhs = rng.choice([_v(rng), "y", "f(x)"])
        lines = [
            rf"{lhs} &= {_expr(rng, 2)} + {_expr(rng, 1)}",
            rf"&= {_expr(rng, 2)}",
        ]

    elif style == 3:
        v = _v(rng)
        fn = rng.choice([rf"\frac{{d}}{{d{v}}}", rf"\frac{{d^2}}{{d{v}^2}}"])
        lines = [
            rf"{fn}\left[{_expr(rng, 2)}\right] &= {_expr(rng, 2)}",
            rf"&= {_expr(rng, 2)}",
        ]

    elif style == 4:
        a, b, c = _expr(rng, 1), _expr(rng, 1), _expr(rng, 1)
        op = rng.choice([r"\leq", r"\geq"])
        lines = [
            rf"{a} &{op} {b}",
            rf"&{op} {c}",
        ]

    elif style == 5:
        n = rng.choice(["n", "N"])
        lines = [
            rf"\sum_{{k=1}}^{{{n}}} a_k &= a_1 + a_2 + \cdots + a_{{{n}}}",
            rf"&= {_expr(rng, 2)}",
        ]

    elif style == 6:
        # piecewise / cases — rendered inside equation*
        v = _v(rng)
        e1, e2 = _expr(rng, 1), _expr(rng, 1)
        cond = rng.choice(["0", _s(rng), r"\pi"])
        body = (
            rf"\begin{{cases}} "
            rf"{e1} & \text{{if }} {v} \geq {cond} \\"
            rf" {e2} & \text{{if }} {v} < {cond} "
            rf"\end{{cases}}"
        )
        lines = [rf"f({v}) &= {body}"]
        env = "align*"

    elif style == 7:
        v = _v(rng)
        e1, e2, e3 = _expr(rng, 1), _expr(rng, 1), _expr(rng, 1)
        body = (
            rf"\begin{{cases}} "
            rf"{e1} & \text{{if }} {v} > 0 \\"
            rf" {e2} & \text{{if }} {v} = 0 \\"
            rf" {e3} & \text{{if }} {v} < 0 "
            rf"\end{{cases}}"
        )
        lines = [rf"g({v}) &= {body}"]
        env = "align*"

    elif style == 8:
        e1, e2 = _expr(rng, 1), _expr(rng, 1)
        n = rng.choice(["n", "N"])
        ub = _underbrace(rf"a_1 + a_2 + \cdots + a_{{{n}}}", rf"{n} \text{{ terms}}")
        lines = [
            rf"S_{{{n}}} &= {ub}",
            rf"&= {e1} + {e2}",
        ]

    elif style == 9:
        a, b, c = _s(rng), _s(rng), _s(rng)
        r1, r2, r3 = _s(rng), _s(rng), _s(rng)
        lines = [
            rf"{a} x + {b} y + {c} z &= {r1}",
            rf"{_s(rng)} x + {_s(rng)} y &= {r2}",
            rf"{_s(rng)} z &= {r3}",
        ]

    elif style == 10:
        n = rng.choice(["n", "3", "4"])
        ob = _overbrace(r"a^2 + 2ab + b^2", r"(a+b)^2")
        lines = [
            rf"(a+b)^{{{n}}} &= {ob} \cdot (a+b)^{{{n}-2}}",
            rf"&= {_expr(rng, 1)}",
        ]

    elif style == 11:
        rel = rng.choice([r"\leq", r"\geq", r"\ll"])
        e1, e2, e3 = _expr(rng, 1), _expr(rng, 1), _expr(rng, 1)
        lines = [
            rf"{e1} &{rel} {e2}",
            rf"&{rel} {e3}",
        ]

    elif style == 12:
        # Invisible-bracket split: \left( ... \right. / \left. ... \right)
        lhs = rng.choice(["f(x)", "g(t)", "y", "S"])
        delim_open, delim_close = rng.choice([("(", ")"), ("[", "]")])
        e1, e2 = _expr(rng, 2), _expr(rng, 2)
        e3, e4 = _expr(rng, 1), _expr(rng, 1)
        lines = [
            rf"{lhs} &= \left{delim_open} {e1} + {e2} + \cdots \right.",
            rf"&\left. \quad + {e3} + {e4} \right{delim_close}",
        ]

    elif style == 13:
        # Derivative bracket spanning two rows, evaluated at bounds
        v = _v(rng)
        fn = rng.choice([rf"\frac{{d}}{{d{v}}}", rf"\frac{{d^2}}{{d{v}^2}}"])
        a, b = _s(rng), _s(rng)
        e1, e2, e3, e4 = _expr(rng, 2), _expr(rng, 1), _expr(rng, 1), _expr(rng, 1)
        lines = [
            rf"I &= \left[ {fn}\left( {e1} \right) + {e2} \right.",
            rf"&\left. \quad - {e3} \cdot {e4} \right]_{{{a}}}^{{{b}}}",
        ]

    elif style == 14:
        # Grouped expression with coefficient spanning two rows
        lhs = rng.choice(["y", "z", "w"])
        c = _s(rng)
        e1, e2 = _expr(rng, 2), _expr(rng, 2)
        e3, e4, e5, e6 = _expr(rng, 1), _expr(rng, 1), _expr(rng, 1), _expr(rng, 1)
        lines = [
            rf"{lhs} &= {c} \left( \frac{{{e1}}}{{{e2}}} + {e3} \right.",
            rf"&\left. \qquad + \frac{{{e4}}}{{{e5}}} \right) + {e6}",
        ]

    # --- multline* (styles 15–16) and multline numbered (style 17) ---

    elif style == 15:
        # multline*: long polynomial split across two lines
        env = "multline*"
        v = _v(rng)
        poly = _poly(rng, v, max_degree=rng.randint(4, 6))
        # Split the polynomial at the midpoint by inserting a line break
        terms = poly.split("+")
        mid = max(1, len(terms) // 2)
        first = "+".join(terms[:mid]).rstrip()
        rest = "+".join(terms[mid:]).lstrip()
        lines = [first, rf"\quad + {rest}"]

    elif style == 16:
        # multline*: summation expansion split across two lines
        env = "multline*"
        n = rng.choice(["n", "N"])
        e1, e2, e3, e4 = _expr(rng, 1), _expr(rng, 1), _expr(rng, 1), _expr(rng, 1)
        lhs = rng.choice(["f(x)", "S", "T", "I"])
        lines = [
            rf"{lhs} = {e1} + {e2} + \cdots",
            rf"\quad + {e3} + {e4}",
        ]

    elif style == 17:
        # multline (numbered): same polynomial split as style 15
        env = "multline"
        v = _v(rng)
        poly = _poly(rng, v, max_degree=rng.randint(4, 6))
        terms = poly.split("+")
        mid = max(1, len(terms) // 2)
        first = "+".join(terms[:mid]).rstrip()
        rest = "+".join(terms[mid:]).lstrip()
        lines = [first, rf"\quad + {rest}"]

    # --- gather* (style 18) and gather numbered (style 19) ---

    elif style == 18:
        # gather*: 2–3 independent centered equations, no alignment column
        env = "gather*"
        n_eqns = rng.randint(2, 3)
        eqn_lines = []
        for _ in range(n_eqns):
            lhs = rng.choice([_v(rng), "f(x)", "g(t)", "y"])
            eqn_lines.append(rf"{lhs} = {_expr(rng, 2)}")
        lines = eqn_lines

    elif style == 19:
        # gather (numbered): same structure, numbered
        env = "gather"
        n_eqns = rng.randint(2, 3)
        eqn_lines = []
        for _ in range(n_eqns):
            lhs = rng.choice([_v(rng), "f(x)", "g(t)", "y"])
            eqn_lines.append(rf"{lhs} = {_expr(rng, 2)}")
        lines = eqn_lines

    # --- equation + split (styles 20–21) ---

    elif style == 20:
        # equation + split: algebraic derivation (single equation number)
        env = "equation"
        use_split = True
        lhs = rng.choice(["f(x)", "g(t)", "y", "S", "I"])
        lines = [
            rf"{lhs} &= {_expr(rng, 2)} + {_expr(rng, 1)}",
            rf"&= {_expr(rng, 2)}",
            rf"&= {_expr(rng, 1)}",
        ]

    elif style == 21:
        # equation + split: calculus derivation
        env = "equation"
        use_split = True
        v = _v(rng)
        a, b = _s(rng), _s(rng)
        e1, e2, e3 = _expr(rng, 1), _expr(rng, 1), _expr(rng, 1)
        lines = [
            rf"\int_{{{a}}}^{{{b}}} {e1} \, d{v} &= \left[ {e2} \right]_{{{a}}}^{{{b}}}",
            rf"&= {e3}",
        ]

    # --- multi-column align* (styles 22–24) ---

    elif style == 22:
        # align*: 3 columns × 3 rows of simple scalar equalities
        env = "align*"
        rows = []
        for _ in range(3):
            v1, v2, v3 = _v(rng), _v(rng), _v(rng)
            c1, c2, c3 = _s(rng), _s(rng), _s(rng)
            rows.append(rf"{v1} &= {c1} && {v2} &= {c2} && {v3} &= {c3}")
        lines = rows

    elif style == 23:
        # align*: 3 columns × 2 rows of mixed expressions
        env = "align*"
        rows = []
        for _ in range(2):
            v1, v2, v3 = _v(rng), _v(rng), _v(rng)
            e1, e2, e3 = _expr(rng, 1), _expr(rng, 1), _expr(rng, 1)
            rows.append(rf"{v1} &= {e1} && {v2} &= {e2} && {v3} &= {e3}")
        lines = rows

    elif style == 24:
        # align*: 2 columns × 3 rows of inequality chains
        env = "align*"
        rel = rng.choice([r"\leq", r"\geq"])
        rows = []
        for _ in range(3):
            v1, v2 = _v(rng), _v(rng)
            e1, e2 = _expr(rng, 1), _expr(rng, 1)
            rows.append(rf"{v1} &{rel} {e1} && {v2} &{rel} {e2}")
        lines = rows

    # --- labeled equation environments (styles 25–26) ---

    elif style == 25:
        # equation with \label: single expression
        env = "equation"
        lbl = rng.choice(_LABELS)
        e = _expr(rng, 2)
        # Assembly is handled specially below via early return
        return rf"\begin{{equation}}\label{{{lbl}}} {e} \end{{equation}}"

    else:  # style == 26
        # align (numbered) with \label on the first row
        env = "align"
        lbl = rng.choice(_LABELS)
        lhs = rng.choice(["f(x)", "g(t)", "y", "S"])
        lines = [
            rf"\label{{{lbl}}} {lhs} &= {_expr(rng, 2)}",
            rf"&= {_expr(rng, 1)}",
        ]

    body = r" \\".join(lines)
    if use_split:
        inner = rf"\begin{{split}}{body}\end{{split}}"
        return rf"\begin{{{env}}}{inner}\end{{{env}}}"
    return rf"\begin{{{env}}}{body}\end{{{env}}}"
