"""Multi-line LaTeX align* and cases environment generator."""

from __future__ import annotations

import random

from ._vocab import _expr, _overbrace, _s, _underbrace, _v


def _align(rng: random.Random) -> str:
    """Generate a multi-line LaTeX block (align* or equation*/cases)."""
    style = rng.randint(0, 11)
    env = "align*"  # default wrapper; overridden below for cases styles

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

    else:  # style == 11
        rel = rng.choice([r"\leq", r"\geq", r"\ll"])
        e1, e2, e3 = _expr(rng, 1), _expr(rng, 1), _expr(rng, 1)
        lines = [
            rf"{e1} &{rel} {e2}",
            rf"&{rel} {e3}",
        ]

    body = r" \\".join(lines)
    return rf"\begin{{{env}}}{body}\end{{{env}}}"
