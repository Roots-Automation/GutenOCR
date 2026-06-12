"""Per-domain KaTeX render success-rate tests.

Requires Node.js and the katex npm package.  Skip gracefully if either is absent.

Run with:
    pytest tests/test_render_katex_ci.py -m render -v
"""

from __future__ import annotations

import random
import shutil

import pytest
from formula_combinatorics.domains import GENERATORS
from formula_combinatorics.engine.render import _check_katex, _KatexRenderer, _strip_display_delimiters

RENDER_N = 30
RENDER_FLOOR = 0.90

_NODE_BIN = "node"


def _node_available() -> bool:
    return shutil.which(_NODE_BIN) is not None


def _katex_available() -> bool:
    return _node_available() and _check_katex(_NODE_BIN)


_SKIP_REASON = "node or katex npm package not available (install node + npm install katex)"
_requires_katex = pytest.mark.skipif(not _katex_available(), reason=_SKIP_REASON)


@pytest.fixture(scope="module")
def katex_renderer():
    if not _katex_available():
        pytest.skip(_SKIP_REASON)
    return _KatexRenderer(node_bin=_NODE_BIN)


@pytest.mark.render
@_requires_katex
@pytest.mark.parametrize("domain", sorted(GENERATORS.keys()))
def test_katex_render_rate(domain: str, katex_renderer: _KatexRenderer) -> None:
    rng = random.Random(42)
    gen = GENERATORS[domain]
    formulas = [_strip_display_delimiters(gen(rng)) for _ in range(RENDER_N)]
    results = katex_renderer.validate_batch(formulas)
    n_ok = sum(1 for ok, _ in results if ok)
    rate = n_ok / RENDER_N
    assert rate >= RENDER_FLOOR, (
        f"domain={domain!r}: KaTeX success rate {rate:.0%} ({n_ok}/{RENDER_N}) < {RENDER_FLOOR:.0%}. "
        f"First failures: {[err for ok, err in results if not ok][:3]}"
    )
