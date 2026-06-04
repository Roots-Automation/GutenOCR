# formula-combinatorics

Synthetic LaTeX mathematical formula generator for OCR training data. Produces mathematically realistic LaTeX strings across 25 math domains using hand-crafted, domain-specific generators. All output is synthetic — no third-party content is used.

Output is a JSON object compatible with `GutenOCR/data/grounded_latex/generate_equations.py`.

---

## Installation

From within this directory:

```bash
pip install -e .
# or with uv:
uv sync
```

This installs the `formula-combinatorics` package and the `formula-generate` CLI command.

The package has no required third-party dependencies — it uses only the stdlib. If `roots-ocr` is already installed in your environment it will be used for richer CLI logging; otherwise a stdlib `basicConfig` fallback is used transparently.

---

## CLI Usage

```bash
formula-generate --output formulas.json --count 50000 --seed 42
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--output PATH` | *(required)* | Output JSON file path |
| `--count N` | `100000` | Number of unique formulas to generate |
| `--seed INT` | `None` | Random seed for reproducibility |
| `--align-fraction F` | `0.15` | Fraction of output using multi-line `align*` environments |
| `--domains D [D ...]` | all | Restrict to specific domains (see list below) |

### Examples

Generate 10,000 formulas from all domains with a fixed seed:
```bash
formula-generate --output train.json --count 10000 --seed 0
```

Generate only calculus and probability formulas:
```bash
formula-generate --output calc_prob.json --count 5000 --domains calculus probability
```

Generate with a higher proportion of multi-line expressions:
```bash
formula-generate --output wide.json --count 10000 --align-fraction 0.30
```

---

## Output Format

```json
{
  "0": "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}",
  "1": "P(X = k) = \\frac{\\lambda^k e^{-\\lambda}}{k!}",
  "2": "\\begin{align*}f(x) &= \\frac{\\alpha}{\\beta} + x^2 \\\\&= \\sqrt{x + 1}\\end{align*}"
}
```

Keys are sequential string integers starting at `"0"`. Values are valid LaTeX strings, either single-line inline expressions or multi-line `align*` / `cases` environments.

---

## Python API

```python
from formula_combinatorics import generate, GENERATORS, DEFAULT_WEIGHTS

# Generate 1000 formulas across all domains
corpus = generate(
    count=1000,
    domains=list(GENERATORS.keys()),
    generators=GENERATORS,
    weights=DEFAULT_WEIGHTS,
    seed=42,
    align_fraction=0.15,
)
# corpus == {"0": "...", "1": "...", ...}
```

### Calling individual domain generators

Each domain generator is a plain function `(rng: random.Random) -> str`:

```python
import random
from formula_combinatorics.domains import GENERATORS

rng = random.Random(0)

# Single domain
formula = GENERATORS["calculus"](rng)

# Specific domain modules expose the same interface
from formula_combinatorics.domains.probability import _probability
formula = _probability(rng)
```

### Using vocabulary and template helpers directly

```python
from formula_combinatorics._vocab import _atom, _expr, _v, _g
from formula_combinatorics._templates import _def_integral, _limit, _partial_deriv

rng = random.Random(0)

expr = _expr(rng, depth=2)            # recursive LaTeX expression
integral = _def_integral(rng)          # \int_{lo}^{hi} expr dv
limit = _limit(rng)                    # \lim_{v \to pt} expr
pd = _partial_deriv(rng, order=2)      # \frac{\partial^2 f}{\partial v^2}
```

---

## Domains

25 domains are available, organized into thematic modules. Default sampling weights reflect approximate prevalence in mathematical OCR corpora.

| Module file | Domains | Default weight |
|---|---|---|
| `domains/algebra.py` | `algebra` | 9% |
| | `trigonometry` | 4% |
| `domains/calculus.py` | `calculus` | 10% |
| | `analysis` | 4% |
| | `differential_equations` | 3% |
| `domains/linear_algebra.py` | `linear_algebra` | 8% |
| `domains/probability.py` | `probability` | 9% |
| | `information_theory` | 3% |
| `domains/discrete.py` | `number_theory` | 4% |
| | `combinatorics` | 3% |
| | `graph_theory` | 2% |
| `domains/group_theory.py` | `group_theory` | 3% |
| | `concrete_groups` | 1% |
| `domains/ring_theory.py` | `ring_field_theory` | 1% |
| `domains/representation.py` | `representation_theory` | 1% |
| `domains/geometry.py` | `differential_geometry` | 2% |
| | `topology` | 4% |
| `domains/signals.py` | `complex_analysis` | 3% |
| | `fourier` | 2% |
| `domains/physics.py` | `physics` | 6% |
| `domains/measure.py` | `measure_theory` | 2% |
| | `p_adic` | 2% |
| `domains/optimization.py` | `optimization` | 5% |
| `domains/foundations.py` | `set_theory` | 5% |
| | `logic` | 4% |

Sampling weights are renormalized automatically when `--domains` restricts the active set, so partial runs produce the correct relative distribution.

An additional `align_fraction` (default 15%) of output uses multi-line `align*` or `cases` environments drawn from a separate pool of 12 structural templates independent of domain.

---

## Adding a New Domain

1. Create a new file in `domains/`, e.g. `domains/topology_advanced.py`.

2. Define a generator function and expose `GENERATORS` and `WEIGHTS` dicts:

```python
# domains/topology_advanced.py
import random
from collections.abc import Callable

def _my_domain(rng: random.Random) -> str:
    c = rng.randint(0, 2)
    if c == 0:
        return r"\pi_1(S^1) \cong \mathbb{Z}"
    if c == 1:
        return r"H^n(M; \mathbb{Z}) \cong H_n(M; \mathbb{Z})"
    return r"\chi(S^2) = 2"

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "topology_advanced": _my_domain,
}

WEIGHTS: dict[str, float] = {
    "topology_advanced": 0.02,
}
```

3. Import and merge in `domains/__init__.py`:

```python
from .topology_advanced import GENERATORS as _G_TOPO_ADV, WEIGHTS as _W_TOPO_ADV

GENERATORS = {
    ...existing...,
    **_G_TOPO_ADV,
}

DEFAULT_WEIGHTS = {
    ...existing...,
    **_W_TOPO_ADV,
}
```

The import-time assertion `assert set(DEFAULT_WEIGHTS) == set(GENERATORS)` will catch any mismatch immediately.

4. Adjust the total weights so they sum to 1.0 (the engine renormalizes, but clean weights are easier to read).

### Using shared helpers

Pull from `_vocab.py` and `_templates.py` to avoid reinventing common patterns:

```python
from .._vocab import _v, _g, _s, _atom, _expr, _two, _SCALARS
from .._templates import _def_integral, _limit, _norm, _partial_deriv, _matrix_env
```

---

## Module Layout

```
formula_combinatorics/
├── pyproject.toml          # package metadata and CLI entry point
├── __init__.py             # public API: generate, GENERATORS, DEFAULT_WEIGHTS
├── generate.py             # CLI entry point (main())
├── corpus.py               # generation engine: dedup loop, exception handling
├── align.py                # multi-line align* / cases environment builder
├── _vocab.py               # shared constants and atomic sampling helpers
├── _templates.py           # shared LaTeX fragment builders
└── domains/
    ├── __init__.py         # merged GENERATORS + DEFAULT_WEIGHTS registry
    ├── algebra.py
    ├── calculus.py
    ├── discrete.py
    ├── foundations.py
    ├── geometry.py
    ├── group_theory.py
    ├── linear_algebra.py
    ├── measure.py
    ├── optimization.py
    ├── physics.py
    ├── probability.py
    ├── representation.py
    ├── ring_theory.py
    └── signals.py
```
