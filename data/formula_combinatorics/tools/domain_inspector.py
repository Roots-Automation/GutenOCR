#!/usr/bin/env python3
"""
Formula Domain Inspector

Samples N formulas from each domain and renders them into a self-contained HTML
file using MathJax for visual inspection. Use this to spot diversity gaps,
rendering issues, and notation problems across domains.

Usage:
    python domain_inspector.py                          # 8 samples/domain, open in browser
    python domain_inspector.py --samples 15            # more samples
    python domain_inspector.py --seed 42               # reproducible output
    python domain_inspector.py --domains calculus algebra  # specific domains only
    python domain_inspector.py --output report.html --no-open  # save without opening
"""

from __future__ import annotations

import argparse
import html
import random
import sys
import webbrowser
from pathlib import Path

# Allow running as a plain script from inside the package directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formula_combinatorics.domains import DEFAULT_WEIGHTS, GENERATORS

_align = GENERATORS["align"]

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Formula Domain Inspector</title>
<script>
MathJax = {
  tex: {
    inlineMath: [['$','$']], displayMath: [['$$','$$']],
    macros: {
      textcircled: ["\\mathord{\\bigcirc\\!\\!\\!\\!\\raise{0.05em}{\\scriptstyle\\text{#1}}}", 1]
    }
  },
  options: { skipHtmlTags: ['script','noscript','style','textarea','pre'] }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f1117;
    color: #e2e8f0;
    margin: 0;
    padding: 0;
  }
  header {
    background: #1a1d2e;
    border-bottom: 1px solid #2d3248;
    padding: 1.2rem 2rem;
    position: sticky;
    top: 0;
    z-index: 100;
    display: flex;
    align-items: center;
    gap: 1.5rem;
  }
  header h1 { margin: 0; font-size: 1.2rem; color: #a5b4fc; }
  header .meta { font-size: 0.8rem; color: #64748b; }
  .toc {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    padding: 1rem 2rem;
    border-bottom: 1px solid #1e2133;
    background: #12151f;
  }
  .toc a {
    font-size: 0.75rem;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    background: #1e2133;
    color: #94a3b8;
    text-decoration: none;
    transition: background 0.15s;
  }
  .toc a:hover { background: #2d3248; color: #e2e8f0; }
  .domain-section {
    padding: 1.5rem 2rem 0.5rem;
    border-bottom: 1px solid #1a1d2e;
  }
  .domain-header {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    margin-bottom: 1rem;
  }
  .domain-header h2 {
    margin: 0;
    font-size: 1rem;
    color: #818cf8;
    font-weight: 600;
  }
  .domain-header .weight {
    font-size: 0.72rem;
    color: #475569;
  }
  .formula-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1rem;
  }
  .formula-card {
    background: #1a1d2e;
    border: 1px solid #2d3248;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }
  .formula-card:hover {
    border-color: #4f46e5;
  }
  .rendered {
    min-height: 2.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #f1f5f9;
    font-size: 1rem;
    overflow-x: auto;
    padding: 0.25rem 0;
  }
  .source {
    font-family: "Fira Code", "Cascadia Code", monospace;
    font-size: 0.68rem;
    color: #64748b;
    background: #0f1117;
    border-radius: 4px;
    padding: 0.4rem 0.6rem;
    word-break: break-all;
    white-space: pre-wrap;
    border: 1px solid #1e2233;
    user-select: all;
  }
  .align-section {
    padding: 1.5rem 2rem 0.5rem;
    border-bottom: 1px solid #1a1d2e;
  }
  .align-section h2 {
    font-size: 1rem;
    color: #34d399;
    font-weight: 600;
    margin: 0 0 1rem;
  }
  .index { color: #334155; font-size: 0.65rem; margin-bottom: 0.15rem; }
  footer {
    padding: 2rem;
    text-align: center;
    font-size: 0.75rem;
    color: #334155;
  }
</style>
</head>
<body>
"""

_HTML_TAIL = """\
<footer>formula-combinatorics inspector &mdash; {n_total} formulas across {n_domains} domains</footer>
</body></html>
"""


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _card(idx: int, latex: str) -> str:
    """Render one formula card (rendered MathJax + raw source)."""
    escaped = html.escape(latex)
    # \begin{...} environments are already display-math; others need $$ wrapping
    if latex.startswith(r"\begin"):
        display = escaped
    else:
        display = f"$${escaped}$$"
    return (
        f'<div class="formula-card">'
        f'<div class="index">#{idx}</div>'
        f'<div class="rendered">{display}</div>'
        f'<div class="source">{escaped}</div>'
        f"</div>"
    )


def _domain_section(
    domain: str,
    weight: float,
    samples: list[str],
) -> str:
    anchor = domain.replace("_", "-")
    cards = "\n".join(_card(i + 1, s) for i, s in enumerate(samples))
    return (
        f'<div class="domain-section" id="{anchor}">'
        f'<div class="domain-header">'
        f"<h2>{domain}</h2>"
        f'<span class="weight">weight {weight:.0%}</span>'
        f"</div>"
        f'<div class="formula-grid">{cards}</div>'
        f"</div>\n"
    )


def _align_section(samples: list[str]) -> str:
    cards = "\n".join(_card(i + 1, s) for i, s in enumerate(samples))
    return (
        f'<div class="align-section" id="align">'
        f"<h2>align* / cases environments</h2>"
        f'<div class="formula-grid">{cards}</div>'
        f"</div>\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_html(
    domains: list[str],
    samples_per_domain: int,
    align_samples: int,
    rng: random.Random,
) -> str:
    sections = []

    # Domain sections
    for domain in domains:
        gen = GENERATORS[domain]
        samples = [gen(rng) for _ in range(samples_per_domain)]
        weight = DEFAULT_WEIGHTS.get(domain, 0.0)
        sections.append(_domain_section(domain, weight, samples))

    # Align section
    align_out = [_align(rng) for _ in range(align_samples)]
    sections.append(_align_section(align_out))

    # TOC
    toc_links = "".join(f'<a href="#{d.replace("_", "-")}">{d}</a>' for d in domains)
    toc_links += '<a href="#align">align*</a>'
    toc = f'<div class="toc">{toc_links}</div>\n'

    n_total = len(domains) * samples_per_domain + align_samples
    header = (
        f"<header>"
        f"<h1>Formula Domain Inspector</h1>"
        f'<span class="meta">{len(domains)} domains &middot; '
        f"{samples_per_domain} samples each &middot; "
        f"{align_samples} align* &middot; "
        f"seed {rng.getstate()[1][0]}</span>"
        f"</header>\n"
    )

    tail = _HTML_TAIL.format(n_total=n_total, n_domains=len(domains))
    return _HTML_HEAD + header + toc + "".join(sections) + tail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=8, metavar="N", help="Samples per domain (default: 8)")
    parser.add_argument(
        "--align-samples", type=int, default=12, metavar="N", help="align* environment samples (default: 12)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(DEFAULT_WEIGHTS.keys()),
        choices=list(GENERATORS.keys()),
        metavar="DOMAIN",
        help="Domains to include (default: all)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("formula_inspect.html"), metavar="PATH", help="Output HTML file"
    )
    parser.add_argument("--no-open", action="store_true", help="Don't open the browser automatically")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"Sampling {args.samples} formulas × {len(args.domains)} domains + {args.align_samples} align* ...")
    html_content = build_html(args.domains, args.samples, args.align_samples, rng)

    args.output.write_text(html_content, encoding="utf-8")
    print(f"Written → {args.output.resolve()}")

    if not args.no_open:
        webbrowser.open(args.output.resolve().as_uri())


if __name__ == "__main__":
    main()
