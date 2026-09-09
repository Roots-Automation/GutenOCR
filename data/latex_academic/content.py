"""SyntheticContent: CC0-clean academic document content generation."""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from faker import Faker
from layout import LayoutConfig

logger = logging.getLogger(__name__)

_RESOURCES = Path(__file__).parent / "resources"
_WORDLISTS = _RESOURCES / "wordlists"

_CORPUS_PATH = _RESOURCES / "corpus_sentences.txt"
_LOREM_PATH = _RESOURCES / "lorem_ipsum.txt"

_LATEX_ESCAPE_MAP: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "^": r"\^{}",
    "~": r"\textasciitilde{}",
}


def escape_latex(s: str) -> str:
    """Escape LaTeX special characters, processing each character exactly once."""
    return "".join(_LATEX_ESCAPE_MAP.get(ch, ch) for ch in s)


def _load_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_sentences() -> list[str]:
    if _CORPUS_PATH.exists():
        return [escape_latex(s) for s in _load_lines(_CORPUS_PATH)]
    logger.warning(
        "corpus_sentences.txt not found; falling back to lorem ipsum. "
        "Run scripts/fetch_corpus.py to build the real-text corpus."
    )
    return [
        s.strip()
        for para in _LOREM_PATH.read_text(encoding="utf-8").splitlines()
        for s in para.replace(". ", ".\n").splitlines()
        if s.strip()
    ]


_SECTION_HEADINGS: list[str] = _load_lines(_WORDLISTS / "section_headings.txt")
_ADJECTIVES: list[str] = _load_lines(_WORDLISTS / "academic_adjectives.txt")
_NOUNS: list[str] = _load_lines(_WORDLISTS / "field_nouns.txt")
_EQUATIONS_INLINE: list[str] = _load_lines(_WORDLISTS / "equations_inline.txt")
_EQUATIONS_DISPLAY: list[str] = _load_lines(_WORDLISTS / "equations_display.txt")
_THEOREM_BODIES: list[str] = _load_lines(_WORDLISTS / "theorem_bodies.txt")
_CORPUS_SENTENCES: list[str] = _load_sentences()

_FAKE = Faker()

# Multi-line align-environment math blocks (without \begin/\end wrappers)
_ALIGN_BLOCKS: list[str] = [
    r"  f(\mathbf{x} + \delta) &= f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \delta \\" + "\n"
    r"  &\quad + \tfrac{1}{2}\delta^\top H(\mathbf{x})\,\delta + O(\|\delta\|^3)",
    r"  \mathbb{E}[X^2] &= \mathrm{Var}(X) + (\mathbb{E}[X])^2 \\" + "\n"
    r"  &= \sigma^2 + \mu^2",
    r"  \ell(\theta) &= \sum_{i=1}^n \log p(x_i \mid \theta) \\" + "\n"
    r"  &= -\tfrac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2",
    r"  \|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 &\leq \|\mathbf{x}_t - \eta\nabla f(\mathbf{x}_t) - \mathbf{x}^*\|^2 \\"
    + "\n"
    r"  &\leq (1 - 2\eta\alpha + \eta^2 L^2)\|\mathbf{x}_t - \mathbf{x}^*\|^2",
    r"  \mathrm{KL}(P \| Q) &= \sum_x P(x)\log\frac{P(x)}{Q(x)} \\" + "\n"
    r"  &= \mathbb{E}_P\!\left[\log P(X) - \log Q(X)\right] \geq 0",
    r"  \hat{\mathbf{w}} &= \operatorname*{arg\,min}_{\mathbf{w}}\;\frac{1}{n}\sum_{i=1}^n(y_i - \mathbf{w}^\top\mathbf{x}_i)^2 \\"
    + "\n"
    r"  &= (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}",
    r"  \sigma(x) &= \frac{1}{1+e^{-x}} \\" + "\n"
    r"  \sigma'(x) &= \sigma(x)\bigl(1-\sigma(x)\bigr)",
    r"  H(X,Y) &= H(X) + H(Y \mid X) \\" + "\n"
    r"  &= H(Y) + H(X \mid Y) \leq H(X) + H(Y)",
    r"  \|\mathbf{A}+\mathbf{B}\|_F &\leq \|\mathbf{A}\|_F + \|\mathbf{B}\|_F \\" + "\n"
    r"  \|\mathbf{A}\mathbf{B}\|_F &\leq \|\mathbf{A}\|_F\|\mathbf{B}\|_F",
    r"  p(\mathbf{z} \mid \mathbf{x}) &= \frac{p(\mathbf{x} \mid \mathbf{z})\,p(\mathbf{z})}{p(\mathbf{x})} \\" + "\n"
    r"  &\propto p(\mathbf{x} \mid \mathbf{z})\,p(\mathbf{z})",
]

_ALGO_VARS: list[str] = list("xyzwuv")
_ALGO_LOOP_VARS: list[str] = list("ijtk")
_ALGO_ACCUMS: list[str] = [
    r"\mathit{S}",
    r"\mathit{best}",
    r"\mathit{result}",
    r"\mathit{total}",
]
_ALGO_THRESHOLDS: list[str] = [r"\theta", r"\epsilon", r"0", r"\tau", r"\delta"]
_ALGO_UPDATES: list[str] = [
    "{acc} \\gets {acc} + {v}_{{{lv}}}",
    "{acc} \\gets \\max({acc},\\; {v}_{{{lv}}})",
    "{acc} \\gets {acc} + \\alpha \\cdot \\nabla f({v}_{{{lv}}})",
    "{acc} \\gets {acc} \\cdot (1 - \\eta) + \\eta \\cdot {v}_{{{lv}}}",
]

_THEOREM_ENVS: list[str] = [
    "theorem",
    "lemma",
    "definition",
    "corollary",
    "proposition",
    "remark",
]

_MATH_SPAN_RE = re.compile(r"\$[^$]+\$")


def _extract_math_tokens(text: str) -> list[str]:
    return _MATH_SPAN_RE.findall(text)


# ---------------------------------------------------------------------------
# Content dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ListContent:
    items: list[str]
    ordered: bool


@dataclass
class TheoremContent:
    env_type: str
    body: str


@dataclass
class AlgorithmContent:
    caption: str
    lines: list[str]
    idx: int


@dataclass
class FigureContent:
    height_cm: float
    caption: str
    idx: int


@dataclass
class SectionContent:
    heading: str
    body: str
    display_equation: str | None
    align_block: str | None
    table_rows: list[str] | None
    table_header: str | None
    table_spec: str | None
    table_caption: str | None
    list_content: ListContent | None
    theorem: TheoremContent | None
    algorithm: AlgorithmContent | None
    figure: FigureContent | None
    inline_math_tokens: list[str] = field(default_factory=list)


@dataclass
class ReferenceEntry:
    key: str
    text: str


@dataclass
class SyntheticContent:
    title: str
    authors: list[str]
    year: int
    abstract: str
    sections: list[SectionContent]
    references: list[ReferenceEntry]
    # Layout mirror fields (needed by Jinja2 template)
    n_cols: int
    font_package: str
    font_size_pt: int
    margins_mm: int
    page_size: str
    line_spread: float
    has_abstract: bool
    has_page_numbers: bool

    @property
    def full_text(self) -> str:
        """Concatenated text for content-hash split assignment."""
        parts = [self.title, self.abstract]
        for sec in self.sections:
            parts.append(sec.heading)
            parts.append(sec.body)
        return " ".join(parts)

    def math_token_sequence(self) -> list[str]:
        """All $...$ math tokens across the document, in reading order."""
        tokens: list[str] = []
        tokens.extend(_extract_math_tokens(self.abstract))
        for sec in self.sections:
            tokens.extend(sec.inline_math_tokens)
        return tokens

    @classmethod
    def generate(cls, layout: LayoutConfig, rng: random.Random) -> SyntheticContent:
        title = _make_title(rng)
        authors = [_make_author(rng) for _ in range(rng.randint(1, 5))]
        year = rng.randint(2018, 2025)
        abstract = _make_paragraph(rng, n_sentences=rng.randint(3, 6)) if layout.has_abstract else ""

        # Generate reference keys first so body text can cite them
        n_refs = rng.randint(5, 15) if layout.has_references else 0
        ref_keys = [f"ref{i}" for i in range(n_refs)]

        table_budget = layout.tables_per_doc
        algo_budget = layout.algorithms_per_doc
        figure_budget = layout.figures_per_doc
        theorem_budget = layout.theorem_envs_per_doc
        list_budget = layout.lists_per_doc

        algo_idx = 0
        figure_idx = 0
        table_idx = 0

        sections: list[SectionContent] = []
        used_headings: set[str] = set()

        for _ in range(layout.n_sections):
            heading = _pick_heading(rng, used_headings)
            used_headings.add(heading)

            body = _make_body(
                rng,
                n_paragraphs=rng.randint(1, 3),
                inline_equations_per_para=layout.equations_per_section,
                ref_keys=ref_keys,
            )

            # Math display: pick one of single-line, align, or none
            display_equation: str | None = None
            align_block: str | None = None
            math_roll = rng.random()
            if math_roll < 0.25:
                align_block = _make_align_equation(rng)
            elif math_roll < 0.60:
                display_equation = rng.choice(_EQUATIONS_DISPLAY)

            # Table
            table_rows, table_header, table_spec, table_caption = None, None, None, None
            if table_budget > 0 and rng.random() < 0.4:
                table_rows, table_header, table_spec, table_caption = _make_table(rng, table_idx)
                table_idx += 1
                table_budget -= 1

            # List
            list_content: ListContent | None = None
            if list_budget > 0 and rng.random() < 0.45:
                list_content = _make_list(rng, ordered=rng.random() < 0.4, ref_keys=ref_keys)
                list_budget -= 1

            # Theorem/lemma/definition
            theorem: TheoremContent | None = None
            if theorem_budget > 0 and rng.random() < 0.40:
                theorem = _make_theorem(rng)
                theorem_budget -= 1

            # Algorithm box
            algorithm: AlgorithmContent | None = None
            if algo_budget > 0 and rng.random() < 0.35:
                algorithm = _make_algorithm(rng, algo_idx)
                algo_idx += 1
                algo_budget -= 1

            # Figure placeholder
            figure: FigureContent | None = None
            if figure_budget > 0 and rng.random() < 0.35:
                figure = _make_figure(rng, figure_idx)
                figure_idx += 1
                figure_budget -= 1

            sec_math: list[str] = []
            sec_math.extend(_extract_math_tokens(body))
            if theorem:
                sec_math.extend(_extract_math_tokens(theorem.body))
            if list_content:
                for item in list_content.items:
                    sec_math.extend(_extract_math_tokens(item))
            if algorithm:
                for ln in algorithm.lines:
                    sec_math.extend(_extract_math_tokens(ln))

            sections.append(
                SectionContent(
                    heading=heading,
                    body=body,
                    display_equation=display_equation,
                    align_block=align_block,
                    table_rows=table_rows,
                    table_header=table_header,
                    table_spec=table_spec,
                    table_caption=table_caption,
                    list_content=list_content,
                    theorem=theorem,
                    algorithm=algorithm,
                    figure=figure,
                    inline_math_tokens=sec_math,
                )
            )

        references = [_make_reference(rng, i) for i in range(n_refs)]

        return cls(
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            sections=sections,
            references=references,
            n_cols=layout.n_cols,
            font_package=layout.font_package,
            font_size_pt=layout.font_size_pt,
            margins_mm=layout.margins_mm,
            page_size=layout.page_size,
            line_spread=layout.line_spread,
            has_abstract=layout.has_abstract,
            has_page_numbers=layout.has_page_numbers,
        )

    def to_template_context(self) -> dict:
        """Return a dict suitable for Jinja2 template rendering."""

        def _list_ctx(lc: ListContent | None) -> dict | None:
            if lc is None:
                return None
            return {"entries": lc.items, "ordered": lc.ordered}

        def _thm_ctx(t: TheoremContent | None) -> dict | None:
            if t is None:
                return None
            return {"env_type": t.env_type, "body": t.body}

        def _algo_ctx(a: AlgorithmContent | None) -> dict | None:
            if a is None:
                return None
            return {"caption": a.caption, "lines": a.lines, "idx": a.idx}

        def _fig_ctx(f: FigureContent | None) -> dict | None:
            if f is None:
                return None
            return {"height_cm": f.height_cm, "caption": f.caption, "idx": f.idx}

        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract,
            "sections": [
                {
                    "heading": s.heading,
                    "body": s.body,
                    "display_equation": s.display_equation,
                    "align_block": s.align_block,
                    "table_rows": s.table_rows,
                    "table_header": s.table_header,
                    "table_spec": s.table_spec,
                    "table_caption": s.table_caption,
                    "list_content": _list_ctx(s.list_content),
                    "theorem": _thm_ctx(s.theorem),
                    "algorithm": _algo_ctx(s.algorithm),
                    "figure": _fig_ctx(s.figure),
                }
                for s in self.sections
            ],
            "references": [{"key": r.key, "text": r.text} for r in self.references],
            "n_cols": self.n_cols,
            "font_package": self.font_package,
            "font_size_pt": self.font_size_pt,
            "margins_mm": self.margins_mm,
            "page_size": self.page_size,
            "line_spread": self.line_spread,
            "has_abstract": self.has_abstract,
            "has_page_numbers": self.has_page_numbers,
        }


# ---------------------------------------------------------------------------
# Content generators
# ---------------------------------------------------------------------------


def _make_title(rng: random.Random) -> str:
    patterns = [
        lambda: (
            f"{rng.choice(_ADJECTIVES).title()} {rng.choice(_NOUNS).title()} via {rng.choice(_ADJECTIVES).title()} {rng.choice(_NOUNS).title()}"
        ),
        lambda: (
            f"Towards {rng.choice(_ADJECTIVES).title()} {rng.choice(_NOUNS).title()}s: A {rng.choice(_ADJECTIVES).title()} Approach"
        ),
        lambda: (
            f"On the {rng.choice(_ADJECTIVES).title()} {rng.choice(_NOUNS).title()} of {rng.choice(_ADJECTIVES).title()} {rng.choice(_NOUNS).title()}s"
        ),
        lambda: (
            f"A {rng.choice(_ADJECTIVES).title()} Framework for {rng.choice(_ADJECTIVES).title()} {rng.choice(_NOUNS).title()} {rng.choice(_NOUNS).title()}"
        ),
        lambda: f"{rng.choice(_ADJECTIVES).title()} {rng.choice(_NOUNS).title()}: Theory and Practice",
        lambda: (
            f"Learning {rng.choice(_ADJECTIVES).title()} {rng.choice(_NOUNS).title()}s with {rng.choice(_ADJECTIVES).title()} {rng.choice(_NOUNS).title()}"
        ),
    ]
    return escape_latex(rng.choice(patterns)())


def _make_author(rng: random.Random) -> str:
    name = escape_latex(_FAKE.name())
    institution = escape_latex(_FAKE.company())
    return f"{name} ({institution})"


def _pick_heading(rng: random.Random, used: set[str]) -> str:
    candidates = [h for h in _SECTION_HEADINGS if h not in used]
    if not candidates:
        candidates = _SECTION_HEADINGS
    return rng.choice(candidates)


def _make_paragraph(rng: random.Random, n_sentences: int) -> str:
    sentences = rng.sample(_CORPUS_SENTENCES, min(n_sentences, len(_CORPUS_SENTENCES)))
    return " ".join(sentences)


def _make_body(
    rng: random.Random,
    n_paragraphs: int,
    inline_equations_per_para: int,
    ref_keys: list[str],
) -> str:
    paragraphs = []
    for _ in range(n_paragraphs):
        n_sentences = rng.randint(3, 7)
        sentences = rng.choices(_CORPUS_SENTENCES, k=n_sentences)
        para = " ".join(sentences)
        for _ in range(inline_equations_per_para):
            if rng.random() < 0.5:
                eq = rng.choice(_EQUATIONS_INLINE)
                para += f" In particular, ${eq}$ holds under these conditions."
        if ref_keys and rng.random() < 0.35:
            key = rng.choice(ref_keys)
            para = para.rstrip(".") + f" \\cite{{{key}}}."
        paragraphs.append(para)
    return "\n\n".join(paragraphs)


def _make_align_equation(rng: random.Random) -> str:
    body = rng.choice(_ALIGN_BLOCKS)
    return "\\begin{align}\n" + body + "\n\\end{align}"


def _make_table(rng: random.Random, idx: int) -> tuple[list[str], str, str, str]:
    n_cols = rng.randint(2, 4)
    col_headers = [escape_latex(rng.choice(_NOUNS).title()) for _ in range(n_cols)]
    table_spec = "l" + "r" * (n_cols - 1)
    table_header = " & ".join(col_headers)

    # Vary value formats to look more like real results tables
    n_rows = rng.randint(3, 6)
    rows = []
    fmt = rng.choice(["decimal", "percent", "sci"])
    for _ in range(n_rows):
        label = escape_latex(_FAKE.word().title())
        if fmt == "percent":
            values = [f"{rng.uniform(0.1, 99.9):.1f}\\%" for _ in range(n_cols - 1)]
        elif fmt == "sci":
            values = [f"{rng.uniform(1e-5, 9.9e-2):.2e}" for _ in range(n_cols - 1)]
        else:
            values = [f"{rng.uniform(0.1, 99.9):.2f}" for _ in range(n_cols - 1)]
        rows.append(" & ".join([label] + values))

    adj = escape_latex(rng.choice(_ADJECTIVES))
    noun = escape_latex(rng.choice(_NOUNS))
    caption = f"\\textbf{{Table {idx + 1}.}} {adj.title()} {noun} comparison across methods."
    return rows, table_header, table_spec, caption


def _make_list(rng: random.Random, ordered: bool, ref_keys: list[str]) -> ListContent:
    n_items = rng.randint(3, 5)
    items = list(rng.sample(_CORPUS_SENTENCES, min(n_items, len(_CORPUS_SENTENCES))))
    if ref_keys:
        for i in range(len(items)):
            if rng.random() < 0.25:
                key = rng.choice(ref_keys)
                items[i] = items[i].rstrip(".") + f" \\cite{{{key}}}."
    return ListContent(items=items, ordered=ordered)


def _make_theorem(rng: random.Random) -> TheoremContent:
    return TheoremContent(
        env_type=rng.choice(_THEOREM_ENVS),
        body=rng.choice(_THEOREM_BODIES),
    )


def _make_algorithm(rng: random.Random, idx: int) -> AlgorithmContent:
    caption = escape_latex(f"{rng.choice(_ADJECTIVES).title()} {rng.choice(_NOUNS).title()} Algorithm")

    # Choose variables
    n_params = rng.randint(2, 3)
    data_var = rng.choice(_ALGO_VARS)
    other_params = rng.sample([v for v in _ALGO_VARS if v != data_var], n_params - 1)
    all_params = [data_var] + other_params

    loop_var = rng.choice(_ALGO_LOOP_VARS)
    n_var = rng.choice(["n", "m", "N", "T"])
    accum = rng.choice(_ALGO_ACCUMS)
    threshold = rng.choice(_ALGO_THRESHOLDS)
    update_tmpl = rng.choice(_ALGO_UPDATES)
    update = update_tmpl.format(acc=accum, v=data_var, lv=loop_var)

    param_str = ",\\; ".join(f"${p}$" for p in all_params)

    lines = [
        f"\\Require {param_str}",
        f"\\State ${accum} \\gets 0$",
        f"\\For{{${loop_var} = 1$ \\textbf{{to}} ${n_var}$}}",
        f"    \\If{{${data_var}_{{{loop_var}}} > {threshold}$}}",
        f"        \\State ${update}$",
        "    \\EndIf",
        "\\EndFor",
        f"\\State \\Return ${accum}$",
    ]

    return AlgorithmContent(caption=caption, lines=lines, idx=idx)


def _make_figure(rng: random.Random, idx: int) -> FigureContent:
    height_cm = round(rng.uniform(2.5, 5.0), 1)
    adj = escape_latex(rng.choice(_ADJECTIVES))
    noun = escape_latex(rng.choice(_NOUNS))
    caption = f"\\textbf{{Figure {idx + 1}.}} Illustration of the {adj} {noun} under varying experimental conditions."
    return FigureContent(height_cm=height_cm, caption=caption, idx=idx)


def _make_reference(rng: random.Random, idx: int) -> ReferenceEntry:
    author = escape_latex(_FAKE.last_name())
    year = rng.randint(2010, 2024)
    title = escape_latex(
        f"{rng.choice(_ADJECTIVES).title()} {rng.choice(_NOUNS).title()} "
        f"for {rng.choice(_ADJECTIVES).title()} {rng.choice(_NOUNS).title()}"
    )
    journal = escape_latex(f"Journal of {rng.choice(_ADJECTIVES).title()} {rng.choice(_NOUNS).title()}s")
    vol = rng.randint(1, 50)
    pages_start = rng.randint(1, 900)
    pages_end = pages_start + rng.randint(5, 20)
    return ReferenceEntry(
        key=f"ref{idx}",
        text=f"{author} et al. ({year}). {title}. \\textit{{{journal}}}, {vol}:{pages_start}--{pages_end}.",
    )


# Keep for external callers that rely on this name (e.g. sharding checks)
_strip_non_alpha = re.compile(r"[^a-zA-Z]")
