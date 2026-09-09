"""LayoutConfig: randomized academic document layout parameters."""

from __future__ import annotations

import random
from dataclasses import dataclass

FONT_PACKAGES: dict[str, str] = {
    "lmodern": r"\usepackage{lmodern}",
    "tgpagella": r"\usepackage{tgpagella}",
    "tgtermes": r"\usepackage{tgtermes}",
    "tgbonum": r"\usepackage{tgbonum}",
    # stix2 requires fontspec (lualatex/xelatex only)
    "stix2": "\\usepackage{fontspec}\n\\setmainfont{STIX Two Text}",
}

_FONT_FAMILIES = list(FONT_PACKAGES)
_FONT_SIZES = [10, 11, 12]
_PAGE_SIZES = ["a4paper", "letterpaper"]


@dataclass
class LayoutConfig:
    n_cols: int
    font_family: str
    font_size_pt: int
    line_spread: float
    margins_mm: int
    page_size: str
    has_abstract: bool
    n_sections: int
    has_references: bool
    has_page_numbers: bool
    equations_per_section: int
    tables_per_doc: int
    algorithms_per_doc: int
    figures_per_doc: int
    theorem_envs_per_doc: int
    lists_per_doc: int

    @property
    def font_package(self) -> str:
        return FONT_PACKAGES[self.font_family]

    @classmethod
    def sample(
        cls,
        rng: random.Random,
        *,
        n_cols_weights: list[float] | None = None,
        font_families: list[str] | None = None,
        font_sizes: list[int] | None = None,
        line_spread_range: tuple[float, float] = (1.0, 1.3),
        margins_mm_range: tuple[int, int] = (15, 30),
        page_sizes: list[str] | None = None,
        abstract_prob: float = 0.75,
        n_sections_range: tuple[int, int] = (2, 8),
        references_prob: float = 0.70,
        page_numbers_prob: float = 0.80,
        equations_per_section_range: tuple[int, int] = (0, 3),
        tables_per_doc_range: tuple[int, int] = (0, 2),
        algorithms_per_doc_range: tuple[int, int] = (0, 2),
        figures_per_doc_range: tuple[int, int] = (0, 2),
        theorem_envs_per_doc_range: tuple[int, int] = (0, 3),
        lists_per_doc_range: tuple[int, int] = (0, 3),
    ) -> LayoutConfig:
        weights = n_cols_weights or [0.4, 0.6]
        n_cols = rng.choices([1, 2], weights=weights)[0]

        families = font_families or _FONT_FAMILIES
        sizes = font_sizes or _FONT_SIZES
        pages = page_sizes or _PAGE_SIZES

        return cls(
            n_cols=n_cols,
            font_family=rng.choice(families),
            font_size_pt=rng.choice(sizes),
            line_spread=round(rng.uniform(*line_spread_range), 2),
            margins_mm=rng.randint(*margins_mm_range),
            page_size=rng.choice(pages),
            has_abstract=rng.random() < abstract_prob,
            n_sections=rng.randint(*n_sections_range),
            has_references=rng.random() < references_prob,
            has_page_numbers=rng.random() < page_numbers_prob,
            equations_per_section=rng.randint(*equations_per_section_range),
            tables_per_doc=rng.randint(*tables_per_doc_range),
            algorithms_per_doc=rng.randint(*algorithms_per_doc_range),
            figures_per_doc=rng.randint(*figures_per_doc_range),
            theorem_envs_per_doc=rng.randint(*theorem_envs_per_doc_range),
            lists_per_doc=rng.randint(*lists_per_doc_range),
        )

    @classmethod
    def from_config(cls, rng: random.Random, cfg: dict) -> LayoutConfig:
        layout_cfg = cfg.get("layout", {})
        return cls.sample(
            rng,
            n_cols_weights=layout_cfg.get("n_cols_weights"),
            font_families=layout_cfg.get("font_families"),
            font_sizes=layout_cfg.get("font_sizes"),
            line_spread_range=tuple(layout_cfg.get("line_spread_range", [1.0, 1.3])),
            margins_mm_range=tuple(layout_cfg.get("margins_mm_range", [15, 30])),
            page_sizes=layout_cfg.get("page_sizes"),
            abstract_prob=layout_cfg.get("abstract_prob", 0.75),
            n_sections_range=tuple(layout_cfg.get("n_sections_range", [2, 8])),
            references_prob=layout_cfg.get("references_prob", 0.70),
            page_numbers_prob=layout_cfg.get("page_numbers_prob", 0.80),
            equations_per_section_range=tuple(layout_cfg.get("equations_per_section_range", [0, 3])),
            tables_per_doc_range=tuple(layout_cfg.get("tables_per_doc_range", [0, 2])),
            algorithms_per_doc_range=tuple(layout_cfg.get("algorithms_per_doc_range", [0, 2])),
            figures_per_doc_range=tuple(layout_cfg.get("figures_per_doc_range", [0, 2])),
            theorem_envs_per_doc_range=tuple(layout_cfg.get("theorem_envs_per_doc_range", [0, 3])),
            lists_per_doc_range=tuple(layout_cfg.get("lists_per_doc_range", [0, 3])),
        )

    def to_dict(self) -> dict:
        return {
            "n_cols": self.n_cols,
            "font_family": self.font_family,
            "font_size_pt": self.font_size_pt,
            "line_spread": self.line_spread,
            "margins_mm": self.margins_mm,
            "page_size": self.page_size,
            "has_abstract": self.has_abstract,
            "n_sections": self.n_sections,
            "has_references": self.has_references,
            "has_page_numbers": self.has_page_numbers,
            "equations_per_section": self.equations_per_section,
            "tables_per_doc": self.tables_per_doc,
            "algorithms_per_doc": self.algorithms_per_doc,
            "figures_per_doc": self.figures_per_doc,
            "theorem_envs_per_doc": self.theorem_envs_per_doc,
            "lists_per_doc": self.lists_per_doc,
        }
