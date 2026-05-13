"""Random table structure generation.

Produces TableStructure instances describing the dimensions, span layout,
header configuration, and border style of a synthetic table — without any
rendering concern.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum


class BorderStyle(Enum):
    FULL = "full"
    OUTER_ONLY = "outer_only"
    INNER_ONLY = "inner_only"
    NONE = "none"


@dataclass(frozen=True)
class Span:
    """A merged cell anchored at (row, col) extending rowspan×colspan."""

    row: int
    col: int
    rowspan: int
    colspan: int


@dataclass
class TableStructure:
    """Complete structural description of a synthetic table."""

    rows: int
    cols: int
    spans: list[Span] = field(default_factory=list)
    has_header: bool = False
    irregular_header: bool = False
    border_style: BorderStyle = BorderStyle.FULL


def _build_occupancy(rows: int, cols: int, spans: list[Span]) -> set[tuple[int, int]]:
    """Return the set of all cells occupied by span extensions."""
    occupied: set[tuple[int, int]] = set()
    for span in spans:
        for dr in range(span.rowspan):
            for dc in range(span.colspan):
                if dr == 0 and dc == 0:
                    continue
                occupied.add((span.row + dr, span.col + dc))
    return occupied


def generate_table_structure(
    rng: random.Random,
    *,
    min_rows: int = 2,
    max_rows: int = 12,
    min_cols: int = 2,
    max_cols: int = 8,
    span_prob: float = 0.2,
    header_prob: float = 0.7,
) -> TableStructure:
    """Generate a random TableStructure.

    Args:
        rng: Seeded random instance for reproducibility.
        min_rows: Minimum number of rows.
        max_rows: Maximum number of rows.
        min_cols: Minimum number of columns.
        max_cols: Maximum number of columns.
        span_prob: Probability that any given primary cell starts a span.
        header_prob: Probability that the table has a header row.

    Returns:
        A TableStructure with validated, non-overlapping spans.
    """
    rows = rng.randint(min_rows, max_rows)
    cols = rng.randint(min_cols, max_cols)

    has_header = rng.random() < header_prob
    irregular_header = has_header and rng.random() < 0.3
    border_style = rng.choice(list(BorderStyle))

    spans: list[Span] = []
    # Track occupied extension positions to avoid overlap
    occupied: set[tuple[int, int]] = set()

    for r in range(rows):
        for c in range(cols):
            if (r, c) in occupied:
                continue
            if rng.random() >= span_prob:
                continue
            # Determine maximum possible span extents
            max_cs = min(3, cols - c)
            max_rs = min(3, rows - r)
            colspan = rng.randint(1, max_cs)
            rowspan = rng.randint(1, max_rs)
            if colspan == 1 and rowspan == 1:
                continue

            # Verify no overlap with existing spans
            candidate_occupied: set[tuple[int, int]] = set()
            conflict = False
            for dr in range(rowspan):
                for dc in range(colspan):
                    if dr == 0 and dc == 0:
                        continue
                    pos = (r + dr, c + dc)
                    if pos in occupied:
                        conflict = True
                        break
                    candidate_occupied.add(pos)
                if conflict:
                    break

            if not conflict:
                spans.append(Span(row=r, col=c, rowspan=rowspan, colspan=colspan))
                occupied |= candidate_occupied

    return TableStructure(
        rows=rows,
        cols=cols,
        spans=spans,
        has_header=has_header,
        irregular_header=irregular_header,
        border_style=border_style,
    )
