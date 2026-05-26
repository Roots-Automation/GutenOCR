"""OTSL token vocabulary and structure encoder.

Mirrors the OTSL constants and encoding logic from roots-ocr's
src/roots_ocr/data/tables/formats.py. No cross-repo import — this copy
exists so GutenOCR can encode OTSL without depending on the private
roots-ocr package.

Token vocabulary:
    FCEL  — full cell (primary cell with content)
    ECEL  — empty cell (primary cell without content)
    XCEL  — horizontal span extension (cell continues to the right)
    YCEL  — vertical span extension (cell continues from row above)
    NL    — end of row
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from table_structure import TableStructure

FCEL = "FCEL"
ECEL = "ECEL"
XCEL = "XCEL"
YCEL = "YCEL"
NL = "NL"

OTSL_VOCAB = frozenset({FCEL, ECEL, XCEL, YCEL, NL})


def structure_to_otsl(structure: TableStructure, content_grid: list[list[str]]) -> str:
    """Encode a TableStructure as a space-separated OTSL token sequence.

    Args:
        structure: Table structure with rows, cols, and spans.
        content_grid: rows×cols grid of cell text strings (empty string = ECEL).

    Returns:
        Space-separated OTSL token string, one NL per row.
    """
    rows = structure.rows
    cols = structure.cols

    # Build occupancy: primary_cells maps (r,c) → True, span_types maps (r,c) → 'x'|'y'
    primary: set[tuple[int, int]] = set()
    span_types: dict[tuple[int, int], str] = {}

    for span in structure.spans:
        r0, c0 = span.row, span.col
        primary.add((r0, c0))
        for dr in range(span.rowspan):
            for dc in range(span.colspan):
                if dr == 0 and dc == 0:
                    continue
                pos = (r0 + dr, c0 + dc)
                span_types[pos] = "x" if dr == 0 else "y"

    # All non-span, non-extension positions are primary cells
    all_span_extensions = set(span_types.keys())

    tokens: list[str] = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) in all_span_extensions:
                tokens.append(XCEL if span_types[(r, c)] == "x" else YCEL)
            else:
                text = content_grid[r][c] if r < len(content_grid) and c < len(content_grid[r]) else ""
                tokens.append(FCEL if text.strip() else ECEL)
        tokens.append(NL)

    return " ".join(tokens)


def validate_otsl(otsl_str: str, *, rows: int, cols: int) -> None:
    """Validate OTSL token sequence against expected table dimensions.

    Args:
        otsl_str: Space-separated OTSL token string.
        rows: Expected number of rows (NL tokens).
        cols: Expected number of non-NL tokens per row.

    Raises:
        ValueError: On unknown tokens, wrong row count, or wrong column count.
    """
    if not otsl_str:
        raise ValueError("OTSL string is empty")

    tokens = otsl_str.split()
    unknown = [t for t in tokens if t not in OTSL_VOCAB]
    if unknown:
        raise ValueError(f"Unknown OTSL tokens: {unknown}")

    # Split into rows by NL
    token_rows: list[list[str]] = []
    current: list[str] = []
    for t in tokens:
        if t == NL:
            token_rows.append(current)
            current = []
        else:
            current.append(t)
    if current:
        raise ValueError("OTSL sequence does not end with NL")

    if len(token_rows) != rows:
        raise ValueError(f"OTSL has {len(token_rows)} rows, expected {rows}")
    for i, row in enumerate(token_rows):
        if len(row) != cols:
            raise ValueError(f"OTSL row {i} has {len(row)} tokens, expected {cols}")
