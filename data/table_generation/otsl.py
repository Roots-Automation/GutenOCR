"""OTSL token vocabulary and structure encoder.

Matches the docling-project OTSL vocabulary from FinTabNet_OTSL / the paper
"Optimized Table Tokenization for Table Structure Recognition" (arXiv 2305.03393).

Two flavors are supported:

  base (6 tokens):
    fcel  — full cell (primary cell with content)
    ecel  — empty cell (primary cell without content)
    lcel  — left-looking cell (horizontal span extension, colspan)
    ucel  — up-looking cell (vertical span extension, rowspan)
    xcel  — 2D extension (cell extends both right and down from its anchor)
    nl    — end of row

  semantic (9 tokens, superset of base):
    ched  — column header cell (replaces fcel/ecel in the header row)
    rhed  — row header cell (replaces fcel/ecel in the row-header column)
    srow  — section row cell (replaces fcel/ecel in section/separator rows)

Priority for semantic primary-cell assignment (extension cells are always
lcel/ucel/xcel regardless of row/col role):
    section row  →  srow
    header row   →  ched
    row-header col → rhed
    otherwise    →  fcel / ecel
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from table_structure import TableStructure

# Base tokens
FCEL = "fcel"
ECEL = "ecel"
LCEL = "lcel"
UCEL = "ucel"
XCEL = "xcel"
NL = "nl"

# Semantic extension tokens
CHED = "ched"
RHED = "rhed"
SROW = "srow"

OTSL_VOCAB_BASE = frozenset({FCEL, ECEL, LCEL, UCEL, XCEL, NL})
OTSL_VOCAB_SEMANTIC = frozenset({FCEL, ECEL, LCEL, UCEL, XCEL, NL, CHED, RHED, SROW})

# Default: accept both flavors during validation
OTSL_VOCAB = OTSL_VOCAB_SEMANTIC

Flavor = Literal["base", "semantic"]


def structure_to_otsl(
    structure: TableStructure,
    content_grid: list[list[str]],
    *,
    flavor: Flavor = "semantic",
) -> str:
    """Encode a TableStructure as a space-separated OTSL token sequence.

    Args:
        structure: Table structure with rows, cols, spans, and semantic fields.
        content_grid: rows×cols grid of cell text strings (empty string = ecel).
        flavor: ``"base"`` emits only the 6-token vocab (fcel/ecel/lcel/ucel/xcel/nl).
                ``"semantic"`` additionally emits ched/rhed/srow for header and
                section rows (requires ``structure.has_row_header`` and
                ``structure.section_rows``).

    Returns:
        Space-separated OTSL token string, one nl per row.
    """
    rows = structure.rows
    cols = structure.cols

    # Build span_types: maps extension positions → 'l' | 'u' | 'x'
    #   l (lcel) — same row as anchor, different col  (pure horizontal extension)
    #   u (ucel) — different row from anchor, same col (pure vertical extension)
    #   x (xcel) — different row AND different col     (2D extension)
    span_types: dict[tuple[int, int], str] = {}

    for span in structure.spans:
        r0, c0 = span.row, span.col
        for dr in range(span.rowspan):
            for dc in range(span.colspan):
                if dr == 0 and dc == 0:
                    continue
                pos = (r0 + dr, c0 + dc)
                if dr == 0:
                    span_types[pos] = "l"
                elif dc == 0:
                    span_types[pos] = "u"
                else:
                    span_types[pos] = "x"

    use_semantic = flavor == "semantic"
    section_row_set = set(getattr(structure, "section_rows", []))
    has_row_header = getattr(structure, "has_row_header", False)

    tokens: list[str] = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) in span_types:
                t = span_types[(r, c)]
                tokens.append(LCEL if t == "l" else (UCEL if t == "u" else XCEL))
            elif use_semantic and r in section_row_set:
                tokens.append(SROW)
            elif use_semantic and structure.has_header and r == 0:
                tokens.append(CHED)
            elif use_semantic and has_row_header and c == 0:
                tokens.append(RHED)
            else:
                text = content_grid[r][c] if r < len(content_grid) and c < len(content_grid[r]) else ""
                tokens.append(FCEL if text.strip() else ECEL)
        tokens.append(NL)

    return " ".join(tokens)


def validate_otsl(otsl_str: str, *, rows: int, cols: int, flavor: Flavor = "semantic") -> None:
    """Validate OTSL token sequence against expected table dimensions.

    Args:
        otsl_str: Space-separated OTSL token string.
        rows: Expected number of rows (nl tokens).
        cols: Expected number of non-nl tokens per row.
        flavor: Which vocab to validate against (``"base"`` or ``"semantic"``).

    Raises:
        ValueError: On unknown tokens, wrong row count, or wrong column count.
    """
    if not otsl_str:
        raise ValueError("OTSL string is empty")

    vocab = OTSL_VOCAB_BASE if flavor == "base" else OTSL_VOCAB_SEMANTIC
    tokens = otsl_str.split()
    unknown = [t for t in tokens if t not in vocab]
    if unknown:
        raise ValueError(f"Unknown OTSL tokens: {unknown}")

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
