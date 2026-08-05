"""Convert HTML table markup to a flat list of OTSL tokens.

Handles rowspan/colspan, empty cells, and header rows (th → ched in semantic mode).
Returns None for malformed or empty tables.
"""

from __future__ import annotations

from html.parser import HTMLParser


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[dict]] = []  # rows → cells with text/rowspan/colspan/is_header
        self._current_row: list[dict] | None = None
        self._current_cell: dict | None = None
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attr = dict(attrs)
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th") and self._current_row is not None:
            self._current_cell = {
                "text": "",
                "rowspan": int(attr.get("rowspan", 1)),
                "colspan": int(attr.get("colspan", 1)),
                "is_header": tag == "th",
            }
            self._in_cell = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._current_cell is not None:
            self._current_row.append(self._current_cell)
            self._current_cell = None
            self._in_cell = False
        elif tag == "tr" and self._current_row is not None:
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str) -> None:
        if self._in_cell and self._current_cell is not None:
            self._current_cell["text"] += data


def html_table_to_otsl(html: str, *, semantic: bool = False, max_cells: int = 10_000) -> list[str] | None:
    """Parse an HTML table and return OTSL tokens, or None if unparseable.

    Args:
        html: Raw HTML string containing a <table> element.
        semantic: If True, emit ``ched`` for header cells (th) in the first row
                  instead of ``fcel``/``ecel``. Requires the table to use <th>
                  tags for header cells.
    """
    parser = _TableParser()
    try:
        parser.feed(html)
    except Exception:
        return None

    if not parser.rows:
        return None

    # Build a sparse grid: grid[r][c] = anchor cell dict, or None for extension slots
    # First pass: determine grid dimensions
    # We expand rowspan/colspan into a full occupancy map
    max_cols = 0
    # occupancy[r][c] = True means cell is occupied by a span extension
    occupancy: dict[tuple[int, int], bool] = {}
    # anchor_at[r][c] = cell dict for the anchor of the span covering (r,c)
    anchor_at: dict[tuple[int, int], dict] = {}
    # span_origin[r][c] = (anchor_r, anchor_c)
    span_origin: dict[tuple[int, int], tuple[int, int]] = {}

    row_idx = 0
    for raw_row in parser.rows:
        col_idx = 0
        for cell in raw_row:
            # Advance past occupied slots
            while (row_idx, col_idx) in occupancy:
                col_idx += 1
            rs = max(1, cell["rowspan"])
            cs = max(1, cell["colspan"])
            anchor_at[(row_idx, col_idx)] = cell
            for dr in range(rs):
                for dc in range(cs):
                    pos = (row_idx + dr, col_idx + dc)
                    if dr == 0 and dc == 0:
                        span_origin[pos] = (row_idx, col_idx)
                        continue
                    occupancy[pos] = True
                    span_origin[pos] = (row_idx, col_idx)
            col_idx += cs
        max_cols = max(max_cols, col_idx)
        # Account for cells that extend into this row from above
        while (row_idx, max_cols) in occupancy or (row_idx, max_cols) in anchor_at:
            max_cols += 1
        row_idx += 1

    n_rows = row_idx
    if n_rows == 0 or max_cols == 0:
        return None

    # Determine actual max cols by scanning all anchor positions
    actual_cols = max(c + 1 for (_, c) in anchor_at) if anchor_at else 0
    # Also check occupied
    if occupancy:
        actual_cols = max(actual_cols, max(c + 1 for (_, c) in occupancy))
    if actual_cols == 0:
        return None
    if n_rows * actual_cols > max_cells:
        return None

    # Check first row is all-header (for semantic ched)
    first_row_cells = [anchor_at.get((0, c)) for c in range(actual_cols) if (0, c) in anchor_at]
    first_row_is_header = (
        semantic and bool(first_row_cells) and all(c["is_header"] for c in first_row_cells if c is not None)
    )

    tokens: list[str] = []
    for r in range(n_rows):
        for c in range(actual_cols):
            if (r, c) in anchor_at:
                cell = anchor_at[(r, c)]
                if semantic and first_row_is_header and r == 0:
                    tokens.append("ched")
                elif cell["text"].strip():
                    tokens.append("fcel")
                else:
                    tokens.append("ecel")
            elif (r, c) in span_origin:
                ar, ac = span_origin[(r, c)]
                dr, dc = r - ar, c - ac
                if dr > 0 and dc > 0:
                    tokens.append("xcel")
                elif dc > 0:
                    tokens.append("lcel")
                else:
                    tokens.append("ucel")
            else:
                # Gap — treat as empty cell
                tokens.append("ecel")
        tokens.append("nl")

    return tokens


def mustard_to_otsl(otsl_str: str) -> list[str] | None:
    """Remap MUSTARD's uppercase single-char OTSL dialect to docling vocab.

    MUSTARD encodes tokens as a concatenated string of single chars (no spaces):
      F=fcel, L=lcel, U=ucel, E=xcel, N=nl
    (no ecel distinction — all primary cells are F)
    """
    _MAP = {"F": "fcel", "L": "lcel", "U": "ucel", "E": "xcel", "N": "nl"}
    tokens = []
    for ch in otsl_str.strip():
        mapped = _MAP.get(ch)
        if mapped is None:
            return None
        tokens.append(mapped)
    if not tokens or tokens[-1] != "nl":
        return None
    return tokens
