"""PDFExtractor: pdfplumber-based word/line/block bbox extraction.

Coordinate notes:
  - pdfplumber's char dicts return {x0, top, x1, bottom} in *top-down*
    PDF coordinate space (y=0 at top of page). This matches image conventions.
  - PDF points are converted to pixels via: pixel = pt * (dpi / 72.0)
  - All output bboxes are normalized to [0, 1].

Math handling:
  - Chars in Computer Modern / STIX2 math fonts are flagged as math.
  - Consecutive math chars on the same line are grouped into one word token.
  - If source_math_tokens is provided to extract_pages(), math token text is
    replaced with the original LaTeX source string (e.g. "$\\epsilon$").
  - Display-equation lines (all-math chars, no prose) are dropped entirely.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from serialization import BlockAnnotation, LineAnnotation, WordAnnotation

# Region detection: strings that anchor structural zones
_ABSTRACT_ANCHORS = {"abstract"}
_REFERENCE_ANCHORS = {"references", "bibliography", "bibliography."}
_HEADING_MAX_WORDS = 8

# Word boundary: gap ≥ this (PDF points) between consecutive prose chars → new word
_WORD_GAP_PT = 1.5

# "Strong" math fonts: rendered ONLY in math mode (LuaLaTeX/lmodern/CM/STIX2).
# LuaLaTeX with lmodern uses Latin Modern math fonts:
#   LMMathItalic  → math italic variables (replaces CMMI)
#   LMMathSymbols → math operators/symbols (replaces CMSY)
#   LMMathExtension → large operators/delimiters (replaces CMEX)
_MATH_FONT_RE = re.compile(
    r"^LMMath(Italic|Symbols|Extension)"
    r"|^(CMMI|CMSY|CMEX|CMR|CMBX|CMSS|CMTI|CMTT)\d"
    r"|STIXTwoMath",
    re.IGNORECASE,
)

# Bold Roman font used exclusively by \mathbf{} in math mode.
# Single bold-Roman chars (x, y, A, …) in a math-confirmed line are treated as
# strong anchors for merge purposes — they are math variables, not prose.
_MATHBF_FONT_RE = re.compile(r"^LMRoman\d+-Bold$", re.IGNORECASE)

# Function/operator names rendered in roman (LMRoman) inside math mode.
# These are "weakly prose" — they should be absorbed into adjacent math tokens.
_MATH_FUNC_NAMES: frozenset[str] = frozenset(
    {
        "log",
        "ln",
        "cos",
        "sin",
        "tan",
        "cot",
        "sec",
        "csc",
        "exp",
        "max",
        "min",
        "det",
        "dim",
        "lim",
        "inf",
        "sup",
        "arg",
        "ker",
        "gcd",
        "lcm",
        "rank",
        "tr",
        "mod",
        "deg",
        "div",
        "grad",
        "curl",
        "sgn",
        "var",
        "cov",
        "erf",
        "Pr",
        "Re",
        "Im",
        "arccos",
        "arcsin",
        "arctan",
    }
)


@dataclass
class PageAnnotations:
    """Extracted annotations for one page."""

    words: list[WordAnnotation]
    lines: list[LineAnnotation]
    blocks: list[BlockAnnotation]
    word_count: int
    line_count: int


def _is_math_font(fontname: str) -> bool:
    base = fontname.split("+")[-1] if "+" in fontname else fontname
    return bool(_MATH_FONT_RE.match(base))


def _build_words_from_chars(
    chars: list[dict],
    page_w_pts: float,
    n_cols: int,
) -> list[dict]:
    """Build word-level tokens from pdfplumber char dicts.

    Each returned dict has: text, x0, top, x1, bottom, _col, _math.
    Display-equation lines (all-math chars, no prose) are dropped.
    """
    if not chars:
        return []

    char_heights = [c["bottom"] - c["top"] for c in chars if c["bottom"] > c["top"]]
    med_h = statistics.median(char_heights) if char_heights else 10.0
    line_tol = med_h * 0.5

    tagged = []
    for c in chars:
        x_mid = (c["x0"] + c["x1"]) / 2.0
        col = 0 if (n_cols == 1 or x_mid < page_w_pts / 2.0) else 1
        tagged.append({**c, "_col": col, "_math": _is_math_font(c.get("fontname", ""))})

    tagged.sort(key=lambda c: (c["_col"], c["top"], c["x0"]))

    # Group into lines within each column
    line_groups: list[tuple[int, list[dict]]] = []
    for col_id in range(n_cols):
        col_chars = [c for c in tagged if c["_col"] == col_id]
        if not col_chars:
            continue
        current = [col_chars[0]]
        for c in col_chars[1:]:
            if abs(c["top"] - current[-1]["top"]) <= line_tol:
                current.append(c)
            else:
                line_groups.append((col_id, sorted(current, key=lambda x: x["x0"])))
                current = [c]
        line_groups.append((col_id, sorted(current, key=lambda x: x["x0"])))

    # Classify line groups: prose lines vs math-only lines (fraction sub-lines).
    # A line is math-only if every char is EITHER from a math font OR is a
    # single letter and the line contains at least one confirmed-math char.
    # The second condition handles \mathbf variables (LMRoman-Bold letters like
    # 'a', 'b' in ‖a‖‖b‖): the ‖ chars confirm the line is math, so lone
    # single-letter bold chars are treated as math too.
    # This keeps prose bold text (section headings, etc.) as prose_lines —
    # those lines have no confirmed-math chars.
    #
    # Additionally, display/align-block formula lines — which contain math chars
    # plus LMRoman formula punctuation ((, ), =, +, |, …) but NO multi-character
    # alphabetic LMRoman sequences — are also classified as math_only.  This
    # catches entropy-style formulas H(X,Y)=H(X)+H(Y|X) that don't pass the
    # char-by-char check above (because '(' etc. are single non-alpha, not
    # single-alpha), while still letting inline expressions inside prose lines
    # fall through to prose_lines (because those lines contain "In", "particular",
    # "holds", etc. as multi-char alpha runs).
    def _is_display_formula_line(line_chars: list[dict]) -> bool:
        """True if line has confirmed math and no multi-char alphabetic non-math runs.

        Multi-char alpha non-math runs (like 'In', 'particular', 'holds') indicate
        embedded prose — the line is an inline expression inside a prose sentence.
        A line with only math chars, single operators/brackets, and single letters
        (regardless of font) is a display/align-block formula.
        """
        if not any(c["_math"] for c in line_chars):
            return False
        prev_x1 = -1.0
        run: list[dict] = []
        for c in sorted(line_chars, key=lambda ch: ch["x0"]):
            if prev_x1 >= 0 and c["x0"] - prev_x1 >= _WORD_GAP_PT:
                rtext = "".join(ch.get("text", "") for ch in run)
                if len(rtext) > 1 and rtext.isalpha() and not any(ch["_math"] for ch in run):
                    return False
                run = [c]
            else:
                run.append(c)
            prev_x1 = c["x1"]
        if run:
            rtext = "".join(c.get("text", "") for c in run)
            if len(rtext) > 1 and rtext.isalpha() and not any(c["_math"] for c in run):
                return False
        return True

    prose_lines: list[tuple[int, list[dict]]] = []
    math_only_lines: list[tuple[int, list[dict]]] = []
    for col_id, line_chars in line_groups:
        if not line_chars:
            continue
        has_confirmed = any(c["_math"] for c in line_chars)
        if all(
            c["_math"] or (has_confirmed and len(c.get("text", "")) == 1 and c.get("text", "").isalpha())
            for c in line_chars
        ) or _is_display_formula_line(line_chars):
            math_only_lines.append((col_id, line_chars))
        else:
            prose_lines.append((col_id, line_chars))

    words: list[dict] = []
    for col_id, line_chars in prose_lines:
        line_chars = sorted(line_chars, key=lambda c: c["x0"])

        # Split only on word gap — do NOT split at font-type boundaries.
        # This keeps "cos(θ)" together even though "cos" is in a prose font
        # and "(" is in a math font.
        runs: list[list[dict]] = [[line_chars[0]]]
        for i in range(1, len(line_chars)):
            c = line_chars[i]
            prev = line_chars[i - 1]
            if c["x0"] - prev["x1"] >= _WORD_GAP_PT:
                runs.append([c])
            else:
                runs[-1].append(c)

        # Merge math runs into single inline-expression tokens.
        #
        # "Strong math" runs contain chars from math-only fonts (LMMathItalic,
        # LMMathSymbols, LMMathExtension).  They anchor an expression.
        #
        # "Weak prose" runs sit BETWEEN strong-math runs but are rendered in
        # LMRoman by LaTeX: single non-letter chars (=, +, >, 0–9, …) and
        # known function names (log, cos, …).  We absorb them into the
        # growing expression.
        #
        # True prose runs (long words, multi-char alpha not in func-names)
        # break the merge — they are word boundaries between expressions.
        # True if this prose line has any confirmed-math chars (LMMathItalic, etc.).
        # Used by _is_strong to identify \mathbf bold-Roman chars as strong anchors.
        has_confirmed_math = any(ch["_math"] for ch in line_chars)

        def _is_strong(chars: list[dict]) -> bool:
            if any(ch["_math"] for ch in chars):
                return True
            # On a confirmed-math line, LMRoman-Bold chars are \mathbf variables —
            # treat any such char in the run as a strong anchor so runs like "y)"
            # (LMRoman12-Bold y + LMRoman12-Regular )) don't break the merge.
            if has_confirmed_math:
                for ch in chars:
                    fn = ch.get("fontname", "")
                    base = fn.split("+")[-1] if "+" in fn else fn
                    if _MATHBF_FONT_RE.match(base):
                        return True
            return False

        def _is_weak_prose(chars: list[dict]) -> bool:
            """Backward-absorption weak: single operators, digits, OR single letters."""
            text = "".join(ch.get("text", "") for ch in chars).strip()
            if not text:
                return True
            if len(text) == 1 and not text.isalpha():
                return True  # single operator / digit / punctuation
            if len(text) == 1 and text.isalpha():
                return True  # single letter — likely \mathbf variable
            return text.isalpha() and text in _MATH_FUNC_NAMES

        def _is_weak_bridge(chars: list[dict]) -> bool:
            """Forward-lookahead weak: only operators/digits/funcnames, NOT single letters.
            Single-letter variables (e.g. 'b' in LMMathItalic) anchor absorption;
            treating them as bridges would cause the lookahead to skip past them.
            """
            text = "".join(ch.get("text", "") for ch in chars).strip()
            if not text:
                return True
            if len(text) == 1 and not text.isalpha():
                return True  # single operator / digit / punctuation
            return text.isalpha() and text in _MATH_FUNC_NAMES

        merged: list[list[dict]] = []
        i = 0
        while i < len(runs):
            if _is_strong(runs[i]):
                # Pull in the immediately preceding weak run (e.g. \mathbf{A}
                # in "A ∈ ℝ^{m×n}" — bold 'A' sits to the left of the '∈').
                if merged and _is_weak_prose(merged[-1]):
                    combined = merged.pop() + list(runs[i])
                else:
                    combined = list(runs[i])
                j = i + 1
                while j < len(runs):
                    if _is_strong(runs[j]):
                        combined.extend(runs[j])
                        j += 1
                    elif _is_weak_prose(runs[j]):
                        # Absorb weak run only if a strong run follows it;
                        # otherwise it's trailing punctuation attached to prose.
                        # Use _is_weak_bridge (not _is_weak_prose) for the lookahead
                        # so single-letter math vars (e.g. 'b') anchor the scan.
                        k = j + 1
                        while k < len(runs) and _is_weak_bridge(runs[k]):
                            k += 1
                        if k < len(runs) and _is_strong(runs[k]):
                            for m in range(j, k):
                                combined.extend(runs[m])
                            j = k
                        else:
                            break
                    else:
                        break
                merged.append(combined)
                i = j
            else:
                merged.append(runs[i])
                i += 1
        runs = merged

        for run in runs:
            text = "".join(ch.get("text", "") for ch in run)
            if not text.strip():
                continue
            is_math = any(ch["_math"] for ch in run)
            words.append(
                {
                    "text": text,
                    "x0": run[0]["x0"],
                    "top": min(ch["top"] for ch in run),
                    "x1": run[-1]["x1"],
                    "bottom": max(ch["bottom"] for ch in run),
                    "_col": col_id,
                    "_math": is_math,
                }
            )

    # Orphan math absorption pass.
    # An "orphan" math word has no prose word in the same column overlapping its
    # y-range — meaning it sits on a line with no surrounding prose text.  Two
    # cases arise:
    #   1. Display equations (\[ ... \] blocks): isolated, wide, far from any
    #      inline math word → no x-overlap candidate within _orphan_y_tol → dropped.
    #   2. Superscript / subscript splits: when \prod_{k=0}^n renders its
    #      superscript 'n' into a separate line group, that group is isolated and
    #      very close to the main-body math word → absorbed (bbox expanded, text
    #      prepended if orphan is above).
    # Phase 1: identify orphan math words.
    _orphan_y_tol = med_h * 3.0
    orphan_ids: set[int] = set()
    for w in words:
        if not w["_math"]:
            continue
        has_prose = any(
            not pw["_math"] and pw["_col"] == w["_col"] and pw["bottom"] > w["top"] and pw["top"] < w["bottom"]
            for pw in words
        )
        if not has_prose:
            orphan_ids.add(id(w))

    # Phase 2: absorb each orphan into the nearest x-overlapping non-removed math word,
    # or drop if none is within _orphan_y_tol.
    to_remove_ids: set[int] = set()
    for w in words:
        if id(w) not in orphan_ids:
            continue
        best: dict | None = None
        best_dist = float("inf")
        for other in words:
            if other is w or not other["_math"] or other["_col"] != w["_col"]:
                continue
            if id(other) in to_remove_ids:
                continue
            x_ov = min(w["x1"], other["x1"]) - max(w["x0"], other["x0"])
            if x_ov <= 0:
                continue
            y_gap = max(0.0, max(w["top"], other["top"]) - min(w["bottom"], other["bottom"]))
            if y_gap < best_dist and y_gap <= _orphan_y_tol:
                best_dist = y_gap
                best = other
        if best is not None:
            if w["top"] < best["top"]:
                best["text"] = w["text"] + best["text"]  # superscript: prepend
            else:
                best["text"] += w["text"]  # subscript: append
            best["x0"] = min(best["x0"], w["x0"])
            best["x1"] = max(best["x1"], w["x1"])
            best["top"] = min(best["top"], w["top"])
            best["bottom"] = max(best["bottom"], w["bottom"])
        to_remove_ids.add(id(w))

    if to_remove_ids:
        words = [w for w in words if id(w) not in to_remove_ids]

    # Line-wrap merge pass.
    # LaTeX may break a long inline expression across two rendered lines:
    #   line N  ends with "…‖A‖₂"   (near the column's right margin)
    #   line N+1 starts with "σmax(A)…" (near the column's left margin)
    # Both pieces are separate math words from the SAME source token.  Merging
    # them keeps the sequential source-token alignment from drifting.
    #
    # For each math word near the right margin, scan forward through later words
    # (sorted by y) to find the first word on the next text line that sits near
    # the left margin.  Intermediate words on the SAME line (y_gap ≤ 0 or
    # bottom-overlapping) are skipped; we stop if the gap grows beyond one
    # line height.
    _WRAP_END_FRAC = 0.75
    _WRAP_START_FRAC = 0.25
    col_w = page_w_pts / n_cols
    math_by_col: dict[int, list[dict]] = {c: [] for c in range(n_cols)}
    for w in words:
        if w["_math"]:
            math_by_col[w["_col"]].append(w)
    wrap_merge_ids: set[int] = set()  # ids of words absorbed (to remove)
    for col, mwords in math_by_col.items():
        mwords_sorted = sorted(mwords, key=lambda x: x["top"])
        col_start = col * col_w
        for i, w1 in enumerate(mwords_sorted):
            if id(w1) in wrap_merge_ids:
                continue
            w1_near_end = w1["x1"] > col_start + col_w * _WRAP_END_FRAC
            if not w1_near_end:
                continue
            # Scan forward for a word on the next line that starts near left margin
            for w2 in mwords_sorted[i + 1 :]:
                if id(w2) in wrap_merge_ids:
                    continue
                y_gap = w2["top"] - w1["bottom"]
                if y_gap <= 0:
                    continue  # Same line or above — skip
                if y_gap > med_h * 0.8:
                    break  # Too far below — no line-wrap candidate
                if w2["x0"] < col_start + col_w * _WRAP_START_FRAC:
                    w1["text"] += w2["text"]
                    w1["x1"] = max(w1["x1"], w2["x1"])
                    w1["bottom"] = max(w1["bottom"], w2["bottom"])
                    wrap_merge_ids.add(id(w2))
                    break  # Merge only one continuation word
                # w2 is on the next line but not near the left margin — keep scanning
    if wrap_merge_ids:
        words = [w for w in words if id(w) not in wrap_merge_ids]

    # Absorb math-only lines (fraction sub-lines) into the nearest math word.
    # The math_only_line classification now handles \mathbf denominators too
    # (single-letter bold chars in lines that also have confirmed math chars).
    _FRAC_Y_TOL = med_h * 1.5
    for col_id, math_chars in math_only_lines:
        mo_x0 = min(c["x0"] for c in math_chars)
        mo_x1 = max(c["x1"] for c in math_chars)
        mo_top = min(c["top"] for c in math_chars)
        mo_bot = max(c["bottom"] for c in math_chars)
        mo_width = mo_x1 - mo_x0

        best: dict | None = None
        best_dist = float("inf")
        for w in words:
            if w["_col"] != col_id or not w["_math"]:
                continue
            if mo_width > (w["x1"] - w["x0"]) * 2.0:
                continue  # Too wide — likely a display equation, skip
            x_overlap = min(w["x1"], mo_x1) - max(w["x0"], mo_x0)
            if x_overlap <= 0:
                continue
            y_gap = max(0.0, max(mo_top, w["top"]) - min(mo_bot, w["bottom"]))
            if y_gap < best_dist and y_gap <= _FRAC_Y_TOL:
                best_dist = y_gap
                best = w

        if best is not None:
            best["x0"] = min(best["x0"], mo_x0)
            best["x1"] = max(best["x1"], mo_x1)
            best["top"] = min(best["top"], mo_top)
            best["bottom"] = max(best["bottom"], mo_bot)
            best["text"] += "".join(c.get("text", "") for c in math_chars)

    return words


def _apply_math_alignment(
    words: list[dict],
    source_math_tokens: list[str],
    start_idx: int,
) -> tuple[list[dict], int]:
    """Replace _math word text with source LaTeX tokens. Returns (words, new_idx)."""
    idx = start_idx
    result = []
    for w in words:
        if w["_math"]:
            if idx < len(source_math_tokens):
                w = {**w, "text": source_math_tokens[idx]}
                idx += 1
            else:
                w = {**w, "text": f"${w['text']}$"}
        result.append(w)
    return result, idx


def extract_pages(
    pdf_path: Path | str,
    page_images_sizes: list[tuple[int, int]],
    dpi: int,
    layout_n_cols: int,
    source_math_tokens: list[str] | None = None,
) -> list[PageAnnotations]:
    """Extract word/line/block annotations for every page of a PDF.

    Args:
        pdf_path: Path to the compiled PDF.
        page_images_sizes: List of (width_px, height_px) for each rasterized page.
        dpi: DPI used to rasterize (needed for pt→px conversion).
        layout_n_cols: Number of text columns (1 or 2) from LayoutConfig.
        source_math_tokens: Ordered list of $...$ source tokens for math alignment.

    Returns:
        One PageAnnotations per page.
    """
    import pdfplumber

    scale = dpi / 72.0
    results: list[PageAnnotations] = []
    math_idx = 0

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            if page_idx >= len(page_images_sizes):
                break
            img_w, img_h = page_images_sizes[page_idx]
            page_w_pts = float(page.width)

            chars = page.chars
            if not chars:
                results.append(PageAnnotations(words=[], lines=[], blocks=[], word_count=0, line_count=0))
                continue

            raw_words = _build_words_from_chars(chars, page_w_pts, layout_n_cols)

            if source_math_tokens is not None:
                raw_words, math_idx = _apply_math_alignment(raw_words, source_math_tokens, math_idx)

            if not raw_words:
                results.append(PageAnnotations(words=[], lines=[], blocks=[], word_count=0, line_count=0))
                continue

            page_annotations = _build_page_annotations(
                raw_words=raw_words,
                img_w=img_w,
                img_h=img_h,
                scale=scale,
                page_w_pts=page_w_pts,
                page_h_pts=float(page.height),
                n_cols=layout_n_cols,
            )
            results.append(page_annotations)

    return results


def _norm(val: float, dim: int) -> float:
    return round(max(0.0, min(1.0, val / dim)), 3)


def _word_to_pixel_bbox(w: dict, scale: float) -> list[float]:
    """Convert word dict → pixel bbox [x1,y1,x2,y2]."""
    return [w["x0"] * scale, w["top"] * scale, w["x1"] * scale, w["bottom"] * scale]


def _assign_column(x_center: float, page_w_pts: float, n_cols: int) -> int:
    """Return 0-indexed column for a word given its center x in PDF pts."""
    if n_cols == 1:
        return 0
    midpoint = page_w_pts / 2.0
    return 0 if x_center < midpoint else 1


def _build_page_annotations(
    raw_words: list[dict],
    img_w: int,
    img_h: int,
    scale: float,
    page_w_pts: float,
    page_h_pts: float,
    n_cols: int,
) -> PageAnnotations:
    if not raw_words:
        return PageAnnotations(words=[], lines=[], blocks=[], word_count=0, line_count=0)

    word_heights = [w["bottom"] - w["top"] for w in raw_words]
    med_height = statistics.median(word_heights) if word_heights else 10.0
    line_cluster_tol = med_height * 0.6
    block_gap_tol = med_height * 1.5

    # Use _col from _build_words_from_chars if present, else compute
    annotated = []
    for w in raw_words:
        if "_col" in w:
            col = w["_col"]
        else:
            x_center = (w["x0"] + w["x1"]) / 2.0
            col = _assign_column(x_center, page_w_pts, n_cols)
        annotated.append({**w, "_col": col})
    annotated.sort(key=lambda w: (w["_col"], w["top"], w["x0"]))

    # Cluster words into lines within each column
    lines_raw: list[dict] = []
    for col_id in range(n_cols):
        col_words = [w for w in annotated if w["_col"] == col_id]
        if not col_words:
            continue
        current_line_words = [col_words[0]]
        for w in col_words[1:]:
            prev_top = current_line_words[-1]["top"]
            if abs(w["top"] - prev_top) <= line_cluster_tol:
                current_line_words.append(w)
            else:
                lines_raw.append(_finalize_line_cluster(current_line_words, col_id))
                current_line_words = [w]
        lines_raw.append(_finalize_line_cluster(current_line_words, col_id))

    lines_raw.sort(key=lambda ln: (ln["col"], ln["top_mean"]))
    lines_raw = _assign_region_types(lines_raw, page_h_pts)

    blocks_raw: list[list[dict]] = []
    if lines_raw:
        current_block = [lines_raw[0]]
        for ln in lines_raw[1:]:
            prev = current_block[-1]
            same_col = ln["col"] == prev["col"]
            small_gap = (ln["top_mean"] - prev["bottom_mean"]) < block_gap_tol
            same_region = ln["region_type"] == prev["region_type"]
            if same_col and small_gap and same_region:
                current_block.append(ln)
            else:
                blocks_raw.append(current_block)
                current_block = [ln]
        blocks_raw.append(current_block)

    word_id = 0
    line_id = 0
    block_id = 0
    all_words: list[WordAnnotation] = []
    all_lines: list[LineAnnotation] = []
    all_blocks: list[BlockAnnotation] = []

    for block_lines in blocks_raw:
        block_line_ids: list[int] = []
        block_texts: list[str] = []
        block_bboxes: list[list[float]] = []

        for ln_raw in block_lines:
            line_word_ids: list[int] = []
            line_texts: list[str] = []
            line_px_bboxes: list[list[float]] = []

            for w in ln_raw["words"]:
                px = _word_to_pixel_bbox(w, scale)
                norm_bbox = [
                    _norm(px[0], img_w),
                    _norm(px[1], img_h),
                    _norm(px[2], img_w),
                    _norm(px[3], img_h),
                ]
                all_words.append(
                    WordAnnotation(
                        text=w["text"],
                        bbox=norm_bbox,
                        line_id=line_id,
                        word_id=word_id,
                    )
                )
                line_word_ids.append(word_id)
                line_texts.append(w["text"])
                line_px_bboxes.append(px)
                word_id += 1

            if not line_texts:
                continue

            line_px = [
                min(b[0] for b in line_px_bboxes),
                min(b[1] for b in line_px_bboxes),
                max(b[2] for b in line_px_bboxes),
                max(b[3] for b in line_px_bboxes),
            ]
            line_norm_bbox = [
                _norm(line_px[0], img_w),
                _norm(line_px[1], img_h),
                _norm(line_px[2], img_w),
                _norm(line_px[3], img_h),
            ]
            line_text = " ".join(line_texts)

            all_lines.append(
                LineAnnotation(
                    text=line_text,
                    bbox=line_norm_bbox,
                    block_id=block_id,
                    line_id=line_id,
                )
            )
            block_line_ids.append(line_id)
            block_texts.append(line_text)
            block_bboxes.append(line_norm_bbox)
            line_id += 1

        if not block_line_ids:
            continue

        block_norm_bbox = [
            min(b[0] for b in block_bboxes),
            min(b[1] for b in block_bboxes),
            max(b[2] for b in block_bboxes),
            max(b[3] for b in block_bboxes),
        ]
        all_blocks.append(
            BlockAnnotation(
                text=" ".join(block_texts),
                block_id=block_id,
                bbox=block_norm_bbox,
                line_ids=block_line_ids,
                region_type=block_lines[0]["region_type"],
            )
        )
        block_id += 1

    return PageAnnotations(
        words=all_words,
        lines=all_lines,
        blocks=all_blocks,
        word_count=len(all_words),
        line_count=len(all_lines),
    )


def _finalize_line_cluster(words: list[dict], col: int) -> dict:
    words_sorted = sorted(words, key=lambda w: w["x0"])
    top_mean = sum(w["top"] for w in words) / len(words)
    bottom_mean = sum(w["bottom"] for w in words) / len(words)
    return {
        "col": col,
        "words": words_sorted,
        "top_mean": top_mean,
        "bottom_mean": bottom_mean,
        "text": " ".join(w["text"] for w in words_sorted),
        "region_type": "body",
    }


def _assign_region_types(lines: list[dict], page_h_pts: float) -> list[dict]:
    """Heuristically assign region_type to each line dict (mutates in place)."""
    header_y_max = page_h_pts * 0.08
    footer_y_min = page_h_pts * 0.92

    abstract_end_y: float | None = None
    reference_start_y: float | None = None
    first_section_y: float | None = None

    for ln in lines:
        text_lower = ln["text"].strip().lower().rstrip(".")
        if text_lower in _ABSTRACT_ANCHORS:
            abstract_end_y = ln["bottom_mean"]
        elif text_lower in _REFERENCE_ANCHORS:
            reference_start_y = ln["top_mean"]
        elif first_section_y is None and _looks_like_section_heading(ln["text"]):
            first_section_y = ln["top_mean"]

    in_abstract = abstract_end_y is not None
    in_references = False

    for ln in lines:
        top = ln["top_mean"]
        text_lower = ln["text"].strip().lower().rstrip(".")

        if top <= header_y_max:
            ln["region_type"] = "header"
        elif top >= footer_y_min:
            ln["region_type"] = "footer"
        elif text_lower in _REFERENCE_ANCHORS:
            in_references = True
            ln["region_type"] = "reference"
        elif in_references or (reference_start_y is not None and top >= reference_start_y):
            ln["region_type"] = "reference"
        elif text_lower in _ABSTRACT_ANCHORS:
            ln["region_type"] = "abstract"
            in_abstract = True
        elif in_abstract and (first_section_y is None or top < first_section_y):
            ln["region_type"] = "abstract"
        elif _looks_like_section_heading(ln["text"]) and len(ln["words"]) <= _HEADING_MAX_WORDS:
            ln["region_type"] = "heading"
            in_abstract = False
        else:
            ln["region_type"] = "body"

    return lines


def _looks_like_section_heading(text: str) -> bool:
    words = text.split()
    if not words or len(words) > _HEADING_MAX_WORDS:
        return False
    title_words = sum(1 for w in words if w and w[0].isupper())
    return title_words / len(words) >= 0.6
