"""Entry rendering (headword + definition or index) for index_definition generator."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import ImageDraw, ImageFont

from elements.layout import ColumnCursor, ColumnRect


@dataclass
class RenderedLine:
    text: str
    block_id: int
    region_type: str  # "heading" or "body"
    x1_px: int
    y1_px: int
    x2_px: int
    y2_px: int
    words: list[str] = field(default_factory=list)
    word_x1s: list[int] = field(default_factory=list)
    word_x2s: list[int] = field(default_factory=list)
    font_family: str = ""
    font_size_px: int = 0
    text_color_rgb: list[int] = field(default_factory=lambda: [0, 0, 0])


def _load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(path), size)
    except Exception:
        return ImageFont.load_default()


def _truncate_word(word: str, font: ImageFont.FreeTypeFont, max_width_px: int) -> str:
    """Truncate a single word to fit within max_width_px."""
    if font.getlength(word) <= max_width_px:
        return word
    ellipsis = "…"
    for i in range(len(word), 0, -1):
        candidate = word[:i] + ellipsis
        if font.getlength(candidate) <= max_width_px:
            return candidate
    return ellipsis


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width_px: int) -> list[str]:
    """Greedy word-wrap. Returns list of line strings."""
    words = text.split()
    if not words:
        return []
    space_w = font.getlength(" ")
    lines: list[str] = []
    current: list[str] = []
    current_w = 0.0
    for word in words:
        word = _truncate_word(word, font, max_width_px)
        word_w = font.getlength(word)
        gap = space_w if current else 0.0
        if current and current_w + gap + word_w > max_width_px:
            lines.append(" ".join(current))
            current = [word]
            current_w = word_w
        else:
            current.append(word)
            current_w += gap + word_w
    if current:
        lines.append(" ".join(current))
    return lines


def compute_word_positions(
    line_text: str,
    font: ImageFont.FreeTypeFont,
    line_x: int,
) -> tuple[list[int], list[int]]:
    """Return (x1s, x2s) pixel positions for each whitespace-delimited word."""
    words = line_text.split()
    if not words:
        return [], []
    space_w = font.getlength(" ")
    x = float(line_x)
    x1s: list[int] = []
    x2s: list[int] = []
    for i, word in enumerate(words):
        word_w = font.getlength(word)
        x1s.append(int(x))
        x2s.append(int(x + word_w))
        x += word_w + (space_w if i < len(words) - 1 else 0.0)
    return x1s, x2s


def _line_height(font: ImageFont.FreeTypeFont, extra_px: int = 0) -> int:
    ascent, descent = font.getmetrics()
    return ascent + descent + extra_px


def _entry_height_definition(
    headword: str,
    definition: str,
    col: ColumnRect,
    body_font: ImageFont.FreeTypeFont,
    head_font: ImageFont.FreeTypeFont,
    head_style: str,
    extra_px: int,
) -> int:
    hfont = head_font if head_style == "large" else body_font
    hw_lines = wrap_text(headword, hfont, col.width) or [headword]
    head_h = _line_height(hfont, extra_px) * len(hw_lines)
    def_lines = wrap_text(definition, body_font, col.width)
    body_h = _line_height(body_font, extra_px) * max(1, len(def_lines))
    return head_h + body_h


def _entry_height_index(
    col: ColumnRect,
    body_font: ImageFont.FreeTypeFont,
    extra_px: int,
) -> int:
    return _line_height(body_font, extra_px)


class EntryRenderer:
    def __init__(self, config: dict, project_root: Path):
        font_cfg = config.get("font", {})
        font_dirs = [(project_root / p).resolve() for p in font_cfg.get("paths", [])]
        self._font_paths = self._discover_fonts(font_dirs)
        self._base_size_range = font_cfg.get("base_size", [8, 14])
        self._head_scale_range = font_cfg.get("headword_scale", [1.2, 1.8])
        self._head_style_choices = config.get("headword_style", {}).get("choices", ["bold", "large", "small_caps"])
        self._head_style_weights = config.get("headword_style", {}).get("weights", [4, 3, 3])
        self._sep_choices = config.get("entry_separator", {}).get("choices", ["rule", "gap", "dot_leader"])
        self._sep_weights = config.get("entry_separator", {}).get("weights", [3, 4, 3])
        self._sep_rule_width = config.get("entry_separator", {}).get("rule_width_px", [1, 2])
        self._sep_rule_color_range = config.get("entry_separator", {}).get("rule_color", [100, 180])
        self._index_cfg = config.get("index", {})
        self._layout_cfg = config.get("layout", {})

    # Font name fragments that indicate decorative/script/handwriting faces
    # unsuitable for dense body text at small sizes.
    _DECORATIVE_KEYWORDS = {
        "hand",
        "script",
        "cursive",
        "calligra",
        "brush",
        "casual",
        "display",
        "deco",
        "ornament",
        "swash",
        "grunge",
        "sketch",
        "comic",
        "cartoon",
        "blackletter",
        "gothic",
        "fraktur",
        "crayon",
        "writing",
        "video",
        "napalm",
        "operation",
        "erratic",
        "plox",
        "gib",
        "coconut",
        "gotfridus",
        "pixel",
        "raster",
        "rainbow",
        "stamp",
        "scabber",
        "unsightly",
        "forge",
    }

    @classmethod
    def _is_decorative(cls, stem: str) -> bool:
        low = stem.lower()
        return any(kw in low for kw in cls._DECORATIVE_KEYWORDS)

    @staticmethod
    def _discover_fonts(font_dirs: list[Path]) -> dict[str, dict[str, Path]]:
        """Build {family_stem: {"regular": path, "bold": path}} map."""
        families: dict[str, dict[str, Path]] = {}
        for d in font_dirs:
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() not in (".ttf", ".otf"):
                    continue
                if EntryRenderer._is_decorative(f.stem):
                    continue
                stem = f.stem
                is_bold = "bold" in stem.lower()
                family = stem
                for suffix in [
                    "-Bold",
                    "-Regular",
                    "-Italic",
                    "-Light",
                    "-Medium",
                    "Bold",
                    "Regular",
                    "Italic",
                    "Light",
                    "Medium",
                ]:
                    family = family.replace(suffix, "")
                family = family.rstrip("-_ ")
                if not family:
                    family = stem
                if family not in families:
                    families[family] = {}
                key = "bold" if is_bold else "regular"
                families[family][key] = f
        # Ensure both "regular" and "bold" are present in every family
        for fam in families:
            if "regular" not in families[fam] and "bold" in families[fam]:
                families[fam]["regular"] = families[fam]["bold"]
            if "bold" not in families[fam] and "regular" in families[fam]:
                families[fam]["bold"] = families[fam]["regular"]
        return families

    def sample_page_style(self, rng: np.random.Generator) -> dict:
        if not self._font_paths:
            raise RuntimeError("No fonts found. Check font paths in config.")
        family_keys = list(self._font_paths.keys())
        family_name = family_keys[int(rng.integers(len(family_keys)))]
        family = self._font_paths[family_name]

        base_size = int(rng.integers(self._base_size_range[0], self._base_size_range[1] + 1))
        head_scale = float(rng.uniform(*self._head_scale_range))
        head_size = max(base_size + 1, int(base_size * head_scale))

        w = np.array(self._head_style_weights, dtype=float)
        w /= w.sum()
        head_style = str(self._head_style_choices[int(rng.choice(len(self._head_style_choices), p=w))])

        w2 = np.array(self._sep_weights, dtype=float)
        w2 /= w2.sum()
        sep_style = str(self._sep_choices[int(rng.choice(len(self._sep_choices), p=w2))])

        extra_px_range = self._layout_cfg.get("line_spacing_extra", [0, 2])
        extra_px = int(rng.integers(extra_px_range[0], extra_px_range[1] + 1))

        return {
            "family_name": family_name,
            "regular_path": family["regular"],
            "bold_path": family["bold"],
            "base_size": base_size,
            "head_size": head_size,
            "head_style": head_style,
            "sep_style": sep_style,
            "extra_px": extra_px,
        }

    def measure_definition_entry(self, headword: str, definition: str, col: ColumnRect, style: dict) -> int:
        body_font = _load_font(style["regular_path"], style["base_size"])
        if style["head_style"] == "bold":
            head_font = _load_font(style["bold_path"], style["base_size"])
        elif style["head_style"] == "large":
            head_font = _load_font(style["regular_path"], style["head_size"])
        else:
            head_font = _load_font(style["regular_path"], style["base_size"])
        return _entry_height_definition(
            headword, definition, col, body_font, head_font, style["head_style"], style["extra_px"]
        )

    def measure_index_entry(self, col: ColumnRect, style: dict) -> int:
        body_font = _load_font(style["regular_path"], style["base_size"])
        return _entry_height_index(col, body_font, style["extra_px"])

    def render_definition_entry(
        self,
        draw: ImageDraw.Draw,
        headword: str,
        definition: str,
        col: ColumnRect,
        cursor: ColumnCursor,
        style: dict,
        text_color: tuple[int, int, int],
        block_id: int,
        rng: np.random.Generator,
    ) -> list[RenderedLine]:
        rendered: list[RenderedLine] = []
        body_font = _load_font(style["regular_path"], style["base_size"])

        # Headword
        if style["head_style"] == "bold":
            hfont = _load_font(style["bold_path"], style["base_size"])
            hw_text = headword
        elif style["head_style"] == "large":
            hfont = _load_font(style["regular_path"], style["head_size"])
            hw_text = headword
        else:  # small_caps
            hfont = _load_font(style["regular_path"], style["base_size"])
            hw_text = headword.upper()

        h_line_h = _line_height(hfont, style["extra_px"])
        font_size_px = getattr(hfont, "size", style["base_size"])
        hw_lines = wrap_text(hw_text, hfont, col.width) or [hw_text]
        for hw_line in hw_lines:
            if cursor.is_full:
                break
            y = cursor.current_y
            draw.text((col.x1, y), hw_line, font=hfont, fill=text_color)
            x1s, x2s = compute_word_positions(hw_line, hfont, col.x1)
            x2_px = col.x1 + int(hfont.getlength(hw_line))
            rendered.append(
                RenderedLine(
                    text=hw_line,
                    block_id=block_id,
                    region_type="heading",
                    x1_px=col.x1,
                    y1_px=y,
                    x2_px=x2_px,
                    y2_px=y + h_line_h,
                    words=hw_line.split(),
                    word_x1s=x1s,
                    word_x2s=x2s,
                    font_family=style["family_name"],
                    font_size_px=font_size_px,
                    text_color_rgb=list(text_color),
                )
            )
            cursor.advance(h_line_h)

        # Definition lines
        b_line_h = _line_height(body_font, style["extra_px"])
        def_lines = wrap_text(definition, body_font, col.width)
        for line_text in def_lines:
            if cursor.is_full:
                break
            y = cursor.current_y
            draw.text((col.x1, y), line_text, font=body_font, fill=text_color)
            lx1s, lx2s = compute_word_positions(line_text, body_font, col.x1)
            lx2_px = col.x1 + int(body_font.getlength(line_text))
            rendered.append(
                RenderedLine(
                    text=line_text,
                    block_id=block_id,
                    region_type="body",
                    x1_px=col.x1,
                    y1_px=y,
                    x2_px=lx2_px,
                    y2_px=y + b_line_h,
                    words=line_text.split(),
                    word_x1s=lx1s,
                    word_x2s=lx2s,
                    font_family=style["family_name"],
                    font_size_px=style["base_size"],
                    text_color_rgb=list(text_color),
                )
            )
            cursor.advance(b_line_h)

        self._draw_separator(draw, col, cursor, style, rng)
        return rendered

    def render_index_entry(
        self,
        draw: ImageDraw.Draw,
        headword: str,
        col: ColumnRect,
        cursor: ColumnCursor,
        style: dict,
        text_color: tuple[int, int, int],
        block_id: int,
        rng: np.random.Generator,
    ) -> list[RenderedLine]:
        body_font = _load_font(style["regular_path"], style["base_size"])
        b_line_h = _line_height(body_font, style["extra_px"])

        # Build index line: "headword  42, 107, 203"
        refs_range = self._index_cfg.get("page_refs_count", [2, 6])
        n_refs = int(rng.integers(refs_range[0], refs_range[1] + 1))
        prange = self._index_cfg.get("page_number_range", [1, 500])
        page_nums = sorted(int(x) for x in rng.integers(prange[0], prange[1] + 1, size=n_refs).tolist())
        line_text = headword + "  " + ", ".join(str(p) for p in page_nums)

        # Drop trailing page refs until it fits; if the headword alone is too wide, skip entry
        while body_font.getlength(line_text) > col.width and "  " in line_text:
            line_text = line_text.rsplit(",", 1)[0] if "," in line_text.split("  ", 1)[-1] else headword
            if line_text == headword:
                break
        if body_font.getlength(line_text) > col.width:
            return []  # headword alone doesn't fit — skip this entry

        y = cursor.current_y
        draw.text((col.x1, y), line_text, font=body_font, fill=text_color)
        x1s, x2s = compute_word_positions(line_text, body_font, col.x1)
        x2_px = col.x1 + int(body_font.getlength(line_text))
        cursor.advance(b_line_h)

        return [
            RenderedLine(
                text=line_text,
                block_id=block_id,
                region_type="body",
                x1_px=col.x1,
                y1_px=y,
                x2_px=x2_px,
                y2_px=y + b_line_h,
                words=line_text.split(),
                word_x1s=x1s,
                word_x2s=x2s,
                font_family=style["family_name"],
                font_size_px=style["base_size"],
                text_color_rgb=list(text_color),
            )
        ]

    def _draw_separator(
        self,
        draw: ImageDraw.Draw,
        col: ColumnRect,
        cursor: ColumnCursor,
        style: dict,
        rng: np.random.Generator,
    ) -> None:
        sep = style["sep_style"]
        if cursor.is_full:
            return
        gap_range = self._layout_cfg.get("entry_gap", [2, 8])
        gap = int(rng.integers(gap_range[0], gap_range[1] + 1))
        if sep == "rule":
            rw = int(rng.integers(self._sep_rule_width[0], self._sep_rule_width[1] + 1))
            rc = int(rng.integers(self._sep_rule_color_range[0], self._sep_rule_color_range[1] + 1))
            y = cursor.current_y + gap // 2
            if y < col.y2:
                draw.line([(col.x1, y), (col.x2, y)], fill=(rc, rc, rc), width=rw)
        elif sep == "dot_leader":
            y = cursor.current_y + gap // 2
            if y < col.y2:
                body_font = _load_font(style["regular_path"], style["base_size"])
                dot_w = body_font.getlength(".")
                x = float(col.x1)
                while x + dot_w <= col.x2:
                    draw.text((int(x), y), ".", font=body_font, fill=(180, 180, 180))
                    x += dot_w * 2
        cursor.advance(gap)
