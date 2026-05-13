"""Pillow-based synthetic table renderer.

Renders a TableStructure + content grid to a PIL Image, returning the image
alongside per-word bounding boxes for the JSON sidecar.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont
from table_structure import BorderStyle, TableStructure


@dataclass
class WordBox:
    """A single word with its pixel bounding box [x1, y1, x2, y2]."""

    text: str
    box: list[int]


@dataclass
class LineBox:
    """A line of text with its pixel bounding box [x1, y1, x2, y2]."""

    text: str
    box: list[int]


def _get_font(size: int) -> ImageFont.ImageFont:
    """Load a font, falling back to the default PIL bitmap font."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
            )
        except OSError:
            return ImageFont.load_default()


def _measure_text(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont
) -> tuple[int, int]:
    """Return (width, height) of rendered text."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def render_table(
    structure: TableStructure,
    content_grid: list[list[str]],
    *,
    font_size: int = 14,
    cell_padding: int = 6,
    min_cell_width: int = 40,
    min_cell_height: int = 24,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    text_color: tuple[int, int, int] = (0, 0, 0),
    border_color: tuple[int, int, int] = (80, 80, 80),
    header_bg: tuple[int, int, int] = (220, 230, 242),
) -> tuple[Image.Image, list[WordBox], list[LineBox]]:
    """Render a table to a PIL Image.

    Args:
        structure: Table structure (rows, cols, spans, border style).
        content_grid: rows×cols grid of cell text strings.
        font_size: Base font size in pixels.
        cell_padding: Padding inside each cell.
        min_cell_width: Minimum cell width in pixels.
        min_cell_height: Minimum cell height in pixels.
        bg_color: Background RGB colour.
        text_color: Cell text RGB colour.
        border_color: Border line RGB colour.
        header_bg: Header row background RGB colour.

    Returns:
        Tuple of (image, word_boxes, line_boxes).
    """
    rows, cols = structure.rows, structure.cols
    font = _get_font(font_size)

    # Build occupancy map: extension positions → ('x'|'y', anchor_row, anchor_col)
    ext: dict[tuple[int, int], tuple[str, int, int]] = {}
    for span in structure.spans:
        for dr in range(span.rowspan):
            for dc in range(span.colspan):
                if dr == 0 and dc == 0:
                    continue
                ext[(span.row + dr, span.col + dc)] = (
                    "x" if dr == 0 else "y",
                    span.row,
                    span.col,
                )

    # Compute per-column widths and per-row heights from content
    col_widths = [min_cell_width] * cols
    row_heights = [min_cell_height] * rows

    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)

    for r in range(rows):
        for c in range(cols):
            if (r, c) in ext:
                continue
            text = (
                content_grid[r][c]
                if r < len(content_grid) and c < len(content_grid[r])
                else ""
            )
            if text:
                tw, th = _measure_text(dummy_draw, text, font)
                col_widths[c] = max(col_widths[c], tw + cell_padding * 2)
                row_heights[r] = max(row_heights[r], th + cell_padding * 2)

    # Build cumulative offsets
    col_x = [0] * (cols + 1)
    for c in range(cols):
        col_x[c + 1] = col_x[c] + col_widths[c]
    row_y = [0] * (rows + 1)
    for r in range(rows):
        row_y[r + 1] = row_y[r] + row_heights[r]

    img_w = col_x[cols] + 2
    img_h = row_y[rows] + 2
    img = Image.new("RGB", (img_w, img_h), bg_color)
    draw = ImageDraw.Draw(img)

    bs = structure.border_style
    draw_outer = bs in (BorderStyle.FULL, BorderStyle.OUTER_ONLY)
    draw_inner = bs in (BorderStyle.FULL, BorderStyle.INNER_ONLY)

    word_boxes: list[WordBox] = []
    line_boxes: list[LineBox] = []

    # Render cells
    for r in range(rows):
        for c in range(cols):
            if (r, c) in ext:
                continue

            # Determine span extents
            span_obj = next(
                (s for s in structure.spans if s.row == r and s.col == c), None
            )
            rowspan = span_obj.rowspan if span_obj else 1
            colspan = span_obj.colspan if span_obj else 1

            x1 = col_x[c]
            y1 = row_y[r]
            x2 = col_x[c + colspan]
            y2 = row_y[r + rowspan]

            # Background fill for header
            if structure.has_header and r == 0:
                draw.rectangle([x1, y1, x2 - 1, y2 - 1], fill=header_bg)

            # Draw borders
            is_left = c == 0
            is_top = r == 0
            is_right = c + colspan == cols
            is_bottom = r + rowspan == rows

            if draw_outer and is_top:
                draw.line([(x1, y1), (x2, y1)], fill=border_color)
            if draw_outer and is_bottom:
                draw.line([(x1, y2), (x2, y2)], fill=border_color)
            if draw_outer and is_left:
                draw.line([(x1, y1), (x1, y2)], fill=border_color)
            if draw_outer and is_right:
                draw.line([(x2, y1), (x2, y2)], fill=border_color)
            if draw_inner and not is_top:
                draw.line([(x1, y1), (x2, y1)], fill=border_color)
            if draw_inner and not is_bottom:
                draw.line([(x1, y2), (x2, y2)], fill=border_color)
            if draw_inner and not is_left:
                draw.line([(x1, y1), (x1, y2)], fill=border_color)
            if draw_inner and not is_right:
                draw.line([(x2, y1), (x2, y2)], fill=border_color)

            # Draw text and record bboxes
            text = (
                content_grid[r][c]
                if r < len(content_grid) and c < len(content_grid[r])
                else ""
            )
            if text:
                tx = x1 + cell_padding
                ty = y1 + cell_padding
                draw.text((tx, ty), text, fill=text_color, font=font)

                # Line box
                tw, th = _measure_text(draw, text, font)
                line_box = [tx, ty, tx + tw, ty + th]
                line_boxes.append(LineBox(text=text, box=line_box))

                # Word-level boxes (simple horizontal split by spaces)
                words = text.split()
                cursor_x = tx
                for word in words:
                    ww, wh = _measure_text(draw, word, font)
                    word_boxes.append(
                        WordBox(text=word, box=[cursor_x, ty, cursor_x + ww, ty + wh])
                    )
                    # Advance by word width + one space
                    space_w, _ = _measure_text(draw, " ", font)
                    cursor_x += ww + space_w

    return img, word_boxes, line_boxes


def augment_image(
    img: Image.Image,
    rng: random.Random,
    *,
    shear_range: float = 0.05,
    rotation_range: float = 3.0,
    padding: int = 8,
) -> Image.Image:
    """Apply a random affine augmentation (shear + rotation + padding).

    Uses Pillow's built-in affine transform — no scikit-image dependency.

    Args:
        img: Source PIL image.
        rng: Seeded random instance.
        shear_range: Maximum shear coefficient (±).
        rotation_range: Maximum rotation in degrees (±).
        padding: Pixels of white padding added before augmentation.

    Returns:
        Augmented PIL image (same mode as input).
    """
    # Add padding so border pixels aren't cropped after transform
    padded_w = img.width + padding * 2
    padded_h = img.height + padding * 2
    canvas = Image.new(img.mode, (padded_w, padded_h), (255, 255, 255))
    canvas.paste(img, (padding, padding))

    # Build affine matrix: rotation + shear
    angle = math.radians(rng.uniform(-rotation_range, rotation_range))
    shear = rng.uniform(-shear_range, shear_range)

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    cx, cy = padded_w / 2, padded_h / 2

    # Pillow affine: (a,b,c, d,e,f) where output(x,y) = input(a*x+b*y+c, d*x+e*y+f)
    a = cos_a
    b = -sin_a + shear
    c = cx - a * cx - b * cy
    d = sin_a
    e = cos_a
    f = cy - d * cx - e * cy

    return canvas.transform(
        (padded_w, padded_h),
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.BILINEAR,
        fillcolor=(255, 255, 255),
    )
