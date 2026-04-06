"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import numpy as np
from synthtiger import components

from layouts import Grid, GridStack, Layout

from .readers import _READER_TYPES, LiteralTextCursor, TextCursor
from .textbox import TextBox


def _relative_luminance(r, g, b):
    def channel(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def _make_adaptive_color(color_config: dict, gray_range: list[int], lum: float) -> components.Switch:
    """Build a Switch(Gray) component whose prob is forced to 1.0 on dark backgrounds."""
    args = {**color_config.get("args", {}), "gray": gray_range}
    prob = color_config.get("prob", 0)
    # 0.179 is the WCAG crossover: (L+0.05)² = 0.0525 → L ≈ 0.179.
    # Below this, dark text [0,64] achieves lower contrast than light text [191,255];
    # above it, dark text wins. Using 0.5 here was wrong — see SYNTHDOG-VALIDATION.md Thread 1.
    if lum < 0.179:
        prob = 1.0
    return components.Switch(components.Gray(), prob=prob, args=args)


def _compute_layout_bbox(width: int, height: int, margin: list[float]) -> list[float]:
    """Sample 4 independent margins and return [left, top, w, h] for the content area."""
    layout_left = width * np.random.uniform(margin[0], margin[1])
    layout_right = width * np.random.uniform(margin[0], margin[1])
    layout_top = height * np.random.uniform(margin[0], margin[1])
    layout_bottom = height * np.random.uniform(margin[0], margin[1])
    layout_width = max(width - layout_left - layout_right, 0)
    layout_height = max(height - layout_top - layout_bottom, 0)
    return [layout_left, layout_top, layout_width, layout_height]


_LAYOUT_TYPES: dict[str, type] = {
    "grid_stack": GridStack,
}


class Content:
    def __init__(self, config):
        self.margin = config.get("margin", [0, 0.1])

        # Choose text reader based on configuration
        text_config = config.get("text", {})
        reader_type = text_config.get("type", "file")
        reader_cls = _READER_TYPES[reader_type]  # KeyError = clear signal of bad config
        reader_kwargs = {k: v for k, v in text_config.items() if k != "type"}
        self.reader: TextCursor = reader_cls(**reader_kwargs)

        self.font = components.BaseFont(**config.get("font", {}))
        layout_config = config.get("layout", {})
        layout_type = layout_config.get("type", "grid_stack")
        layout_cls = _LAYOUT_TYPES[layout_type]
        self.layout: Layout = layout_cls(layout_config)
        self.textbox = TextBox(config.get("textbox", {}))
        self.textbox_color_config = config.get("textbox_color", {})
        self.content_color_config = config.get("content_color", {})
        self.text_sprinkle = components.Switch(
            components.TextSprinkle(),
            **config.get("text_sprinkle", {}),
        )

        # Zone configs
        self.page_header_cfg = config.get("page_header", {})
        self.page_footer_cfg = config.get("page_footer", {})
        self.footnote_cfg = config.get("footnote", {})
        section_heading_cfg = config.get("section_heading", {})
        self.section_heading_cfg = section_heading_cfg
        self.heading_prob = section_heading_cfg.get("prob", 0.0)
        self.heading_font = components.BaseFont(**section_heading_cfg.get("font", config.get("font", {})))

    def close(self):
        if hasattr(self.reader, "close"):
            self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _render_cells(
        self,
        cells: list[tuple],
        cursor,
        font,
        region_type: str,
        next_block_id: int,
        block_region_types: dict[int, str],
        text_layers: list,
        texts: list,
        block_ids: list,
        words_per_line: list,
    ) -> tuple[int, int, int]:
        """Render a sequence of layout cells and append results to the output lists.

        Args:
            cells: Sequence of (bbox, align, col_key) tuples from a Grid layout.
                   col_key can be any hashable — zones pass plain col_idx (int),
                   the body loop passes (grid_idx, col_idx) tuples.
            cursor: TextCursor or LiteralTextCursor to read text from.
            font: Sampled BaseFont instance.
            region_type: Annotation region type string (e.g. "body", "header").
            next_block_id: Next available block ID counter.
            block_region_types: Dict to populate with {block_id: region_type}.
            text_layers: Output list (mutated in-place).
            texts: Output list (mutated in-place).
            block_ids: Output list (mutated in-place).
            words_per_line: Output list (mutated in-place).

        Returns:
            (next_block_id, null_count, total_count)
        """
        col_key_to_block_id: dict = {}
        null_count = 0
        total_count = 0

        for cell_bbox, align, col_key in cells:
            total_count += 1
            x, y, w, h = cell_bbox
            text_layer, text, word_local_data = self.textbox.generate((w, h), cursor, font)

            if text_layer is None:
                null_count += 1
                continue

            text_layer.center = (x + w / 2, y + h / 2)
            if align == "left":
                text_layer.left = x
            if align == "right":
                text_layer.right = x + w

            if col_key not in col_key_to_block_id:
                col_key_to_block_id[col_key] = next_block_id
                block_region_types[next_block_id] = region_type
                next_block_id += 1

            text_layers.append(text_layer)
            texts.append(text)
            block_ids.append(col_key_to_block_id[col_key])
            words_per_line.append(word_local_data)

        return next_block_id, null_count, total_count

    def _render_zone(
        self,
        cfg: dict,
        zone_bbox: list[float],
        region_type: str,
        canvas_ref: float,
        next_block_id: int,
        block_region_types: dict[int, str],
        font_override=None,
        use_page_number: bool = False,
    ) -> tuple[list, list[str], list[int], list[list[dict]], int, int, int]:
        """Render a 1-row zone and return accumulated data.

        Args:
            cfg: Zone config dict (text_scale, max_col, etc.)
            zone_bbox: [x, y, w, h] for the zone area
            region_type: Annotation region type string
            canvas_ref: min(canvas_w, canvas_h) used as text-scale reference dimension
            next_block_id: Next available block ID counter
            block_region_types: Dict to populate with {block_id: region_type}
            font_override: Optional BaseFont to use instead of self.font
            use_page_number: If True, use a LiteralTextCursor with a random page number

        Returns:
            (text_layers, texts, block_ids, words_per_line,
             next_block_id, null_count, total_count)
        """
        zone_x, zone_y, zone_w, zone_h = zone_bbox
        if zone_w <= 0 or zone_h <= 0:
            return [], [], [], [], next_block_id, 0, 0

        text_scale_range = cfg.get("text_scale", [0.5, 0.9])
        # Convert canvas-relative scale to an absolute font size, then to zone-relative
        text_size = canvas_ref * np.random.uniform(text_scale_range[0], text_scale_range[1])
        zone_min = min(zone_w, zone_h)
        # Cap at 0.99 × zone_min so Grid can always fit at least 1 row
        zone_text_scale = min(text_size / zone_min, 0.99)

        max_col = cfg.get("max_col", 3)
        grid = Grid({"max_row": 1, "max_col": max_col, "align": cfg.get("align", ["left", "right", "center"])})
        layout = grid.generate(zone_bbox, fill_range=(0.5, 1.0), text_scale_range=(zone_text_scale, zone_text_scale))
        if layout is None:
            return [], [], [], [], next_block_id, 0, 1  # 1 total, 1 null

        cursor: TextCursor = LiteralTextCursor(str(np.random.randint(1, 500))) if use_page_number else self.reader

        font_sampler = font_override if font_override is not None else self.font
        font = font_sampler.sample()

        text_layers: list = []
        texts: list[str] = []
        block_ids: list[int] = []
        words_per_line: list[list[dict]] = []

        next_block_id, null_count, total_count = self._render_cells(
            list(layout),
            cursor,
            font,
            region_type,
            next_block_id,
            block_region_types,
            text_layers,
            texts,
            block_ids,
            words_per_line,
        )

        return text_layers, texts, block_ids, words_per_line, next_block_id, null_count, total_count

    def generate(self, size, bg_color=(255, 255, 255)):
        width, height = size

        lum = _relative_luminance(*bg_color)
        # WCAG crossover is ≈ 0.179, not 0.5 — see SYNTHDOG-VALIDATION.md Thread 1.
        # Empirical analysis showed 21% of samples fell in the wrong zone under the old threshold,
        # with worst-case contrast of 1.04:1 (near-invisible text).
        gray_range = [0, 64] if lum > 0.179 else [191, 255]

        textbox_color = _make_adaptive_color(self.textbox_color_config, gray_range, lum)
        content_color = _make_adaptive_color(self.content_color_config, gray_range, lum)
        layout_bbox = list(_compute_layout_bbox(width, height, self.margin))

        text_layers, texts, block_ids, words_per_line = [], [], [], []
        block_region_types: dict[int, str] = {}
        textbox_total_count = 0
        textbox_null_count = 0
        next_block_id = 0

        # Reference dimension for text-scale calculations (same convention as GridStack)
        canvas_ref = float(min(width, height))

        # Advance reader to a random word boundary once, shared by all zones and body
        self.reader.move(np.random.randint(len(self.reader)))
        for _ in range(len(self.reader)):
            if self.reader.get().isspace():
                break
            self.reader.next()
        for _ in range(len(self.reader)):
            if not self.reader.get().isspace():
                break
            self.reader.next()

        # ── Page header ───────────────────────────────────────────────────────
        if np.random.rand() < self.page_header_cfg.get("prob", 0.0):
            h_frac = np.random.uniform(*self.page_header_cfg.get("height", [0.04, 0.08]))
            zone_h = min(height * h_frac, layout_bbox[3])
            if zone_h > 0:
                zone_bbox = [layout_bbox[0], layout_bbox[1], layout_bbox[2], zone_h]
                zl, zt, zbi, zwpl, next_block_id, znull, ztot = self._render_zone(
                    self.page_header_cfg, zone_bbox, "header", canvas_ref, next_block_id, block_region_types
                )
                text_layers.extend(zl)
                texts.extend(zt)
                block_ids.extend(zbi)
                words_per_line.extend(zwpl)
                textbox_null_count += znull
                textbox_total_count += ztot
                layout_bbox[1] += zone_h
                layout_bbox[3] = max(layout_bbox[3] - zone_h, 0)

        # ── Page footer ───────────────────────────────────────────────────────
        if np.random.rand() < self.page_footer_cfg.get("prob", 0.0):
            h_frac = np.random.uniform(*self.page_footer_cfg.get("height", [0.04, 0.08]))
            zone_h = min(height * h_frac, layout_bbox[3])
            if zone_h > 0:
                footer_top = layout_bbox[1] + layout_bbox[3] - zone_h
                zone_bbox = [layout_bbox[0], footer_top, layout_bbox[2], zone_h]
                pn_cfg = self.page_footer_cfg.get("page_number", {})
                use_pn = np.random.rand() < pn_cfg.get("prob", 0.0)
                zl, zt, zbi, zwpl, next_block_id, znull, ztot = self._render_zone(
                    self.page_footer_cfg,
                    zone_bbox,
                    "footer",
                    canvas_ref,
                    next_block_id,
                    block_region_types,
                    use_page_number=use_pn,
                )
                text_layers.extend(zl)
                texts.extend(zt)
                block_ids.extend(zbi)
                words_per_line.extend(zwpl)
                textbox_null_count += znull
                textbox_total_count += ztot
                layout_bbox[3] = max(layout_bbox[3] - zone_h, 0)

        # ── Footnote ─────────────────────────────────────────────────────────
        if np.random.rand() < self.footnote_cfg.get("prob", 0.0):
            h_frac = np.random.uniform(*self.footnote_cfg.get("height", [0.05, 0.12]))
            zone_h = min(layout_bbox[3] * h_frac, layout_bbox[3])
            if zone_h > 0:
                footnote_top = layout_bbox[1] + layout_bbox[3] - zone_h
                zone_bbox = [layout_bbox[0], footnote_top, layout_bbox[2], zone_h]
                zl, zt, zbi, zwpl, next_block_id, znull, ztot = self._render_zone(
                    self.footnote_cfg, zone_bbox, "footnote", canvas_ref, next_block_id, block_region_types
                )
                text_layers.extend(zl)
                texts.extend(zt)
                block_ids.extend(zbi)
                words_per_line.extend(zwpl)
                textbox_null_count += znull
                textbox_total_count += ztot
                layout_bbox[3] = max(layout_bbox[3] - zone_h, 0)

        # ── Section heading ───────────────────────────────────────────────────
        if np.random.rand() < self.heading_prob:
            h_frac = np.random.uniform(*self.section_heading_cfg.get("height", [0.06, 0.14]))
            zone_h = min(layout_bbox[3] * h_frac, layout_bbox[3])
            if zone_h > 0:
                zone_bbox = [layout_bbox[0], layout_bbox[1], layout_bbox[2], zone_h]
                zl, zt, zbi, zwpl, next_block_id, znull, ztot = self._render_zone(
                    self.section_heading_cfg,
                    zone_bbox,
                    "heading",
                    canvas_ref,
                    next_block_id,
                    block_region_types,
                    font_override=self.heading_font,
                )
                text_layers.extend(zl)
                texts.extend(zt)
                block_ids.extend(zbi)
                words_per_line.extend(zwpl)
                textbox_null_count += znull
                textbox_total_count += ztot
                layout_bbox[1] += zone_h
                layout_bbox[3] = max(layout_bbox[3] - zone_h, 0)

        # ── Body GridStack ────────────────────────────────────────────────────
        layouts = self.layout.generate(layout_bbox)

        for grid_idx, layout in enumerate(layouts):
            font = self.font.sample()
            cells = [(bbox, align, (grid_idx, col_idx)) for bbox, align, col_idx in layout]
            next_block_id, gnull, gtot = self._render_cells(
                cells,
                self.reader,
                font,
                "body",
                next_block_id,
                block_region_types,
                text_layers,
                texts,
                block_ids,
                words_per_line,
            )
            textbox_null_count += gnull
            textbox_total_count += gtot

        # Apply color: content_color (uniform) takes priority; if it does not fire,
        # textbox_color applies per-line variation instead. The two modes are mutually
        # exclusive so neither silently discards the other's work.
        content_meta = content_color.sample()
        if content_meta["state"]:
            content_color.apply(text_layers, meta=content_meta)
        else:
            for text_layer in text_layers:
                textbox_color.apply([text_layer])

        for text_layer in text_layers:
            self.text_sprinkle.apply([text_layer])

        return (
            text_layers,
            texts,
            block_ids,
            words_per_line,
            block_region_types,
            textbox_null_count,
            textbox_total_count,
        )
