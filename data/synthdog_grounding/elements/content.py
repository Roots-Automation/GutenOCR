"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

from pathlib import Path

import numpy as np
from annotations import _linearize_channel
from synthtiger import components

from layouts import Grid, GridStack, Layout

from .readers import _READER_TYPES, LiteralTextCursor, TextCursor
from .textbox import TextBox


def _relative_luminance(r, g, b):
    return 0.2126 * _linearize_channel(r) + 0.7152 * _linearize_channel(g) + 0.0722 * _linearize_channel(b)


def _font_family_from_path(path: str) -> str:
    """Font identifier for per-line error-analysis metadata (filename stem, not the full path)."""
    return Path(path).stem


def _make_adaptive_color(color_config: dict, gray_range: list[int], lum: float) -> components.Switch:
    """Build a Switch(Gray) component whose prob is forced to 1.0 on dark backgrounds."""
    args = {**color_config.get("args", {}), "gray": gray_range}
    prob = color_config.get("prob", 0)
    if lum < 0.179:  # force light text on dark backgrounds (see crossover derivation above)
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
        line_font_info: list,
    ) -> tuple[int, int, int]:
        """Render a sequence of layout cells, appending results to the output lists.

        Returns (next_block_id, null_count, total_count).
        """
        col_key_to_block_id: dict = {}
        null_count = 0
        total_count = 0
        font_family = _font_family_from_path(font["path"])

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
            # font size is exactly the cell height passed to TextBox.generate (see textbox.py).
            line_font_info.append({"font_family": font_family, "font_size_px": int(round(h))})

        return next_block_id, null_count, total_count

    def _render_zone(
        self,
        cfg: dict,
        zone_bbox: list[float],
        region_type: str,
        canvas_ref: float,
        next_block_id: int,
        block_region_types: dict[int, str],
        text_layers: list,
        texts: list,
        block_ids: list,
        words_per_line: list,
        line_font_info: list,
        font_override=None,
        use_page_number: bool = False,
    ) -> tuple[int, int, int]:
        """Render a 1-row zone, appending results to the output lists.

        Returns (next_block_id, null_count, total_count).
        """
        zone_x, zone_y, zone_w, zone_h = zone_bbox
        if zone_w <= 0 or zone_h <= 0:
            return next_block_id, 0, 0

        text_scale_range = cfg.get("text_scale", [0.5, 0.9])
        text_size = canvas_ref * np.random.uniform(text_scale_range[0], text_scale_range[1])
        zone_min = min(zone_w, zone_h)
        # Cap at 0.99 × zone_min so Grid can always fit at least 1 row.
        zone_text_scale = min(text_size / zone_min, 0.99)

        max_col = cfg.get("max_col", 3)
        grid = Grid({"max_row": 1, "max_col": max_col, "align": cfg.get("align", ["left", "right", "center"])})
        layout = grid.generate(zone_bbox, fill_range=(0.5, 1.0), text_scale_range=(zone_text_scale, zone_text_scale))
        if layout is None:
            return next_block_id, 0, 0

        cursor: TextCursor = LiteralTextCursor(str(np.random.randint(1, 500))) if use_page_number else self.reader
        font = (font_override if font_override is not None else self.font).sample()

        return self._render_cells(
            layout,
            cursor,
            font,
            region_type,
            next_block_id,
            block_region_types,
            text_layers,
            texts,
            block_ids,
            words_per_line,
            line_font_info,
        )

    def generate(self, size, bg_color=(255, 255, 255)):
        width, height = size

        lum = _relative_luminance(*bg_color)
        # 0.179 = WCAG black-vs-white crossover: √(1.05×0.05) − 0.05.
        # Below it, light text has better contrast; above it, dark text wins.
        # Ranges are kept away from mid-gray to preserve contrast headroom after
        # perspective warp, elastic distortion, and blur degrade rendered contrast.
        gray_range = [0, 40] if lum > 0.179 else [215, 255]

        textbox_color = _make_adaptive_color(self.textbox_color_config, gray_range, lum)
        content_color = _make_adaptive_color(self.content_color_config, gray_range, lum)
        layout_bbox = list(_compute_layout_bbox(width, height, self.margin))

        # Each zone collects into its own bucket so we can merge in reading order
        # (header → heading → body → footnote → footer) at the end, independent
        # of the space-allocation order required for correct layout_bbox arithmetic.
        def _bucket():
            return [], [], [], [], []

        h_tl, h_tx, h_bi, h_wpl, h_fi = _bucket()  # header
        ft_tl, ft_tx, ft_bi, ft_wpl, ft_fi = _bucket()  # footer
        fn_tl, fn_tx, fn_bi, fn_wpl, fn_fi = _bucket()  # footnote
        hd_tl, hd_tx, hd_bi, hd_wpl, hd_fi = _bucket()  # heading
        bd_tl, bd_tx, bd_bi, bd_wpl, bd_fi = _bucket()  # body

        block_region_types: dict[int, str] = {}
        textbox_total_count = 0
        textbox_null_count = 0
        next_block_id = 0
        zones_rendered: list[str] = []

        canvas_ref = float(min(width, height))

        # Advance reader to a random word boundary once, shared by all zones and body.
        n = len(self.reader)
        self.reader.move(np.random.randint(n))
        for _ in range(n):
            if self.reader.get().isspace():
                break
            self.reader.next()
        for _ in range(n):
            if not self.reader.get().isspace():
                break
            self.reader.next()

        # ── Page header ───────────────────────────────────────────────────────
        if np.random.rand() < self.page_header_cfg.get("prob", 0.0):
            h_frac = np.random.uniform(*self.page_header_cfg.get("height", [0.04, 0.08]))
            zone_h = min(height * h_frac, layout_bbox[3])
            if zone_h > 0:
                zone_bbox = [layout_bbox[0], layout_bbox[1], layout_bbox[2], zone_h]
                block_id_before = next_block_id
                next_block_id, znull, ztot = self._render_zone(
                    self.page_header_cfg,
                    zone_bbox,
                    "header",
                    canvas_ref,
                    next_block_id,
                    block_region_types,
                    h_tl,
                    h_tx,
                    h_bi,
                    h_wpl,
                    h_fi,
                )
                if next_block_id > block_id_before:
                    zones_rendered.append("header")
                textbox_null_count += znull
                textbox_total_count += ztot
                layout_bbox[1] += zone_h
                layout_bbox[3] = max(layout_bbox[3] - zone_h, 0)

        # ── Page footer ───────────────────────────────────────────────────────
        # Space allocated now (before body) so layout_bbox height is correct,
        # but annotations are merged after body in reading order.
        if np.random.rand() < self.page_footer_cfg.get("prob", 0.0):
            h_frac = np.random.uniform(*self.page_footer_cfg.get("height", [0.04, 0.08]))
            zone_h = min(height * h_frac, layout_bbox[3])
            if zone_h > 0:
                footer_top = layout_bbox[1] + layout_bbox[3] - zone_h
                zone_bbox = [layout_bbox[0], footer_top, layout_bbox[2], zone_h]
                pn_cfg = self.page_footer_cfg.get("page_number", {})
                use_pn = np.random.rand() < pn_cfg.get("prob", 0.0)
                block_id_before = next_block_id
                next_block_id, znull, ztot = self._render_zone(
                    self.page_footer_cfg,
                    zone_bbox,
                    "footer",
                    canvas_ref,
                    next_block_id,
                    block_region_types,
                    ft_tl,
                    ft_tx,
                    ft_bi,
                    ft_wpl,
                    ft_fi,
                    use_page_number=use_pn,
                )
                if next_block_id > block_id_before:
                    zones_rendered.append("footer")
                textbox_null_count += znull
                textbox_total_count += ztot
                layout_bbox[3] = max(layout_bbox[3] - zone_h, 0)

        # ── Footnote ─────────────────────────────────────────────────────────
        # Space allocated now (before body), annotations merged after body.
        if np.random.rand() < self.footnote_cfg.get("prob", 0.0):
            h_frac = np.random.uniform(*self.footnote_cfg.get("height", [0.05, 0.12]))
            zone_h = min(layout_bbox[3] * h_frac, layout_bbox[3])
            if zone_h > 0:
                footnote_top = layout_bbox[1] + layout_bbox[3] - zone_h
                zone_bbox = [layout_bbox[0], footnote_top, layout_bbox[2], zone_h]
                block_id_before = next_block_id
                next_block_id, znull, ztot = self._render_zone(
                    self.footnote_cfg,
                    zone_bbox,
                    "footnote",
                    canvas_ref,
                    next_block_id,
                    block_region_types,
                    fn_tl,
                    fn_tx,
                    fn_bi,
                    fn_wpl,
                    fn_fi,
                )
                if next_block_id > block_id_before:
                    zones_rendered.append("footnote")
                textbox_null_count += znull
                textbox_total_count += ztot
                layout_bbox[3] = max(layout_bbox[3] - zone_h, 0)

        # ── Section heading ───────────────────────────────────────────────────
        if np.random.rand() < self.heading_prob:
            h_frac = np.random.uniform(*self.section_heading_cfg.get("height", [0.06, 0.14]))
            zone_h = min(layout_bbox[3] * h_frac, layout_bbox[3])
            if zone_h > 0:
                zone_bbox = [layout_bbox[0], layout_bbox[1], layout_bbox[2], zone_h]
                block_id_before = next_block_id
                next_block_id, znull, ztot = self._render_zone(
                    self.section_heading_cfg,
                    zone_bbox,
                    "heading",
                    canvas_ref,
                    next_block_id,
                    block_region_types,
                    hd_tl,
                    hd_tx,
                    hd_bi,
                    hd_wpl,
                    hd_fi,
                    font_override=self.heading_font,
                )
                if next_block_id > block_id_before:
                    zones_rendered.append("heading")
                textbox_null_count += znull
                textbox_total_count += ztot
                layout_bbox[1] += zone_h
                layout_bbox[3] = max(layout_bbox[3] - zone_h, 0)

        # ── Body GridStack ────────────────────────────────────────────────────
        layouts = self.layout.generate(layout_bbox)

        body_col_counts = []
        for grid_idx, layout in enumerate(layouts):
            font = self.font.sample()
            cells = [(bbox, align, (grid_idx, col_idx)) for bbox, align, col_idx in layout]
            body_col_counts.append(len({col_idx for _, _, col_idx in layout}))
            next_block_id, gnull, gtot = self._render_cells(
                cells,
                self.reader,
                font,
                "body",
                next_block_id,
                block_region_types,
                bd_tl,
                bd_tx,
                bd_bi,
                bd_wpl,
                bd_fi,
            )
            textbox_null_count += gnull
            textbox_total_count += gtot

        # ── Merge buckets in reading order: header → heading → body → footnote → footer ──
        text_layers, texts, block_ids, words_per_line, line_font_info = [], [], [], [], []
        for tl, tx, bi, wpl, fi in (
            (h_tl, h_tx, h_bi, h_wpl, h_fi),
            (hd_tl, hd_tx, hd_bi, hd_wpl, hd_fi),
            (bd_tl, bd_tx, bd_bi, bd_wpl, bd_fi),
            (fn_tl, fn_tx, fn_bi, fn_wpl, fn_fi),
            (ft_tl, ft_tx, ft_bi, ft_wpl, ft_fi),
        ):
            text_layers.extend(tl)
            texts.extend(tx)
            block_ids.extend(bi)
            words_per_line.extend(wpl)
            line_font_info.extend(fi)

        # Apply color: content_color (uniform) takes priority; if it does not fire,
        # textbox_color applies per-line variation instead. The two modes are mutually
        # exclusive so neither silently discards the other's work.
        content_meta = content_color.sample()
        text_color_rgbs: list[list[int]] = []
        if content_meta["state"]:
            text_color_mode = "uniform"
            content_color.apply(text_layers, meta=content_meta)
            try:
                c = content_meta["meta"]["rgb"]
                text_color_rgbs = [[int(c[0]), int(c[1]), int(c[2])]] * len(text_layers)
            except Exception:
                pass
        else:
            text_color_mode = "per_line"
            for text_layer in text_layers:
                layer_meta = textbox_color.sample()
                textbox_color.apply([text_layer], meta=layer_meta)
                try:
                    c = layer_meta["meta"]["rgb"]
                    text_color_rgbs.append([int(c[0]), int(c[1]), int(c[2])])
                except Exception:
                    pass

        # Summarise text colors as median RGB (sample-level provenance) while also
        # keeping the per-line list so each LineAnnotation can carry its own color.
        if text_color_rgbs:
            text_color_median_rgb = [
                int(np.median([c[0] for c in text_color_rgbs])),
                int(np.median([c[1] for c in text_color_rgbs])),
                int(np.median([c[2] for c in text_color_rgbs])),
            ]
        else:
            text_color_median_rgb = None

        # Guard against the color-extraction except-branch leaving this short.
        line_colors = text_color_rgbs if len(text_color_rgbs) == len(text_layers) else [None] * len(text_layers)

        for text_layer in text_layers:
            self.text_sprinkle.apply([text_layer])

        provenance = {
            "paper_luminance": round(float(lum), 4),
            "text_color_mode": text_color_mode,
            "text_color_median_rgb": text_color_median_rgb,
            "zones_rendered": zones_rendered,
            "body_grid_count": len(layouts),
            "body_col_counts": body_col_counts,
        }

        return (
            text_layers,
            texts,
            block_ids,
            words_per_line,
            block_region_types,
            textbox_null_count,
            textbox_total_count,
            line_font_info,
            line_colors,
            provenance,
        )
