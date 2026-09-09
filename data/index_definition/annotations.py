"""Annotation construction for index_definition generator (Pillow-native)."""

import dataclasses
from collections import defaultdict

import numpy as np
from serialization import BlockAnnotation, LineAnnotation, WordAnnotation


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _linearize_channel(v: float) -> float:
    c = v / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _contrast_ratio(lum_a: float, lum_b: float) -> float:
    lighter, darker = max(lum_a, lum_b), min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)


def _norm(val: float, dim: int) -> float:
    return round(_clamp01(val / dim), 3)


def _norm_pt(x: float, y: float, w: int, h: int) -> list[float]:
    return [_norm(x, w), _norm(y, h)]


def _bbox_area_px(bbox: list[float], image_width: int, image_height: int) -> float:
    return (bbox[2] - bbox[0]) * image_width * (bbox[3] - bbox[1]) * image_height


def _laplacian_variance(gray: np.ndarray) -> float:
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    return float(np.var(lap))


# ── Pillow-native annotation builders ────────────────────────────────────────


def build_pillow_line_annotations(
    rendered_lines,  # list[RenderedLine]
    image_width: int,
    image_height: int,
) -> list[LineAnnotation]:
    lines = []
    for i, rl in enumerate(rendered_lines):
        bbox = [
            _norm(rl.x1_px, image_width),
            _norm(rl.y1_px, image_height),
            _norm(rl.x2_px, image_width),
            _norm(rl.y2_px, image_height),
        ]
        lines.append(
            LineAnnotation(
                text=rl.text,
                bbox=bbox,
                block_id=rl.block_id,
                line_id=i,
                quad=None,
                font_family=rl.font_family or None,
                font_size_px=rl.font_size_px or None,
                text_color_rgb=rl.text_color_rgb or None,
            )
        )
    return lines


def build_pillow_word_annotations(
    rendered_lines,  # list[RenderedLine]
    image_width: int,
    image_height: int,
) -> list[WordAnnotation]:
    words = []
    word_global_id = 0
    for line_idx, rl in enumerate(rendered_lines):
        for word_text, x1_px, x2_px in zip(rl.words, rl.word_x1s, rl.word_x2s):
            bbox = [
                _norm(x1_px, image_width),
                _norm(rl.y1_px, image_height),
                _norm(x2_px, image_width),
                _norm(rl.y2_px, image_height),
            ]
            words.append(
                WordAnnotation(
                    text=word_text,
                    bbox=bbox,
                    line_id=line_idx,
                    word_id=word_global_id,
                    quad=None,
                )
            )
            word_global_id += 1
    return words


# ── Copied verbatim from synthdog_grounding/annotations.py ───────────────────


def build_block_annotations(
    block_ids: list[int],
    line_bboxes: list[list[float]],
    line_texts: list[str],
    block_region_types: dict[int, str] | None = None,
    line_quads: list[list[list[float]]] | None = None,
) -> list[BlockAnnotation]:
    block_to_lines: dict[int, list[int]] = defaultdict(list)
    for i, bid in enumerate(block_ids):
        block_to_lines[bid].append(i)

    blocks = []
    for bid, line_indices in sorted(block_to_lines.items()):
        bboxes = [line_bboxes[i] for i in line_indices]
        bx1 = _clamp01(min(b[0] for b in bboxes))
        by1 = _clamp01(min(b[1] for b in bboxes))
        bx2 = _clamp01(max(b[2] for b in bboxes))
        by2 = _clamp01(max(b[3] for b in bboxes))
        region_type = (block_region_types or {}).get(bid, "body")
        text = " ".join(line_texts[i] for i in line_indices)

        quad = None
        if line_quads is not None:
            all_pts = [pt for i in line_indices for pt in line_quads[i]]
            qx = [p[0] for p in all_pts]
            qy = [p[1] for p in all_pts]
            qx1 = round(_clamp01(min(qx)), 3)
            qy1 = round(_clamp01(min(qy)), 3)
            qx2 = round(_clamp01(max(qx)), 3)
            qy2 = round(_clamp01(max(qy)), 3)
            quad = [[qx1, qy1], [qx2, qy1], [qx2, qy2], [qx1, qy2]]

        blocks.append(
            BlockAnnotation(
                text=text,
                block_id=bid,
                bbox=[round(bx1, 3), round(by1, 3), round(bx2, 3), round(by2, 3)],
                line_ids=line_indices,
                region_type=region_type,
                quad=quad,
            )
        )
    return blocks


def filter_degenerate(
    lines: list[LineAnnotation],
    words: list[WordAnnotation],
    min_area: float,
    w: int,
    h: int,
) -> tuple[list[LineAnnotation], list[WordAnnotation], int, int]:
    degenerate_mask = [_bbox_area_px(ln.bbox, w, h) < min_area for ln in lines]
    deg_line_ct = sum(degenerate_mask)
    deg_word_ct = sum(1 for wd in words if degenerate_mask[wd.line_id]) if deg_line_ct else 0

    if not deg_line_ct:
        return lines, words, 0, 0

    survive = [i for i, degen in enumerate(degenerate_mask) if not degen]
    old_to_new = {old: new for new, old in enumerate(survive)}

    lines = [dataclasses.replace(lines[i], line_id=new_idx) for new_idx, i in enumerate(survive)]

    new_words = []
    new_word_id = 0
    for wd in words:
        if wd.line_id not in old_to_new:
            continue
        new_words.append(dataclasses.replace(wd, line_id=old_to_new[wd.line_id], word_id=new_word_id))
        new_word_id += 1

    return lines, new_words, deg_line_ct, deg_word_ct


def compute_quality_metrics(
    image: np.ndarray,
    lines: list[LineAnnotation],
    words: list[WordAnnotation],
    w: int,
    h: int,
    deg_lines: int,
    deg_words: int,
    null_ct: int,
    total_ct: int,
) -> dict:
    gray = (0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]).astype(np.float32)
    line_contrasts = []
    line_contrast_ratios = []
    line_bbox_areas_px = []
    line_heights_px: list[float] = []
    for ln in lines:
        bbox = ln.bbox
        x1_px = int(round(bbox[0] * w))
        y1_px = int(round(bbox[1] * h))
        x2_px = int(round(bbox[2] * w))
        y2_px = int(round(bbox[3] * h))
        if x2_px <= x1_px or y2_px <= y1_px:
            continue
        region = gray[y1_px:y2_px, x1_px:x2_px]
        if region.size == 0:
            continue
        line_contrasts.append(float(np.std(region)))
        line_bbox_areas_px.append((x2_px - x1_px) * (y2_px - y1_px))
        line_heights_px.append(float(y2_px - y1_px))
        p10_raw, p90_raw = np.percentile(region, [10, 90])
        p10 = _linearize_channel(float(p10_raw))
        p90 = _linearize_channel(float(p90_raw))
        line_contrast_ratios.append(_contrast_ratio(p10, p90))

    word_bbox_areas_px = []
    for wd in words:
        wb = wd.bbox
        wx = int(round(wb[2] * w)) - int(round(wb[0] * w))
        wy = int(round(wb[3] * h)) - int(round(wb[1] * h))
        if wx > 0 and wy > 0:
            word_bbox_areas_px.append(wx * wy)

    max_intra = 0.0
    max_cross = 0.0
    if len(lines) >= 2:
        bboxes = np.array([ln.bbox for ln in lines], dtype=np.float32)
        block_ids = np.array([ln.block_id for ln in lines], dtype=np.int32)
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        i_idx, j_idx = np.triu_indices(len(lines), k=1)
        iw = np.maximum(np.minimum(x2[i_idx], x2[j_idx]) - np.maximum(x1[i_idx], x1[j_idx]), 0)
        ih = np.maximum(np.minimum(y2[i_idx], y2[j_idx]) - np.maximum(y1[i_idx], y1[j_idx]), 0)
        inter = iw * ih
        min_area = np.minimum(areas[i_idx], areas[j_idx])
        valid = (inter > 0) & (min_area > 0)
        frac = np.where(valid, inter / np.where(min_area > 0, min_area, 1.0), 0.0)
        same_block = block_ids[i_idx] == block_ids[j_idx]
        if same_block.any():
            max_intra = float(frac[same_block].max())
        if (~same_block).any():
            max_cross = float(frac[~same_block].max())

    return {
        "min_line_contrast": round(min(line_contrasts), 3) if line_contrasts else None,
        "mean_line_contrast": round(float(np.mean(line_contrasts)), 3) if line_contrasts else None,
        "min_line_contrast_ratio": round(min(line_contrast_ratios), 3) if line_contrast_ratios else None,
        "min_line_bbox_area_px": int(min(line_bbox_areas_px)) if line_bbox_areas_px else None,
        "min_word_bbox_area_px": int(min(word_bbox_areas_px)) if word_bbox_areas_px else None,
        "degenerate_line_count": int(deg_lines),
        "degenerate_word_count": int(deg_words),
        "textbox_null_count": int(null_ct),
        "textbox_total_count": int(total_ct),
        "image_size": [int(w), int(h)],
        "word_segmentation_method": "whitespace",
        "line_count": int(len(lines)),
        "word_count": int(len(words)),
        "textbox_null_frac": round(null_ct / total_ct, 3) if total_ct > 0 else 0.0,
        "min_line_height_px": round(min(line_heights_px), 1) if line_heights_px else None,
        "mean_line_height_px": round(float(np.mean(line_heights_px)), 1) if line_heights_px else None,
        "sharpness": round(_laplacian_variance(gray), 1),
        "max_intra_block_line_overlap": round(max_intra, 3),
        "max_cross_block_line_overlap": round(max_cross, 3),
    }


def build_pillow_annotations(
    rendered_lines,
    image_width: int,
    image_height: int,
    min_bbox_area: float = 16.0,
    block_region_types: dict[int, str] | None = None,
) -> tuple:
    lines = build_pillow_line_annotations(rendered_lines, image_width, image_height)
    words = build_pillow_word_annotations(rendered_lines, image_width, image_height)
    lines, words, deg_line_ct, deg_word_ct = filter_degenerate(lines, words, min_bbox_area, image_width, image_height)

    surviving_block_ids = [ln.block_id for ln in lines]
    surviving_line_bboxes = [ln.bbox for ln in lines]
    surviving_line_texts = [ln.text for ln in lines]
    blocks = build_block_annotations(
        surviving_block_ids, surviving_line_bboxes, surviving_line_texts, block_region_types
    )

    return lines, words, blocks, deg_line_ct, deg_word_ct
