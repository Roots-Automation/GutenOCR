"""Annotation construction, filtering, and quality metrics for SynthDoG."""

from collections import defaultdict

import numpy as np
from serialization import BlockAnnotation, LineAnnotation, WordAnnotation


def _clamp01(v: float) -> float:
    """Clamp a value to [0, 1]."""
    return max(0.0, min(1.0, v))


def _norm(val: float, dim: int) -> float:
    """Normalize a pixel coordinate by a dimension and clamp to [0, 1]."""
    return round(_clamp01(val / dim), 3)


def _norm_pt(x: float, y: float, w: int, h: int) -> list[float]:
    """Normalize an (x, y) point by image dimensions."""
    return [_norm(x, w), _norm(y, h)]


def _bbox_area_px(bbox: list[float], image_width: int, image_height: int) -> float:
    """Area of a normalized [x1, y1, x2, y2] bbox in pixels."""
    return (bbox[2] - bbox[0]) * image_width * (bbox[3] - bbox[1]) * image_height


def build_block_annotations(block_ids: list[int], line_bboxes: list[list[float]]) -> list[BlockAnnotation]:
    """Build block-level annotations by grouping lines that share a block_id."""
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
        blocks.append(
            BlockAnnotation(
                block_id=bid,
                bbox=[round(bx1, 3), round(by1, 3), round(bx2, 3), round(by2, 3)],
                line_ids=line_indices,
            )
        )
    return blocks


def capture_line_bboxes(text_layers, w: int, h: int) -> list[list[float]]:
    """Compute normalized [x1, y1, x2, y2] bboxes for each text layer."""
    bboxes = []
    for text_layer in text_layers:
        bbox = [
            _norm(text_layer.left, w),
            _norm(text_layer.top, h),
            _norm(text_layer.left + text_layer.width, w),
            _norm(text_layer.top + text_layer.height, h),
        ]
        bboxes.append(bbox)
    return bboxes


def capture_line_quads(text_layers, w: int, h: int) -> list[list[list[float]]]:
    """Compute normalized quad coordinates for each text layer."""
    quads = []
    for text_layer in text_layers:
        quad = text_layer.quad
        normalized = [_norm_pt(float(pt[0]), float(pt[1]), w, h) for pt in quad]
        quads.append(normalized)
    return quads


def build_word_annotations(
    text_layers, words_per_line: list[list[dict]], w: int, h: int, emit_quads: bool
) -> list[WordAnnotation]:
    """Build word-level annotations from quad interpolation.

    Word bounding boxes are derived by linear interpolation of line quad
    corners.  Under perspective transforms this is a first-order
    approximation.
    """
    words = []
    word_global_id = 0
    for line_idx, (text_layer, word_local_data) in enumerate(zip(text_layers, words_per_line)):
        line_quad = text_layer.quad
        tl, tr, br, bl = line_quad[0], line_quad[1], line_quad[2], line_quad[3]

        for word in word_local_data:
            r1, r2 = word["x1_ratio"], word["x2_ratio"]
            w_tl = tl + r1 * (tr - tl)
            w_tr = tl + r2 * (tr - tl)
            w_br = bl + r2 * (br - bl)
            w_bl = bl + r1 * (br - bl)

            xs = [float(w_tl[0]), float(w_tr[0]), float(w_br[0]), float(w_bl[0])]
            ys = [float(w_tl[1]), float(w_tr[1]), float(w_br[1]), float(w_bl[1])]
            wx1 = _norm(min(xs), w)
            wy1 = _norm(min(ys), h)
            wx2 = _norm(max(xs), w)
            wy2 = _norm(max(ys), h)

            quad = None
            if emit_quads:
                quad = [
                    _norm_pt(float(w_tl[0]), float(w_tl[1]), w, h),
                    _norm_pt(float(w_tr[0]), float(w_tr[1]), w, h),
                    _norm_pt(float(w_br[0]), float(w_br[1]), w, h),
                    _norm_pt(float(w_bl[0]), float(w_bl[1]), w, h),
                ]

            words.append(
                WordAnnotation(
                    text=word["text"],
                    bbox=[wx1, wy1, wx2, wy2],
                    line_id=line_idx,
                    word_id=word_global_id,
                    quad=quad,
                )
            )
            word_global_id += 1
    return words


def filter_degenerate(
    lines: list[LineAnnotation],
    words: list[WordAnnotation],
    blocks: list[BlockAnnotation],
    min_area: float,
    w: int,
    h: int,
) -> tuple[
    list[LineAnnotation],
    list[WordAnnotation],
    list[BlockAnnotation],
    int,
    int,
]:
    """Remove lines/words whose bbox area is below *min_area* pixels.

    Returns (lines, words, blocks, degenerate_line_count, degenerate_word_count).
    """
    degenerate_mask = [_bbox_area_px(ln.bbox, w, h) < min_area for ln in lines]
    deg_line_ct = sum(degenerate_mask)
    deg_word_ct = sum(1 for wd in words if degenerate_mask[wd.line_id]) if deg_line_ct else 0

    if not deg_line_ct:
        return lines, words, blocks, 0, 0

    survive = [i for i, degen in enumerate(degenerate_mask) if not degen]
    old_to_new = {old: new for new, old in enumerate(survive)}

    lines = [lines[i] for i in survive]
    for new_idx, ln in enumerate(lines):
        ln.line_id = new_idx

    new_words = []
    new_word_id = 0
    for wd in words:
        if wd.line_id not in old_to_new:
            continue
        wd.line_id = old_to_new[wd.line_id]
        wd.word_id = new_word_id
        new_words.append(wd)
        new_word_id += 1

    # Rebuild blocks from surviving lines
    block_ids = [ln.block_id for ln in lines]
    line_bboxes = [ln.bbox for ln in lines]
    blocks = build_block_annotations(block_ids, line_bboxes)

    return lines, new_words, blocks, deg_line_ct, deg_word_ct


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
    """Compute per-sample quality metrics from the rendered image."""
    gray = 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]
    line_contrasts = []
    line_bbox_areas_px = []
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

    word_bbox_areas_px = []
    for wd in words:
        wb = wd.bbox
        wx = int(round(wb[2] * w)) - int(round(wb[0] * w))
        wy = int(round(wb[3] * h)) - int(round(wb[1] * h))
        if wx > 0 and wy > 0:
            word_bbox_areas_px.append(wx * wy)

    return {
        "min_line_contrast": round(min(line_contrasts), 3) if line_contrasts else None,
        "mean_line_contrast": round(float(np.mean(line_contrasts)), 3) if line_contrasts else None,
        "min_line_bbox_area_px": int(min(line_bbox_areas_px)) if line_bbox_areas_px else None,
        "min_word_bbox_area_px": int(min(word_bbox_areas_px)) if word_bbox_areas_px else None,
        "degenerate_line_count": int(deg_lines),
        "degenerate_word_count": int(deg_words),
        "textbox_null_count": int(null_ct),
        "textbox_total_count": int(total_ct),
        "image_size": [int(w), int(h)],
        "word_segmentation_method": "whitespace",
    }


def build_annotations(
    text_layers,
    texts: list[str],
    block_ids: list[int],
    words_per_line: list[list[dict]],
    image_width: int,
    image_height: int,
    emit_quads: bool,
    min_bbox_area: float,
) -> tuple[list[LineAnnotation], list[WordAnnotation], list[BlockAnnotation], int, int]:
    """Orchestrate line/word/block annotation construction and degenerate filtering."""
    line_bboxes = capture_line_bboxes(text_layers, image_width, image_height)
    line_quads = capture_line_quads(text_layers, image_width, image_height) if emit_quads else []

    lines = []
    for i, text in enumerate(texts):
        lines.append(
            LineAnnotation(
                text=text,
                bbox=line_bboxes[i],
                block_id=block_ids[i],
                line_id=i,
                quad=line_quads[i] if line_quads else None,
            )
        )

    words = build_word_annotations(text_layers, words_per_line, image_width, image_height, emit_quads)
    blocks = build_block_annotations(block_ids, line_bboxes)

    lines, words, blocks, deg_line_ct, deg_word_ct = filter_degenerate(
        lines, words, blocks, min_bbox_area, image_width, image_height
    )

    return lines, words, blocks, deg_line_ct, deg_word_ct
