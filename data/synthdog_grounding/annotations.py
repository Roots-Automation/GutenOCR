"""Annotation construction, filtering, and quality metrics for SynthDoG."""

import dataclasses
from collections import defaultdict

import numpy as np
from serialization import BlockAnnotation, LineAnnotation, WordAnnotation


def _clamp01(v: float) -> float:
    """Clamp a value to [0, 1]."""
    return max(0.0, min(1.0, v))


def _gray_lum(v: float) -> float:
    """WCAG relative luminance of a single grayscale value in [0, 255]."""
    c = v / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _contrast_ratio(lum_a: float, lum_b: float) -> float:
    lighter, darker = max(lum_a, lum_b), min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)


def _norm(val: float, dim: int) -> float:
    """Normalize a pixel coordinate by a dimension and clamp to [0, 1]."""
    return round(_clamp01(val / dim), 3)


def _norm_pt(x: float, y: float, w: int, h: int) -> list[float]:
    """Normalize an (x, y) point by image dimensions."""
    return [_norm(x, w), _norm(y, h)]


def _bbox_area_px(bbox: list[float], image_width: int, image_height: int) -> float:
    """Area of a normalized [x1, y1, x2, y2] bbox in pixels."""
    return (bbox[2] - bbox[0]) * image_width * (bbox[3] - bbox[1]) * image_height


def build_block_annotations(
    block_ids: list[int],
    line_bboxes: list[list[float]],
    block_region_types: dict[int, str] | None = None,
) -> list[BlockAnnotation]:
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
        region_type = (block_region_types or {}).get(bid, "body")
        blocks.append(
            BlockAnnotation(
                block_id=bid,
                bbox=[round(bx1, 3), round(by1, 3), round(bx2, 3), round(by2, 3)],
                line_ids=line_indices,
                region_type=region_type,
            )
        )
    return blocks


def capture_line_bboxes(text_layers, w: int, h: int) -> list[list[float]]:
    """Compute normalized [x1, y1, x2, y2] bboxes for each text layer.

    Derives the bounding box from the layer's quad corners so that perspective
    and skew transforms (which update layer.quad but not layer.left/top/width/height)
    are reflected in the bounding box.

    For the y-axis we average the top-edge y-values and the bottom-edge y-values
    rather than taking the global min/max of all four corners.  Under perspective
    warp, long text lines become slightly tilted: the left and right ends sit at
    different image-space y-coordinates.  Taking global min/max inflates the bbox
    height to cover the full tilt range (e.g. 42 px for a 20 px-tall line), which
    causes adjacent lines' bboxes to overlap by up to 57 % even when the text
    itself does not overlap.  Averaging the top-edge and bottom-edge y-values
    collapses that inflation, giving a tight strip around the text and eliminating
    the false overlap between consecutive lines.

    Quad corner order (synthtiger convention): [tl, tr, br, bl].
    """
    bboxes = []
    for text_layer in text_layers:
        quad = text_layer.quad
        xs = [float(pt[0]) for pt in quad]
        # Average top-edge y (corners 0,1) and bottom-edge y (corners 3,2)
        y_top = (float(quad[0][1]) + float(quad[1][1])) / 2
        y_bottom = (float(quad[3][1]) + float(quad[2][1])) / 2
        bbox = [
            _norm(min(xs), w),
            _norm(min(y_top, y_bottom), h),
            _norm(max(xs), w),
            _norm(max(y_top, y_bottom), h),
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
    min_area: float,
    w: int,
    h: int,
) -> tuple[
    list[LineAnnotation],
    list[WordAnnotation],
    int,
    int,
]:
    """Remove lines/words whose bbox area is below *min_area* pixels.

    Returns new annotation instances with reassigned IDs; input objects are not mutated.
    Returns (lines, words, degenerate_line_count, degenerate_word_count).
    """
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


def _laplacian_variance(gray: np.ndarray) -> float:
    """Laplacian variance of a 2-D grayscale array — blur detection proxy."""
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    return float(np.var(lap))


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
        p10 = _gray_lum(float(np.percentile(region, 10)))
        p90 = _gray_lum(float(np.percentile(region, 90)))
        line_contrast_ratios.append(_contrast_ratio(p10, p90))

    word_bbox_areas_px = []
    for wd in words:
        wb = wd.bbox
        wx = int(round(wb[2] * w)) - int(round(wb[0] * w))
        wy = int(round(wb[3] * h)) - int(round(wb[1] * h))
        if wx > 0 and wy > 0:
            word_bbox_areas_px.append(wx * wy)

    # Pairwise intra/cross block line overlap (normalized bbox fractions)
    max_intra = 0.0
    max_cross = 0.0
    for i in range(len(lines)):
        bi = lines[i].bbox
        area_i = (bi[2] - bi[0]) * (bi[3] - bi[1])
        for j in range(i + 1, len(lines)):
            bj = lines[j].bbox
            area_j = (bj[2] - bj[0]) * (bj[3] - bj[1])
            ix1 = max(bi[0], bj[0])
            iy1 = max(bi[1], bj[1])
            ix2 = min(bi[2], bj[2])
            iy2 = min(bi[3], bj[3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            min_area = min(area_i, area_j)
            if min_area <= 0:
                continue
            frac = inter / min_area
            if lines[i].block_id == lines[j].block_id:
                if frac > max_intra:
                    max_intra = frac
            else:
                if frac > max_cross:
                    max_cross = frac

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


def build_annotations(
    text_layers,
    texts: list[str],
    block_ids: list[int],
    words_per_line: list[list[dict]],
    image_width: int,
    image_height: int,
    emit_quads: bool,
    min_bbox_area: float,
    block_region_types: dict[int, str] | None = None,
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

    lines, words, deg_line_ct, deg_word_ct = filter_degenerate(lines, words, min_bbox_area, image_width, image_height)

    surviving_block_ids = [ln.block_id for ln in lines]
    surviving_line_bboxes = [ln.bbox for ln in lines]
    blocks = build_block_annotations(surviving_block_ids, surviving_line_bboxes, block_region_types)

    return lines, words, blocks, deg_line_ct, deg_word_ct
