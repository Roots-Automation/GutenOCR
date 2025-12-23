import numpy as np
from scipy.optimize import linear_sum_assignment


def calculate_bbox_iou(box1: list[int], box2: list[int]) -> float:
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
    try:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
    except (ValueError, TypeError):
        return 0.0
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0



def evaluate_bbox_predictions(
    expected_boxes: list[list[int]], 
    predicted_boxes: list[list[int]], 
    iou_threshold: float = 0.5
) -> dict[str, float]:
    """
    Evaluate bounding box predictions using IoU metrics.
    
    Args:
        expected_boxes: List of ground truth bounding boxes.
        predicted_boxes: List of predicted bounding boxes.
        iou_threshold: IoU threshold for considering a match.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    if not expected_boxes and not predicted_boxes:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'mean_iou': 1.0, 'matches': 0, 'total_expected': 0, 'total_predicted': 0}
    
    if not expected_boxes:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mean_iou': 0.0, 'matches': 0, 'total_expected': 0, 'total_predicted': len(predicted_boxes)}

    if not predicted_boxes:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mean_iou': 0.0, 'matches': 0, 'total_expected': len(expected_boxes), 'total_predicted': 0}

    # Calculate IoU matrix
    iou_matrix = np.zeros((len(expected_boxes), len(predicted_boxes)))
    for i, exp_box in enumerate(expected_boxes):
        for j, pred_box in enumerate(predicted_boxes):
            iou_matrix[i, j] = calculate_bbox_iou(exp_box, pred_box)
    
    # Hungarian algorithm matching
    # Build cost matrix = 1 - IoU, mask IoU < threshold as large cost
    cost_matrix = 1.0 - iou_matrix
    cost_matrix[iou_matrix < iou_threshold] = 1e6
    
    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Filter matches that meet IoU threshold
    matched_pairs = [(i, j) for i, j in zip(row_ind, col_ind) if iou_matrix[i, j] >= iou_threshold]
    
    total_iou = sum(iou_matrix[i, j] for i, j in matched_pairs)
    matches = len(matched_pairs)
    
    # Calculate metrics
    precision = matches / len(predicted_boxes) if predicted_boxes else 0.0
    recall = matches / len(expected_boxes) if expected_boxes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = total_iou / matches if matches > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': mean_iou,
        'matches': matches,
        'total_expected': len(expected_boxes),
        'total_predicted': len(predicted_boxes)
    }


def precision_iou_sweep(
    expected_boxes: list[list[int]], 
    predicted_boxes: list[list[int]], 
    iou_thresholds: list[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
) -> dict[str, float]:
    """Precision@IoU for a sweep of IoU thresholds (COCO-style grid), plus mean over thresholds."""
    
    if not expected_boxes and not predicted_boxes:
        return {f'P@{t:.2f}': 1.0 for t in iou_thresholds} | {'mP': 1.0}
    if not expected_boxes or not predicted_boxes:
        return {f'P@{t:.2f}': 0.0 for t in iou_thresholds} | {'mP': 0.0}

    iou_matrix = np.zeros((len(expected_boxes), len(predicted_boxes)), dtype=float)
    for i, g in enumerate(expected_boxes):
        for j, p in enumerate(predicted_boxes):
            iou_matrix[i, j] = calculate_bbox_iou(g, p)

    precisions = []
    for t in iou_thresholds:
        cost = 1.0 - iou_matrix
        cost[iou_matrix < t] = 1e6
        ri, cj = linear_sum_assignment(cost)
        matches = sum(1 for i, j in zip(ri, cj) if iou_matrix[i, j] >= t)
        precisions.append(matches / len(predicted_boxes))

    return {**{f'P@{t:.2f}': p for t, p in zip(iou_thresholds, precisions)},
            'mP': float(np.mean(precisions))}
