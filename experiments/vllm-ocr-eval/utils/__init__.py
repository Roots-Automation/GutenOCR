from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from transformers import PreTrainedTokenizerBase

from utils.box import evaluate_bbox_predictions, precision_iou_sweep, calculate_bbox_iou
from utils.parse import parse_json_output, normalize_structured_keys, extract_boxes_from_normalized_json
from utils.text import calculate_comprehensive_text_metrics


def evaluate_structured_task(
    expected_output: Any,
    predicted_output: str,
    task_type: str,
    output_type: str,
    tokenizer: PreTrainedTokenizerBase | None = None
) -> dict[str, float]:
    """
    Route to appropriate structured evaluation based on task type.
    
    Args:
        expected_output: Ground truth output
        predicted_output: Raw model output text
        task_type: Type of task (detection, conditional_detection, reading, localized_reading)
        output_type: Expected output format
        tokenizer: Optional tokenizer for text metrics
        
    Returns:
        Dictionary of evaluation metrics
    """
    if task_type in ['detection', 'conditional_detection']:
        return evaluate_detection_task(expected_output, predicted_output, task_type)
    
    elif task_type == 'localized_reading':
        return evaluate_localized_reading_task(expected_output, predicted_output, tokenizer)
    
    elif task_type == 'reading' and '[' in output_type and 'box' in output_type:
        # Structured reading with bounding boxes
        return evaluate_structured_reading_task(expected_output, predicted_output, task_type, tokenizer)
    
    else:
        # Fall back to unstructured evaluation
        if isinstance(expected_output, (list, dict)):
            # Try to extract text content for comparison
            if isinstance(expected_output, list):
                expected_text = ' '.join([
                    item.get('text', str(item)) if isinstance(item, dict) else str(item)
                    for item in expected_output
                ])
            else:
                expected_text = expected_output.get('text', str(expected_output))
        else:
            expected_text = str(expected_output)
        
        return calculate_comprehensive_text_metrics(
            expected_text, 
            predicted_output, 
            tokenizer
        )


def evaluate_detection_task(
    expected_output: list[list[int]] | list[dict[str, Any]], 
    predicted_output: str,
    task_type: str = "detection"
) -> dict[str, float]:
    """
    Evaluate detection tasks (detection, conditional_detection).
    
    Args:
        expected_output: Ground truth boxes or structured data
        predicted_output: Raw model output text
        task_type: Type of detection task
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Extract expected boxes
    expected_boxes = []
    if isinstance(expected_output, list):
        for item in expected_output:
            if isinstance(item, list) and len(item) == 4:
                # Direct box format
                expected_boxes.append(item)
            elif isinstance(item, dict) and 'bbox' in item:
                expected_boxes.append(item['bbox'])
            elif isinstance(item, dict) and 'box' in item:
                expected_boxes.append(item['box'])

    # Parse and normalize predicted output
    parsed_pred = parse_json_output(predicted_output)
    if parsed_pred is None:
        parsed_pred = []
    
    # Normalize the predicted output
    normalized_pred = normalize_structured_keys(parsed_pred)
    
    # Extract bounding boxes from predicted output
    predicted_boxes = []
    if isinstance(normalized_pred, list):
        for item in normalized_pred:
            if isinstance(item, list) and len(item) == 4:
                # Direct box format
                predicted_boxes.append(item)
            elif isinstance(item, dict) and 'bbox' in item:
                predicted_boxes.append(item['bbox'])
    
    # Calculate detection metrics
    return {**precision_iou_sweep(expected_boxes, predicted_boxes), **evaluate_bbox_predictions(expected_boxes, predicted_boxes)}


def evaluate_localized_reading_task(
    expected_output: str | list[str], 
    predicted_output: str,
    tokenizer: PreTrainedTokenizerBase | None = None
) -> dict[str, float]:
    """
    Evaluate localized reading tasks where a specific region is read.
    
    Args:
        expected_output: Expected text content (string or list of strings)
        predicted_output: Raw model output text
        tokenizer: Optional tokenizer for token-level metrics
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Parse predicted output - might be JSON or plain text
    parsed_pred = parse_json_output(predicted_output)
    
    # Extract text from predicted output
    if parsed_pred is not None:
        # If it's structured, try to extract text
        normalized_pred = normalize_structured_keys(parsed_pred)
        
        if isinstance(normalized_pred, list):
            predicted_text = ' '.join([
                item.get('text', '') if isinstance(item, dict) else str(item)
                for item in normalized_pred
            ])
        elif isinstance(normalized_pred, dict) and 'text' in normalized_pred:
            predicted_text = normalized_pred['text']
        else:
            predicted_text = str(normalized_pred)
    else:
        # Treat as plain text
        predicted_text = predicted_output.strip()
    
    # Handle expected output
    if isinstance(expected_output, list):
        expected_text = ' '.join(expected_output)
    else:
        expected_text = str(expected_output)
    
    # Use unstructured text evaluation
    return calculate_comprehensive_text_metrics(
        expected_text, 
        predicted_text, 
        tokenizer
    )


def evaluate_structured_reading_task(
    expected_output: list[dict[str, Any]] | dict[str, Any], 
    predicted_output: str,
    task_type: str = "reading",
    tokenizer: PreTrainedTokenizerBase | None = None
) -> dict[str, float]:
    """
    Evaluate structured reading tasks that output text with bounding boxes.
    
    Args:
        expected_output: Ground truth structured data
        predicted_output: Raw model output text
        task_type: Type of reading task
        tokenizer: Optional tokenizer for token-level metrics
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Parse and normalize predicted output
    parsed_pred = parse_json_output(predicted_output)
    if parsed_pred is None:
        parsed_pred = []
    
    # Normalize both expected and predicted
    normalized_pred = normalize_structured_keys(parsed_pred)
    normalized_expected = normalize_structured_keys(expected_output)
    
    # Extract text and boxes
    def extract_text_and_boxes(data):
        texts = []
        boxes = []
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if 'text' in item:
                        texts.append(item['text'])
                    if 'bbox' in item:
                        boxes.append(item['bbox'])
        elif isinstance(data, dict):
            if 'text' in data:
                texts.append(data['text'])
            if 'bbox' in data:
                boxes.append(data['bbox'])
        
        return texts, boxes
    
    expected_texts, expected_boxes = extract_text_and_boxes(normalized_expected)
    predicted_texts, predicted_boxes = extract_text_and_boxes(normalized_pred)

    # Evaluate bounding boxes and get matching information
    bbox_metrics = evaluate_bbox_predictions(expected_boxes, predicted_boxes)
    
    # Evaluate text quality for spatially matched boxes only
    matched_text_metrics = {}
    if expected_boxes and predicted_boxes and len(expected_texts) == len(expected_boxes) and len(predicted_texts) == len(predicted_boxes):
        # Find box matches using IoU threshold
        iou_matrix = np.zeros((len(expected_boxes), len(predicted_boxes)))
        for i, exp_box in enumerate(expected_boxes):
            for j, pred_box in enumerate(predicted_boxes):
                iou_matrix[i, j] = calculate_bbox_iou(exp_box, pred_box)
        
        # Hungarian algorithm matching above IoU threshold (0.5)
        iou_threshold_match = 0.5
        cost_matrix = 1.0 - iou_matrix
        cost_matrix[iou_matrix < iou_threshold_match] = 1e6
        
        # Find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Filter matches that meet IoU threshold and create matched pairs
        matched_pairs = [(i, j, iou_matrix[i, j]) for i, j in zip(row_ind, col_ind) if iou_matrix[i, j] >= iou_threshold_match]
        
        # Evaluate text for matched pairs only
        if matched_pairs:
            matched_expected_texts = []
            matched_predicted_texts = []
            
            for exp_idx, pred_idx, iou_score in matched_pairs:
                matched_expected_texts.append(expected_texts[exp_idx])
                matched_predicted_texts.append(predicted_texts[pred_idx])
            
            # Calculate text metrics for matched boxes
            matched_combined_expected = ' '.join(matched_expected_texts)
            matched_combined_predicted = ' '.join(matched_predicted_texts)
            
            matched_text_metrics = calculate_comprehensive_text_metrics(
                matched_combined_expected,
                matched_combined_predicted,
                tokenizer
            )
            
            # Prefix with "matched_" to distinguish from overall text metrics
            matched_text_metrics = {f'matched_{k}': v for k, v in matched_text_metrics.items()}
            
            # Add count of matched boxes for context
            matched_text_metrics['matched_box_count'] = len(matched_pairs)
            matched_text_metrics['total_expected_boxes'] = len(expected_boxes)
            matched_text_metrics['matched_box_ratio'] = len(matched_pairs) / len(expected_boxes)
    
    return {**matched_text_metrics, **bbox_metrics}


def analyze_detection_results(
    df: pd.DataFrame,
    iou_thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
) -> dict[str, Any]:
    """
    Generalized analysis for detection-style outputs with [0, N] boxes per sample.
    Expects 'expected_answer' and 'predicted_answer' columns containing JSON-able outputs.
    Computes dataset-level P/R/F1 at multiple IoUs; if scores exist, also PR curves + AP.

    Returns:
        results dict with aggregates and processed DataFrames.
    """
    import warnings

    def extract_scores_from_normalized_json(norm):
        """Return per-pred score list (float|None)."""
        scores = []
        if isinstance(norm, list):
            for x in norm:
                if isinstance(x, dict):
                    s = x.get("score", x.get("confidence", x.get("prob", None)))
                    try:
                        scores.append(float(s) if s is not None else None)
                    except Exception:
                        scores.append(None)
                else:
                    scores.append(None)
        return scores

    def assign_matches(exp_boxes, pred_boxes, thr):
        """
        Hungarian assignment on IoU matrix with threshold gating.
        Returns: matches[(ei,pj,iou)], unmatched_exp_idxs, unmatched_pred_idxs
        """
        E, P = len(exp_boxes), len(pred_boxes)
        if E == 0 and P == 0:
            return [], [], []
        if E == 0:
            return [], [], list(range(P))
        if P == 0:
            return [], list(range(E)), []

        iou = np.zeros((E, P), dtype=float)
        for i, eb in enumerate(exp_boxes):
            for j, pb in enumerate(pred_boxes):
                iou[i, j] = calculate_bbox_iou(eb, pb)

        cost = 1.0 - iou
        cost[iou < thr] = 1e6  # disallow low-IoU pairs

        row_ind, col_ind = linear_sum_assignment(cost)
        matches = [(i, j, iou[i, j]) for i, j in zip(row_ind, col_ind) if iou[i, j] >= thr]

        matched_exp = {i for i, _, _ in matches}
        matched_pred = {j for _, j, _ in matches}
        unmatched_exp = [i for i in range(E) if i not in matched_exp]
        unmatched_pred = [j for j in range(P) if j not in matched_pred]
        return matches, unmatched_exp, unmatched_pred

    def average_precision(recalls, precisions):
        """
        VOC-style AP with precision envelope + area under curve.
        recalls, precisions: lists in score-descending order.
        """
        if not recalls:
            return 0.0
        mrec = [0.0] + list(recalls) + [1.0]
        mpre = [0.0] + list(precisions) + [0.0]
        # precision envelope
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        # integrate where recall changes
        ap = 0.0
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                ap += (mrec[i] - mrec[i - 1]) * mpre[i]
        return ap

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)

        results: dict[str, Any] = {}

        # Parse
        df = df.copy()
        df["expected_json"] = df["expected_answer"].apply(parse_json_output)
        df["predicted_json"] = df["predicted_answer"].apply(parse_json_output)

        # Valid / invalid prediction JSON
        df_valid = df[df["predicted_json"].notnull()].copy()
        df_invalid = df[df["predicted_json"].isnull()].copy()

        results["total_samples"] = len(df)
        results["valid_json_count"] = len(df_valid)
        results["invalid_json_count"] = len(df_invalid)
        results["valid_json_ratio"] = (len(df_valid) / len(df)) if len(df) else 0.0

        if len(df_valid) == 0:
            # still return the frames for consistency
            results["df_all"] = df
            results["df_valid"] = df_valid
            results["df_invalid"] = df_invalid
            results["df_valid_eq_len"] = None  # kept for backward compatibility
            results["df_valid_neq_len"] = None
            return results

        # Normalize + extract boxes for BOTH expected and predicted
        df_valid["normalized_expected_json"] = df_valid["expected_json"].apply(normalize_structured_keys)
        df_valid["normalized_predicted_json"] = df_valid["predicted_json"].apply(normalize_structured_keys)
        df_valid["expected_boxes"] = df_valid["normalized_expected_json"].apply(
            lambda x: extract_boxes_from_normalized_json(x) if x is not None else []
        )
        df_valid["predicted_boxes"] = df_valid["normalized_predicted_json"].apply(
            lambda x: extract_boxes_from_normalized_json(x) if x is not None else []
        )
        df_valid["predicted_scores"] = df_valid["normalized_predicted_json"].apply(extract_scores_from_normalized_json)

        # Counts
        df_valid["n_expected_boxes"] = df_valid["expected_boxes"].apply(len)
        df_valid["n_predicted_boxes"] = df_valid["predicted_boxes"].apply(len)

        results["images_with_no_gt"] = int((df_valid["n_expected_boxes"] == 0).sum())
        results["images_with_no_pred"] = int((df_valid["n_predicted_boxes"] == 0).sum())
        results["images_empty_both"] = int(((df_valid["n_expected_boxes"] == 0) & (df_valid["n_predicted_boxes"] == 0)).sum())

        total_gt = int(df_valid["n_expected_boxes"].sum())
        results["total_expected_boxes"] = total_gt
        results["total_predicted_boxes"] = int(df_valid["n_predicted_boxes"].sum())

        # === Dataset-level metrics across IoU thresholds ===
        # Also build material for score-based PR/AP if scores are present.
        any_scores = False
        for s_list in df_valid["predicted_scores"]:
            if any((s is not None) for s in s_list):
                any_scores = True
                break

        for thr in iou_thresholds:
            tp = fp = fn = 0
            # material for PR/AP
            scored_flags: list[tuple[float, bool]] = []  # (score, is_tp)

            for _, row in df_valid.iterrows():
                exp_boxes = row["expected_boxes"]
                pred_boxes = row["predicted_boxes"]
                scores = row["predicted_scores"]

                matches, unmatched_exp, unmatched_pred = assign_matches(exp_boxes, pred_boxes, thr)
                tp += len(matches)
                fn += len(unmatched_exp)
                fp += len(unmatched_pred)

                if any_scores:
                    matched_pred_idxs = {j for (_, j, _) in matches}
                    for j in range(len(pred_boxes)):
                        sc = scores[j] if j < len(scores) else None
                        if sc is None:
                            # Skip score-less preds for PR curve (optional: treat as 0.0)
                            continue
                        scored_flags.append((float(sc), j in matched_pred_idxs))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            results[f"precision@IoU={thr}"] = precision
            results[f"recall@IoU={thr}"] = recall
            results[f"f1@IoU={thr}"] = f1
            results[f"tp@IoU={thr}"] = tp
            results[f"fp@IoU={thr}"] = fp
            results[f"fn@IoU={thr}"] = fn

            # PR curve + AP if scores exist
            if any_scores and total_gt > 0 and len(scored_flags) > 0:
                scored_flags.sort(key=lambda x: x[0], reverse=True)
                cum_tp = 0
                cum_fp = 0
                precisions = []
                recalls = []
                for score, is_tp in scored_flags:
                    if is_tp:
                        cum_tp += 1
                    else:
                        cum_fp += 1
                    precisions.append(cum_tp / (cum_tp + cum_fp))
                    recalls.append(cum_tp / total_gt)
                ap = average_precision(recalls, precisions)
                results[f"AP@IoU={thr}"] = ap
                results[f"PR_curve@IoU={thr}"] = {"recall": recalls, "precision": precisions}

        # keep fields your downstream code expects
        results["df_all"] = df
        results["df_valid"] = df_valid
        results["df_invalid"] = df_invalid
        # Back-compat keys (not used anymore)
        results["df_valid_eq_len"] = None
        results["df_valid_neq_len"] = None

        return results
