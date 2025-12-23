#!/usr/bin/env python3
"""
Script to score detection predictions from a CSV file.
Adds or updates evaluation metrics columns and writes back to disk.

Usage:
    uv run python score_detection.py predictions.csv --overwrite
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# Compiled regex for faster box parsing
_BOX_PATTERN = re.compile(r'[\[\(](\d+),\s*(\d+),\s*(\d+),\s*(\d+)[\]\)]')

script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from utils.parse import parse_json_output


def _get_default_metrics(iou_metric_columns: List[str]) -> Dict[str, Any]:
    """Generate default metrics dictionary for failed predictions."""
    default_metrics = {
        'expected_boxes': [], 'predicted_boxes': [], 'n_expected_boxes': 0, 'n_predicted_boxes': 0,
        'valid_prediction': False, 'mean_precision': 0.0, 'mean_recall': 0.0, 'mean_f1': 0.0,
        'mean_expected_area': 0.0, 'mean_predicted_area': 0.0,
    }
    # Add IoU-specific metrics
    for col in iou_metric_columns:
        default_metrics[col] = 0.0
    return default_metrics


def _get_perfect_empty_metrics(iou_metric_columns: List[str]) -> Dict[str, Any]:
    """Generate perfect metrics for cases where both expected and predicted are empty."""
    perfect_metrics = {
        'expected_boxes': [], 'predicted_boxes': [], 'n_expected_boxes': 0, 'n_predicted_boxes': 0,
        'valid_prediction': True, 'mean_precision': 1.0, 'mean_recall': 1.0, 'mean_f1': 1.0,
        'mean_expected_area': 0.0, 'mean_predicted_area': 0.0,
    }
    # Add perfect IoU-specific metrics (when both are empty, precision=recall=f1=1.0)
    for col in iou_metric_columns:
        if 'precision' in col or 'recall' in col or 'f1' in col:
            perfect_metrics[col] = 1.0
        else:
            perfect_metrics[col] = 0.0
    return perfect_metrics


def _extract_boxes_from_parsed_json(parsed: Any) -> List[List[int]]:
    """Extract boxes from parsed JSON data."""
    boxes = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, list) and len(item) == 4:
                boxes.append([int(coord) for coord in item])
            elif isinstance(item, dict):
                for key in ['bbox', 'bounding_box', "box", "bbox_2d", "bbox2d"]:
                    if key in item and isinstance(item[key], list) and len(item[key]) == 4:
                        boxes.append([int(coord) for coord in item[key]])
                        break
    elif isinstance(parsed, dict):
        if 'boxes' in parsed and isinstance(parsed['boxes'], list):
            boxes = [[int(coord) for coord in box] for box in parsed['boxes'] if len(box) == 4]
        elif 'bbox' in parsed and isinstance(parsed['bbox'], list) and len(parsed['bbox']) == 4:
            boxes = [[int(coord) for coord in parsed['bbox']]]
        elif 'box' in parsed and isinstance(parsed['box'], list) and len(parsed['box']) == 4:
            boxes = [[int(coord) for coord in parsed['box']]]
        elif 'bbox_2d' in parsed and isinstance(parsed['bbox_2d'], list) and len(parsed['bbox_2d']) == 4:
            boxes = [[int(coord) for coord in parsed['bbox_2d']]]
        elif 'bbox2d' in parsed and isinstance(parsed['bbox2d'], list) and len(parsed['bbox2d']) == 4:
            boxes = [[int(coord) for coord in parsed['bbox2d']]]
    return boxes


def _extract_boxes_from_text(text: str) -> List[List[int]]:
    """Extract coordinate patterns from raw text with optimized regex."""
    # Use pre-compiled regex for better performance
    return [[int(coord) for coord in match] for match in _BOX_PATTERN.findall(text)]


def parse_bounding_boxes(text: str) -> List[List[int]]:
    """Parse bounding boxes from text (handles both expected and predicted formats)."""
    if not isinstance(text, str):
        return []
    
    parsed = parse_json_output(text)
    if parsed is not None:
        boxes = _extract_boxes_from_parsed_json(parsed)
        if boxes:
            return boxes
    
    # Fallback to regex extraction
    return _extract_boxes_from_text(text)


def calculate_box_area(box: List[int]) -> int:
    """Calculate area of a bounding box [x1, y1, x2, y2]."""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def assign_matches(expected_boxes: List[List[int]], 
                  predicted_boxes: List[List[int]], 
                  iou_threshold: float = 0.5) -> Tuple[List[Tuple[int, int, float]], List[int], List[int], np.ndarray]:
    """
    Assign matches between expected and predicted boxes using Hungarian algorithm.
    
    Returns:
        matches: List of (expected_idx, predicted_idx, iou) tuples
        unmatched_expected: List of unmatched expected box indices
        unmatched_predicted: List of unmatched predicted box indices
        iou_matrix: Full IoU matrix for analysis
    """
    E, P = len(expected_boxes), len(predicted_boxes)
    
    if E == 0 and P == 0:
        return [], [], [], np.array([])
    if E == 0:
        return [], [], list(range(P)), np.array([])
    if P == 0:
        return [], list(range(E)), [], np.array([])
    
    # Compute IoU matrix (vectorized version)
    iou_matrix = _compute_iou_matrix_vectorized(expected_boxes, predicted_boxes)
    
    # Hungarian assignment with IoU threshold gating
    cost_matrix = 1.0 - iou_matrix
    cost_matrix[iou_matrix < iou_threshold] = 1e6  # High cost for low IoU
    
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Extract valid matches (above IoU threshold)
    matches = []
    matched_expected = set()
    matched_predicted = set()
    
    for i, j in zip(row_indices, col_indices):
        if iou_matrix[i, j] >= iou_threshold:
            matches.append((i, j, iou_matrix[i, j]))
            matched_expected.add(i)
            matched_predicted.add(j)
    
    # Find unmatched boxes
    unmatched_expected = [i for i in range(E) if i not in matched_expected]
    unmatched_predicted = [j for j in range(P) if j not in matched_predicted]
    
    return matches, unmatched_expected, unmatched_predicted, iou_matrix


def _compute_iou_matrix_vectorized(expected_boxes: List[List[int]], predicted_boxes: List[List[int]]) -> np.ndarray:
    """Vectorized IoU matrix computation for better performance."""
    E, P = len(expected_boxes), len(predicted_boxes)
    if E == 0 or P == 0:
        return np.array([])
    
    # Convert to numpy arrays for vectorization
    exp_boxes = np.array(expected_boxes, dtype=np.float32)  # Shape: (E, 4)
    pred_boxes = np.array(predicted_boxes, dtype=np.float32)  # Shape: (P, 4)
    
    # Broadcast for pairwise operations
    exp_x1, exp_y1, exp_x2, exp_y2 = exp_boxes[:, None, 0], exp_boxes[:, None, 1], exp_boxes[:, None, 2], exp_boxes[:, None, 3]
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes[None, :, 0], pred_boxes[None, :, 1], pred_boxes[None, :, 2], pred_boxes[None, :, 3]
    
    # Intersection coordinates
    inter_x1 = np.maximum(exp_x1, pred_x1)
    inter_y1 = np.maximum(exp_y1, pred_y1)
    inter_x2 = np.minimum(exp_x2, pred_x2)
    inter_y2 = np.minimum(exp_y2, pred_y2)
    
    # Intersection area
    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    
    # Box areas
    exp_areas = (exp_x2 - exp_x1) * (exp_y2 - exp_y1)
    pred_areas = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    
    # Union area
    union_area = exp_areas + pred_areas - inter_area
    
    # IoU with division by zero protection
    iou_matrix = np.divide(inter_area, union_area, out=np.zeros_like(inter_area), where=union_area > 0)
    
    return iou_matrix.squeeze() if iou_matrix.ndim > 2 else iou_matrix


def compute_detection_metrics(expected_boxes: List[List[int]], 
                            predicted_boxes: List[List[int]], 
                            iou_thresholds: List[float] = None) -> Dict[str, Any]:
    """Compute detection evaluation metrics with optimized IoU calculation."""
    if iou_thresholds is None:
        iou_thresholds = [i * 0.05 for i in range(1, 20)]  # 0.05 to 0.95
    
    E, P = len(expected_boxes), len(predicted_boxes)
    
    metrics = {
        'n_expected_boxes': E,
        'n_predicted_boxes': P,
        'expected_boxes': expected_boxes,
        'predicted_boxes': predicted_boxes,
        'valid_prediction': True
    }
    
    # Calculate box areas (vectorized)
    if expected_boxes:
        exp_areas = np.array([calculate_box_area(box) for box in expected_boxes])
        metrics.update({
            'expected_areas': exp_areas.tolist(),
            'mean_expected_area': float(exp_areas.mean()),
        })
    else:
        metrics.update({'expected_areas': [], 'mean_expected_area': 0.0})
    
    if predicted_boxes:
        pred_areas = np.array([calculate_box_area(box) for box in predicted_boxes])
        metrics.update({
            'predicted_areas': pred_areas.tolist(),
            'mean_predicted_area': float(pred_areas.mean()),
        })
    else:
        metrics.update({'predicted_areas': [], 'mean_predicted_area': 0.0})
    
    # Early exit for empty cases
    if E == 0 or P == 0:
        # All thresholds will have same results for empty cases
        for iou_thresh in iou_thresholds:
            thresh_str = f"{iou_thresh:.2f}".replace('.', '_')
            if E == 0 and P == 0:
                # Both empty - perfect match
                metrics.update({f'tp_iou_{thresh_str}': 0, f'fp_iou_{thresh_str}': 0, f'fn_iou_{thresh_str}': 0,
                               f'precision_iou_{thresh_str}': 1.0, f'recall_iou_{thresh_str}': 1.0, f'f1_iou_{thresh_str}': 1.0,
                               f'mean_matched_iou_{thresh_str}': 0.0})
            elif E == 0:
                # Expected empty, predicted non-empty - all false positives
                metrics.update({f'tp_iou_{thresh_str}': 0, f'fp_iou_{thresh_str}': P, f'fn_iou_{thresh_str}': 0,
                               f'precision_iou_{thresh_str}': 0.0, f'recall_iou_{thresh_str}': 1.0, f'f1_iou_{thresh_str}': 0.0,
                               f'mean_matched_iou_{thresh_str}': 0.0})
            else:  # P == 0
                # Expected non-empty, predicted empty - all false negatives
                metrics.update({f'tp_iou_{thresh_str}': 0, f'fp_iou_{thresh_str}': 0, f'fn_iou_{thresh_str}': E,
                               f'precision_iou_{thresh_str}': 1.0, f'recall_iou_{thresh_str}': 0.0, f'f1_iou_{thresh_str}': 0.0,
                               f'mean_matched_iou_{thresh_str}': 0.0})
        
        if E == 0 and P == 0:
            # Both empty - perfect scores
            metrics.update({
                'mean_precision': 1.0, 'mean_recall': 1.0, 'mean_f1': 1.0, 'matched_ious': []
            })
        elif E == 0:
            # Expected empty, predicted non-empty
            metrics.update({
                'mean_precision': 0.0, 'mean_recall': 1.0, 'mean_f1': 0.0, 'matched_ious': []
            })
        else:  # P == 0
            # Expected non-empty, predicted empty
            metrics.update({
                'mean_precision': 1.0, 'mean_recall': 0.0, 'mean_f1': 0.0, 'matched_ious': []
            })
        return metrics
    
    # Compute IoU matrix once (expensive operation)
    iou_matrix = _compute_iou_matrix_vectorized(expected_boxes, predicted_boxes)
    
    # Process all thresholds with pre-computed IoU matrix
    precisions, recalls, f1s = [], [], []
    all_matched_ious = []
    
    for i, iou_thresh in enumerate(iou_thresholds):
        # Create cost matrix for Hungarian algorithm
        cost_matrix = 1.0 - iou_matrix
        cost_matrix[iou_matrix < iou_thresh] = 1e6
        
        # Hungarian assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Count matches above threshold
        valid_matches = iou_matrix[row_indices, col_indices] >= iou_thresh
        tp = valid_matches.sum()
        fp = P - tp
        fn = E - tp
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Mean IoU of matched pairs
        matched_ious = iou_matrix[row_indices[valid_matches], col_indices[valid_matches]]
        mean_matched_iou = float(matched_ious.mean()) if len(matched_ious) > 0 else 0.0
        
        # Store for first threshold (distribution analysis)
        if i == 0:
            all_matched_ious = matched_ious.tolist()
        
        # Store aggregate metrics
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
        # Store per-threshold metrics
        thresh_str = f"{iou_thresh:.2f}".replace('.', '_')
        metrics.update({
            f'tp_iou_{thresh_str}': int(tp),
            f'fp_iou_{thresh_str}': int(fp),
            f'fn_iou_{thresh_str}': int(fn),
            f'precision_iou_{thresh_str}': float(precision),
            f'recall_iou_{thresh_str}': float(recall),
            f'f1_iou_{thresh_str}': float(f1),
            f'mean_matched_iou_{thresh_str}': float(mean_matched_iou),
        })
    
    # Aggregate metrics
    metrics.update({
        'mean_precision': float(np.mean(precisions)),
        'mean_recall': float(np.mean(recalls)), 
        'mean_f1': float(np.mean(f1s)),
        'matched_ious': all_matched_ious,
    })
    
    return metrics


def filter_template_placeholders(df: pd.DataFrame, question_col: str = "question") -> pd.DataFrame:
    """
    Filter out samples that contain template placeholders like {text} in the question.
    
    Args:
        df: DataFrame with predictions
        question_col: Column name containing the question text
        
    Returns:
        Filtered DataFrame without template placeholder samples
    """
    if question_col not in df.columns:
        print(f"Warning: Question column '{question_col}' not found, skipping template filtering")
        return df
    
    # Convert to string and handle NaN values
    df = df.copy()
    df[question_col] = df[question_col].astype(str)
    
    # Check for {text} placeholder in questions
    original_count = len(df)
    template_mask = df[question_col].str.contains(r'\{text\}', na=False, regex=True)
    template_count = template_mask.sum()
    
    if template_count > 0:
        print(f"Filtering out {template_count} samples with template placeholders ({{text}}) in questions")
        df_filtered = df[~template_mask].copy()
        print(f"Samples after filtering: {len(df_filtered)} (removed {original_count - len(df_filtered)})")
        return df_filtered
    else:
        print("No template placeholders found in questions")
        return df


def score_detection_predictions(
    csv_path: str,
    expected_col: str = "expected_answer", 
    predicted_col: str = "predicted_answer",
    iou_thresholds: List[float] = None,
    overwrite: bool = True,
    question_col: str = "question"
) -> None:
    """
    Score detection predictions in a CSV file.
    
    Args:
        csv_path: Path to CSV file with predictions
        expected_col: Column name containing expected boxes (JSON)
        predicted_col: Column name containing predicted boxes (JSON)
        iou_thresholds: List of IoU thresholds to evaluate at
        overwrite: Whether to overwrite existing metric columns
        question_col: Column name containing question text (for template filtering)
    """
    
    if iou_thresholds is None:
        # Default: 0.05 to 0.95 in steps of 0.05 (COCO-style)
        iou_thresholds = [i * 0.05 for i in range(1, 20)]
    
    # Load CSV
    print(f"Loading predictions from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    
    # Filter out template placeholders
    df = filter_template_placeholders(df, question_col)
    
    # Check required columns
    if expected_col not in df.columns:
        raise ValueError(f"Expected column '{expected_col}' not found in CSV")
    if predicted_col not in df.columns:
        raise ValueError(f"Predicted column '{predicted_col}' not found in CSV")
    
    # Define metric columns we'll add (focusing on most important ones to avoid too many columns)
    base_metric_columns = [
        'predicted_boxes', 'expected_boxes', 'n_expected_boxes', 'n_predicted_boxes', 
        'valid_prediction', 'mean_precision', 'mean_recall', 'mean_f1',
        'mean_expected_area', 'mean_predicted_area'
    ]
    
    # Add key IoU thresholds (not all 19 to keep manageable)
    key_ious = [0.30, 0.50, 0.70, 0.90]  # Representative subset
    iou_metric_columns = [
        f'{metric}_iou_{iou_thresh:.2f}'.replace('.', '_')
        for iou_thresh in key_ious if iou_thresh in iou_thresholds
        for metric in ['precision', 'recall', 'f1']
    ]
    
    metric_columns = base_metric_columns + iou_metric_columns
    
    # Check if we need to overwrite existing columns
    existing_metrics = [col for col in metric_columns if col in df.columns]
    if existing_metrics and not overwrite:
        print(f"Found existing metric columns: {existing_metrics}")
        print("Use --overwrite to recalculate metrics")
        return
    
    if existing_metrics:
        print(f"Overwriting existing metric columns: {existing_metrics}")
    
    # Extract boxes and compute metrics with batch processing
    print(f"Computing detection metrics at {len(iou_thresholds)} IoU thresholds...")
    
    # Pre-parse all boxes for faster processing
    print("  Parsing bounding boxes...")
    expected_boxes_list = []
    predicted_boxes_list = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"    Parsing {idx+1}/{len(df)}")
        
        try:
            expected_boxes = parse_bounding_boxes(row[expected_col])
            predicted_boxes = parse_bounding_boxes(row[predicted_col])
            expected_boxes_list.append(expected_boxes)
            predicted_boxes_list.append(predicted_boxes)
        except Exception as e:
            print(f"Warning: Failed to parse boxes for row {idx}: {e}")
            expected_boxes_list.append([])
            predicted_boxes_list.append([])
    
    # Compute metrics
    print("  Computing metrics...")
    metric_results = []
    
    for idx, (expected_boxes, predicted_boxes) in enumerate(zip(expected_boxes_list, predicted_boxes_list)):
        if idx % 100 == 0:
            print(f"    Computing {idx+1}/{len(df)}")
        
        try:
            if not expected_boxes and not predicted_boxes:
                # Both empty - perfect prediction (precision=1, recall=1, f1=1)
                metrics = _get_perfect_empty_metrics(iou_metric_columns)
            else:
                metrics = compute_detection_metrics(expected_boxes, predicted_boxes, iou_thresholds)
            
            metric_results.append(metrics)
            
        except Exception as e:
            print(f"Warning: Failed to compute metrics for row {idx}: {e}")
            metric_results.append(_get_default_metrics(iou_metric_columns))
    
    # Add/update metric columns  
    metrics_df = pd.DataFrame(metric_results)
    for col in metric_columns:
        if col in metrics_df.columns:
            df[col] = metrics_df[col]
    
    # Create backup of original file
    backup_path = csv_path + '.backup'
    if not os.path.exists(backup_path):
        print(f"Creating backup: {backup_path}")
        df_original = pd.read_csv(csv_path)  # Re-read original
        df_original.to_csv(backup_path, index=False)
    
    # Write updated CSV
    print(f"Writing updated predictions to: {csv_path}")
    df.to_csv(csv_path, index=False)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DETECTION SCORING SUMMARY")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Valid predictions: {df['valid_prediction'].sum():.0f} ({100*df['valid_prediction'].mean():.1f}%)")
    print(f"Average expected boxes per image: {df['n_expected_boxes'].mean():.2f}")
    print(f"Average predicted boxes per image: {df['n_predicted_boxes'].mean():.2f}")
    
    print(f"\nOverall Performance:")
    print(f"  Mean Precision: {df['mean_precision'].mean():.3f}")
    print(f"  Mean Recall: {df['mean_recall'].mean():.3f}")
    print(f"  Mean F1: {df['mean_f1'].mean():.3f}")
    
    print(f"\nPerformance at key IoU thresholds:")
    for iou_thresh in key_ious:
        if iou_thresh in iou_thresholds:
            thresh_str = f"{iou_thresh:.2f}".replace('.', '_')
            cols = {metric: f'{metric}_iou_{thresh_str}' for metric in ['precision', 'recall', 'f1']}
            
            if cols['precision'] in df.columns:
                print(f"  IoU ≥ {iou_thresh:.2f}:")
                for metric, col in cols.items():
                    print(f"    {metric.capitalize()}: {df[col].mean():.3f}")
    
    print(f"\nEvaluated at {len(iou_thresholds)} IoU thresholds: {min(iou_thresholds):.2f} to {max(iou_thresholds):.2f}")
    print(f"Metric columns added/updated: {len(metric_columns)}")
    print(f"Backup saved to: {backup_path}")


def main():
    parser = argparse.ArgumentParser(description="Score detection predictions from CSV")
    parser.add_argument("csv_path", help="Path to CSV file with predictions")
    parser.add_argument("--expected-col", default="expected_answer",
                       help="Column name containing expected boxes (JSON)")
    parser.add_argument("--predicted-col", default="predicted_answer", 
                       help="Column name containing predicted boxes (JSON)")
    parser.add_argument("--question-col", default="question",
                       help="Column name containing question text (default: question)")
    parser.add_argument("--iou-step", type=float, default=0.05,
                       help="Step size for IoU thresholds (default: 0.05)")
    parser.add_argument("--iou-min", type=float, default=0.05,
                       help="Minimum IoU threshold (default: 0.05)")
    parser.add_argument("--iou-max", type=float, default=0.95,
                       help="Maximum IoU threshold (default: 0.95)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing metric columns")
    
    args = parser.parse_args()
    
    # Generate IoU thresholds
    iou_thresholds = []
    current_iou = args.iou_min
    while current_iou <= args.iou_max + 1e-6:  # Small epsilon for floating point comparison
        iou_thresholds.append(round(current_iou, 2))
        current_iou += args.iou_step
    
    print(f"Using {len(iou_thresholds)} IoU thresholds: {iou_thresholds[0]} to {iou_thresholds[-1]} (step: {args.iou_step})")
    
    try:
        score_detection_predictions(
            csv_path=args.csv_path,
            expected_col=args.expected_col,
            predicted_col=args.predicted_col,
            iou_thresholds=iou_thresholds,
            overwrite=args.overwrite,
            question_col=args.question_col
        )
        print("\nDetection scoring completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
