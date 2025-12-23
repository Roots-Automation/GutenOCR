#!/usr/bin/env python3
"""
Script to score lines reading predictions from a CSV file.
This task evaluates structured OCR output containing both text and bounding boxes for each line.
Adds or updates evaluation metrics columns and writes back to disk.
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from shutil import copyfile

import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from scipy.optimize import linear_sum_assignment

# Add the utils directory to path
script_dir = Path(__file__).parent
utils_dir = script_dir / "utils"
sys.path.append(str(script_dir))

from utils.text import calculate_comprehensive_text_metrics
from utils.parse import parse_json_output, normalize_structured_keys
from utils.box import calculate_bbox_iou, evaluate_bbox_predictions


def extract_lines_data(predicted_answer: str) -> tuple[list, bool]:
    """Extract structured lines data from the model's predicted answer.
    
    Returns:
        tuple: (parsed_data, parse_success)
    """
    if not isinstance(predicted_answer, str):
        return [], False
    
    # Handle JSON format responses
    parsed = parse_json_output(predicted_answer)
    if parsed is not None:
        if isinstance(parsed, list):
            return parsed, True
        elif isinstance(parsed, dict):
            # Single line case
            return [parsed], True
    
    # If JSON parsing fails, return empty list
    return [], False


def extract_text_and_boxes(data: list) -> tuple:
    """Extract text and bounding boxes from structured data, sorted by spatial position."""
    pairs = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and 'text' in item and 'bbox' in item:
                bbox = item['bbox']
                if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(coord, (int, float)) for coord in bbox):
                    pairs.append((item['text'], bbox))
    elif isinstance(data, dict):
        if 'text' in data and 'bbox' in data:
            bbox = data['bbox']
            if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(coord, (int, float)) for coord in bbox):
                pairs.append((data['text'], bbox))

    if not pairs:
        return [], []

    # Sort by spatial position (y1 primary, x1 secondary)
    boxes = np.array([b for _, b in pairs])
    order = np.lexsort((boxes[:, 0], boxes[:, 1]))
    texts_sorted = [pairs[i][0] for i in order]
    boxes_sorted = boxes[order].tolist()
    return texts_sorted, boxes_sorted


def compute_lines_metrics(expected_output, predicted_output, tokenizer) -> dict:
    """Compute comprehensive metrics for lines reading task."""
    # Parse predicted output and remember if JSON was valid
    parsed_pred, parse_ok = extract_lines_data(predicted_output)
    # Additional validation: all items should be dicts (if any items exist)
    if parse_ok and parsed_pred:
        parse_ok = all(isinstance(item, dict) for item in parsed_pred)
    
    if not parse_ok:
        parsed_pred = []
    
    # Normalize both expected and predicted
    normalized_pred = normalize_structured_keys(parsed_pred)
    normalized_expected = normalize_structured_keys(expected_output if isinstance(expected_output, list) else [expected_output])
    
    # Extract text and boxes from both expected and predicted
    expected_texts, expected_boxes = extract_text_and_boxes(normalized_expected)
    predicted_texts, predicted_boxes = extract_text_and_boxes(normalized_pred)
    
    # Overall text metrics (concatenate all texts)
    combined_expected_text = ' '.join(expected_texts) if expected_texts else ""
    combined_predicted_text = ' '.join(predicted_texts) if predicted_texts else ""
    
    text_metrics = calculate_comprehensive_text_metrics(
        combined_expected_text, 
        combined_predicted_text, 
        tokenizer
    )
    
    # Bounding box metrics
    bbox_metrics = evaluate_bbox_predictions(expected_boxes, predicted_boxes)
    
    # Matched text metrics (only for spatially matched boxes)
    matched_text_metrics = {}
    if (expected_boxes and predicted_boxes and 
        len(expected_texts) == len(expected_boxes) and 
        len(predicted_texts) == len(predicted_boxes)):
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(expected_boxes), len(predicted_boxes)))
        for i, exp_box in enumerate(expected_boxes):
            for j, pred_box in enumerate(predicted_boxes):
                iou_matrix[i, j] = calculate_bbox_iou(exp_box, pred_box)
        
        # Hungarian algorithm matching above IoU threshold (0.5)
        cost_matrix = 1.0 - iou_matrix
        cost_matrix[iou_matrix < 0.5] = 1e6
        
        if cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_pairs = [(i, j, iou_matrix[i, j]) 
                             for i, j in zip(row_ind, col_ind) 
                             if iou_matrix[i, j] >= 0.5]
            
            if matched_pairs:
                matched_expected_texts = [expected_texts[i] for i, _, _ in matched_pairs]
                matched_predicted_texts = [predicted_texts[j] for _, j, _ in matched_pairs]
                
                matched_combined_expected = ' '.join(matched_expected_texts)
                matched_combined_predicted = ' '.join(matched_predicted_texts)
                
                mtm = calculate_comprehensive_text_metrics(
                    matched_combined_expected,
                    matched_combined_predicted,
                    tokenizer
                )
                
                matched_text_metrics = {f'matched_{k}': v for k, v in mtm.items()}
                matched_text_metrics['matched_box_count'] = len(matched_pairs)
                matched_text_metrics['total_expected_boxes'] = len(expected_boxes)
                matched_text_metrics['matched_box_ratio'] = (
                    len(matched_pairs) / len(expected_boxes) if expected_boxes else 0.0
                )
    
    # Combine all metrics
    result = {}
    for key, value in text_metrics.items():
        result[f'overall_{key}'] = value
    for key, value in bbox_metrics.items():
        result[f'bbox_{key}'] = value
    result.update(matched_text_metrics)
    
    # JSON-parse success flag
    result['valid_json'] = parse_ok
    
    return result


def mark_template_placeholders(df: pd.DataFrame, question_col: str = "question") -> pd.DataFrame:
    """
    Mark (do not drop) samples that contain template placeholders like {text} in the question.
    Adds a boolean column 'is_template_placeholder'.
    
    Args:
        df: DataFrame with predictions
        question_col: Column name containing the question text
        
    Returns:
        DataFrame with 'is_template_placeholder' column added
    """
    df = df.copy()
    if question_col not in df.columns:
        print(f"Warning: Question column '{question_col}' not found; skipping template placeholder marking.", flush=True)
        df['is_template_placeholder'] = False
        return df

    df[question_col] = df[question_col].astype(str)
    template_mask = df[question_col].str.contains(r'\{text\}', na=False, regex=True)
    df['is_template_placeholder'] = template_mask

    template_count = int(template_mask.sum())
    if template_count > 0:
        print(f"Detected {template_count} samples with template placeholders ({{text}}) in '{question_col}'. "
              f"These will be excluded from metric computation but kept in the CSV.", flush=True)
    else:
        print("No template placeholders found in questions.", flush=True)
    return df


def score_lines_reading_predictions(
    csv_path: str, 
    tokenizer_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    expected_col: str = "expected_answer",
    predicted_col: str = "predicted_answer",
    overwrite: bool = True,
    question_col: str = "question"
) -> None:
    """
    Score lines reading predictions in a CSV file.
    
    Args:
        csv_path: Path to CSV file with predictions
        tokenizer_name: Name/path of tokenizer to use for token-level metrics
        expected_col: Column name containing expected structured output
        predicted_col: Column name containing predicted structured output/JSON
        overwrite: Whether to overwrite existing metric columns
        question_col: Column name containing question text (for template filtering)
    """
    
    # Load CSV
    print(f"Loading predictions from: {csv_path}", flush=True)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Standard CSV parse failed ({e}). Retrying with engine='python' and on_bad_lines='skip'...", flush=True)
        df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
    
    print(f"Loaded {len(df)} samples", flush=True)
    
    # Mark template placeholders (do not drop yet)
    df = mark_template_placeholders(df, question_col=question_col)
    
    # Check required columns
    if expected_col not in df.columns:
        raise ValueError(f"Expected column '{expected_col}' not found in CSV")
    if predicted_col not in df.columns:
        raise ValueError(f"Predicted column '{predicted_col}' not found in CSV")
    
    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Warning: Failed to load tokenizer {tokenizer_name}: {e}", flush=True)
        print("Token-level metrics will be skipped", flush=True)
        tokenizer = None
    
    # Define metric columns - comprehensive set for lines reading
    metric_columns = [
        'valid_json',
        'overall_exact_match', 'overall_character_error_rate', 'overall_token_error_rate', 'overall_anls',
        'bbox_precision', 'bbox_recall', 'bbox_f1', 'bbox_mean_iou', 'bbox_total_predicted',
        'matched_exact_match', 'matched_character_error_rate', 'matched_token_error_rate', 'matched_anls',
        'matched_box_count', 'total_expected_boxes', 'matched_box_ratio'
    ]
    
    # Check if we need to overwrite or skip existing columns
    existing_metrics = [col for col in metric_columns if col in df.columns]
    if existing_metrics and not overwrite:
        print(f"Found existing metric columns: {existing_metrics}", flush=True)
        print("Use --overwrite to recalculate metrics", flush=True)
        return
    
    if existing_metrics:
        print(f"Overwriting existing metric columns: {existing_metrics}", flush=True)
    
    # Choose rows to score: exclude templates if present
    if 'is_template_placeholder' in df.columns:
        work_mask = ~df['is_template_placeholder']
    else:
        work_mask = pd.Series(True, index=df.index)

    work_df = df[work_mask].copy()
    total_rows = len(work_df)
    print(f"Computing lines reading evaluation metrics for {total_rows} samples (excluding placeholders)...", flush=True)
    
    # Compute metrics for non-template rows
    metric_results = []
    
    for idx, (orig_idx, row) in enumerate(work_df.iterrows()):
        if idx % 100 == 0 or idx == total_rows - 1:
            print(f"  Processing sample {idx+1}/{total_rows}", flush=True)
        
        try:
            # Parse expected output (stored as string in CSV)
            expected_output = eval(row[expected_col])  # Safe for our controlled data
            predicted_output = row[predicted_col]
            
            metrics = compute_lines_metrics(expected_output, predicted_output, tokenizer)
            metric_results.append((orig_idx, metrics))
            
        except Exception as e:
            print(f"Warning: Failed to compute metrics for row {orig_idx}: {e}")
            # Add default values
            default_metrics = {col: 0.0 for col in metric_columns}
            default_metrics['valid_json'] = False
            default_metrics['overall_character_error_rate'] = 1.0
            default_metrics['overall_token_error_rate'] = 1.0 if tokenizer else float('nan')
            default_metrics['matched_character_error_rate'] = 1.0
            default_metrics['matched_token_error_rate'] = 1.0 if tokenizer else float('nan')
            metric_results.append((orig_idx, default_metrics))
    
    # Initialize all metric columns (for template rows they'll be NaN/default)
    for col in metric_columns:
        if col not in df.columns:
            if col == 'valid_json':
                df[col] = False
            elif 'error_rate' in col:
                df[col] = float('nan')
            else:
                df[col] = float('nan')
    
    # Add computed metrics for non-template rows
    for orig_idx, metrics in metric_results:
        for col in metric_columns:
            if col in metrics:
                df.at[orig_idx, col] = metrics[col]
    
    # Create timestamped backup of original file
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = f"{csv_path}.{ts}.backup.csv"
    copyfile(csv_path, backup_path)
    print(f"Backup created: {backup_path}", flush=True)
    
    # Write updated CSV
    print(f"Writing updated predictions to: {csv_path}", flush=True)
    df.to_csv(csv_path, index=False)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("LINES READING SCORING SUMMARY")
    print("="*60)
    total_all = len(df)
    if 'is_template_placeholder' in df.columns:
        scored_mask = ~df['is_template_placeholder']
        scored_df = df[scored_mask]
        print(f"Total samples: {total_all}  (scored after template exclusion: {len(scored_df)})")
    else:
        scored_df = df
        print(f"Total samples: {total_all}")
    
    if len(scored_df) > 0:
        print(f"Valid JSON outputs: {scored_df['valid_json'].sum():.0f} ({100*scored_df['valid_json'].mean():.1f}%)")
    
    # Filter to valid JSON samples among scored samples for metric reporting
    valid_df = scored_df[scored_df['valid_json'] == True].copy() if len(scored_df) > 0 else pd.DataFrame()
    if len(valid_df) > 0:
        print(f"Valid samples for metrics: {len(valid_df)}")
        
        print(f"\n📝 OVERALL TEXT METRICS:")
        print(f"  Exact matches: {valid_df['overall_exact_match'].sum():.0f} ({100*valid_df['overall_exact_match'].mean():.1f}%)")
        print(f"  Mean CER: {valid_df['overall_character_error_rate'].mean():.3f}")
        print(f"  Median CER: {valid_df['overall_character_error_rate'].median():.3f}")
        if tokenizer and 'overall_token_error_rate' in valid_df.columns:
            print(f"  Mean TER: {valid_df['overall_token_error_rate'].mean():.3f}")
            print(f"  Median TER: {valid_df['overall_token_error_rate'].median():.3f}")
        print(f"  Mean ANLS: {valid_df['overall_anls'].mean():.3f}")
        print(f"  Median ANLS: {valid_df['overall_anls'].median():.3f}")
        
        print(f"\n📦 BOUNDING BOX METRICS:")
        print(f"  Precision: {100*valid_df['bbox_precision'].mean():.1f}%")
        print(f"  Recall: {100*valid_df['bbox_recall'].mean():.1f}%")
        print(f"  F1-Score: {100*valid_df['bbox_f1'].mean():.1f}%")
        print(f"  Mean IoU: {valid_df['bbox_mean_iou'].mean():.3f}")
        print(f"  Avg expected boxes: {valid_df['total_expected_boxes'].mean():.1f}")
        print(f"  Avg predicted boxes: {valid_df['bbox_total_predicted'].mean():.1f}")
        
        # Matched text metrics (only for samples with matches)
        matched_df = valid_df[valid_df['matched_box_count'] > 0].copy()
        if len(matched_df) > 0:
            print(f"\n🎯 MATCHED TEXT METRICS (spatially correct boxes):")
            print(f"  Samples with matches: {len(matched_df)} ({100*len(matched_df)/len(valid_df):.1f}%)")
            print(f"  Mean matched box ratio: {100*matched_df['matched_box_ratio'].mean():.1f}%")
            print(f"  Matched exact matches: {100*matched_df['matched_exact_match'].mean():.1f}%")
            print(f"  Matched mean CER: {matched_df['matched_character_error_rate'].mean():.3f}")
            if tokenizer and 'matched_token_error_rate' in matched_df.columns:
                print(f"  Matched mean TER: {matched_df['matched_token_error_rate'].mean():.3f}")
            print(f"  Matched mean ANLS: {matched_df['matched_anls'].mean():.3f}")
        else:
            print(f"\n🎯 MATCHED TEXT METRICS: No spatially matched boxes found")
    
    else:
        print("No valid JSON outputs found for metric calculation")
    
    print(f"\nMetric columns added/updated: {metric_columns}")
    kept_note = "kept" if 'is_template_placeholder' in df.columns else "not found"
    print(f"Template-placeholder rows were {kept_note}.")
    print(f"Backup saved to: {backup_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Score lines reading predictions from CSV")
    parser.add_argument("csv_path", help="Path to CSV file with predictions")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-VL-3B-Instruct", 
                       help="Tokenizer name/path for token-level metrics")
    parser.add_argument("--expected-col", default="expected_answer",
                       help="Column name containing expected structured output")
    parser.add_argument("--predicted-col", default="predicted_answer", 
                       help="Column name containing predicted structured output/JSON")
    parser.add_argument("--question-col", default="question",
                       help="Column name containing question text (default: question)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing metric columns")
    
    args = parser.parse_args()
    
    try:
        score_lines_reading_predictions(
            csv_path=args.csv_path,
            tokenizer_name=args.tokenizer,
            expected_col=args.expected_col,
            predicted_col=args.predicted_col,
            overwrite=args.overwrite,
            question_col=args.question_col
        )
        print("\nLines reading scoring completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
