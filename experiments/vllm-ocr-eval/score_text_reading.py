#!/usr/bin/env python3
"""
Script to score text reading predictions from a CSV file.
Adds or updates evaluation metrics columns and writes back to disk.
"""

import argparse
import os
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from shutil import copyfile

import pandas as pd

script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from utils.text import calculate_character_error_rate, calculate_word_error_rate, calculate_anls
from utils.parse import parse_json_output


# ----------------------------
# Helpers for robust extraction
# ----------------------------

PREFERRED_KEYS = ("text", "answer", "value", "prediction", "content")


def _try_extract_from_obj(obj):
    """Try to extract a text-like field from dict/list objects."""
    if isinstance(obj, dict):
        for k in PREFERRED_KEYS:
            if k in obj and isinstance(obj[k], str):
                return obj[k]
    if isinstance(obj, list) and obj:
        if isinstance(obj[0], dict):
            for k in PREFERRED_KEYS:
                if k in obj[0] and isinstance(obj[0][k], str):
                    return obj[0][k]
        if isinstance(obj[0], str):
            return obj[0]
    return None


def extract_predicted_text(predicted_answer):
    """Extract the actual text from the model's predicted answer (robust)."""
    # Already-structured values (dict/list) that might have been parsed by pandas
    v = _try_extract_from_obj(predicted_answer)
    if v is not None:
        return v.strip()

    if not isinstance(predicted_answer, str):
        return ""

    # Try raw JSON
    parsed = parse_json_output(predicted_answer)
    v = _try_extract_from_obj(parsed) if parsed is not None else None
    if v is not None:
        return v.strip()

    # Any fenced block (```lang ... ```) — and re-parse JSON inside if present
    m = re.search(r'```([a-zA-Z]*)\s*\n?(.*?)\n?```', predicted_answer, re.DOTALL)
    if m:
        inner = m.group(2).strip()
        parsed_inner = parse_json_output(inner)
        v = _try_extract_from_obj(parsed_inner) if parsed_inner is not None else None
        return (v if v is not None else inner).strip()

    # Fallback to raw text
    return predicted_answer.strip()


# ----------------------------
# Text normalization utilities
# ----------------------------

def _looks_cjk(s: str) -> bool:
    """Heuristic: string contains CJK characters."""
    return bool(re.search(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]', s))


def _nfkc_casefold(s: str) -> str:
    """Unicode normalize (NFKC) + casefold."""
    return unicodedata.normalize("NFKC", s)  # .casefold()


# ----------------------------
# Metrics
# ----------------------------

def compute_text_metrics(expected_clean: str, predicted_clean: str, tokenizer=None,
                         task_hint: str = "", handle_cjk_wer: bool = True) -> dict:
    """Compute text evaluation metrics from pre-cleaned texts."""

    # Character Error Rate
    cer = calculate_character_error_rate(expected_clean, predicted_clean)

    # Word Error Rate (skip if CJK without spaces -> whitespace tokenization meaningless)
    if handle_cjk_wer and _looks_cjk(expected_clean) and _looks_cjk(predicted_clean) \
       and (" " not in expected_clean and " " not in predicted_clean):
        wer = float('nan')
    else:
        wer = calculate_word_error_rate(expected_clean, predicted_clean)

    # ANLS
    anls = calculate_anls(expected_clean, predicted_clean)

    return {
        'character_error_rate': cer,
        'word_error_rate': wer,
        'anls': anls
    }


# ----------------------------
# Template placeholder handling
# ----------------------------

def mark_template_placeholders(df: pd.DataFrame, question_col: str = "question") -> pd.DataFrame:
    """
    Mark (do not drop) samples that contain template placeholders like {text} in the question.
    Adds a boolean column 'is_template_placeholder'.
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


# ----------------------------
# Main scoring function
# ----------------------------

def score_text_reading_predictions(
    csv_path: str,
    tokenizer_name: str = None,  # kept for backward compatibility (unused)
    expected_col: str = "expected_answer",
    predicted_col: str = "predicted_answer",
    overwrite: bool = True,
    normalize_whitespace: bool = None,
    question_col: str = "question",
    unicode_normalize: bool = None,
    drop_template_rows: bool = False
) -> None:
    """
    Score text reading predictions in a CSV file.

    Args:
        csv_path: Path to CSV file with predictions
        tokenizer_name: [DEPRECATED] not used
        expected_col: Column with expected text
        predicted_col: Column with predicted text/JSON
        overwrite: Whether to overwrite existing metric columns
        normalize_whitespace: Whether to normalize whitespace (None = auto-detect)
        question_col: Column containing question text (for template marking)
        unicode_normalize: Whether to apply Unicode NFKC + casefold (None = auto-detect)
        drop_template_rows: If True, physically drop rows flagged as template placeholders
    """

    csv_filename = Path(csv_path).name
    csv_filename_l = csv_filename.lower()

    # Auto-detect whitespace normalization if unspecified
    if normalize_whitespace is None:
        is_reading_text = 'text2d' not in csv_filename_l
        normalize_whitespace = is_reading_text

    # Auto-detect Unicode normalization (useful for localized data) if unspecified
    if unicode_normalize is None:
        # looks_localized = ('localized' in csv_filename_l) or ('localised' in csv_filename_l)
        # unicode_normalize = looks_localized
        unicode_normalize = True  # default to True for robustness

    print(f"Whitespace normalization: {'ON' if normalize_whitespace else 'OFF'}", flush=True)
    print(f"Unicode NFKC+casefold: {'ON' if unicode_normalize else 'OFF'}", flush=True)

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

    # Optionally drop template rows (explicit)
    if drop_template_rows and 'is_template_placeholder' in df.columns:
        before = len(df)
        df = df[~df['is_template_placeholder']].copy()
        after = len(df)
        print(f"Dropped {before - after} template-placeholder rows (explicit flag). Remaining: {after}", flush=True)

    # Check required columns
    if expected_col not in df.columns:
        raise ValueError(f"Expected column '{expected_col}' not found in CSV")
    if predicted_col not in df.columns:
        raise ValueError(f"Predicted column '{predicted_col}' not found in CSV")

    # Metric columns we manage
    metric_columns = ['predicted_text', 'exact_match', 'character_error_rate', 'word_error_rate', 'anls']

    existing = [c for c in metric_columns if c in df.columns]
    missing = [c for c in metric_columns if c not in df.columns]

    if not overwrite and not missing:
        print(f"All metric columns already present: {existing}", flush=True)
    elif not overwrite and missing:
        print(f"Will fill ONLY missing metric columns: {missing}", flush=True)
    else:
        print(f"Overwrite mode: recomputing and updating metric columns {metric_columns}", flush=True)

    # 1) Extract predicted text (for ALL rows), then create cleaned views
    print("Extracting predicted text from model outputs...", flush=True)
    df['predicted_text_extracted'] = df[predicted_col].apply(extract_predicted_text)

    print("Pre-processing text columns...", flush=True)
    df['expected_clean'] = df[expected_col].fillna("").astype(str).str.strip()
    df['predicted_clean'] = df['predicted_text_extracted'].fillna("").astype(str).str.strip()

    # Unicode normalization
    if unicode_normalize:
        print("Applying Unicode normalization (NFKC) + casefold...", flush=True)
        df['expected_clean'] = df['expected_clean'].map(_nfkc_casefold)
        df['predicted_clean'] = df['predicted_clean'].map(_nfkc_casefold)

    # Whitespace normalization (after Unicode normalization)
    if normalize_whitespace:
        print(r"Applying vectorized whitespace normalization (re.sub(r'\s+', ' '))", flush=True)
        df['expected_clean'] = df['expected_clean'].str.replace(r'\s+', ' ', regex=True)
        df['predicted_clean'] = df['predicted_clean'].str.replace(r'\s+', ' ', regex=True)
    else:
        print("Preserving whitespace structure (no normalization)", flush=True)

    # Choose rows to score: exclude templates if present and not dropped
    if 'is_template_placeholder' in df.columns:
        work_mask = ~df['is_template_placeholder']
    else:
        work_mask = pd.Series(True, index=df.index)

    work = df[work_mask].copy()
    total_rows = len(work)
    print(f"Scoring {total_rows} samples (excluding placeholders)", flush=True)

    # 2) Compute exact matches (vectorized) on 'work'
    exact_series = (work['expected_clean'] == work['predicted_clean']).astype(int)

    # 3) Compute remaining metrics using apply
    print("Computing remaining text evaluation metrics...", flush=True)

    processed_count = 0

    def compute_metrics_row(row):
        nonlocal processed_count
        processed_count += 1
        if processed_count % 100 == 0 or processed_count == total_rows:
            print(f"  Processed {processed_count}/{total_rows} samples", flush=True)
        try:
            return compute_text_metrics(
                row['expected_clean'],
                row['predicted_clean'],
                tokenizer=None,
                task_hint=csv_path,
                handle_cjk_wer=True
            )
        except Exception as e:
            print(f"Warning: Failed to compute metrics for row {row.name}: {e}", flush=True)
            return {
                'character_error_rate': 1.0,
                'word_error_rate': float('nan'),
                'anls': 0.0
            }

    metrics_series = work.apply(compute_metrics_row, axis=1)
    metrics_df = pd.DataFrame(metrics_series.tolist(), index=work.index)

    # 4) Decide which columns we are allowed to write
    def can_write(col: str) -> bool:
        return overwrite or (col in missing)

    # 5) Write columns back to original df, respecting overwrite rules
    if can_write('predicted_text'):
        # Use the cleaned predicted text as final predicted_text
        df.loc[:, 'predicted_text'] = df['predicted_clean']

    if can_write('exact_match'):
        df.loc[work.index, 'exact_match'] = exact_series

    for col in ['character_error_rate', 'word_error_rate', 'anls']:
        if can_write(col) and col in metrics_df.columns:
            df.loc[metrics_df.index, col] = metrics_df[col]

    # Clean up temporary columns; keep 'is_template_placeholder' for visibility
    df = df.drop(columns=['predicted_text_extracted'], errors='ignore')
    df = df.drop(columns=['expected_clean', 'predicted_clean'], errors='ignore')

    # 6) Make a timestamped backup BEFORE writing
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = f"{csv_path}.{ts}.backup.csv"
    copyfile(csv_path, backup_path)
    print(f"Backup created: {backup_path}", flush=True)

    # 7) Write updated CSV
    print(f"Writing updated predictions to: {csv_path}", flush=True)
    df.to_csv(csv_path, index=False)

    # 8) Print summary statistics (only on scored rows)
    print("\n" + "=" * 50)
    print("SCORING SUMMARY")
    print("=" * 50)
    total_all = len(df)
    if 'is_template_placeholder' in df.columns and not drop_template_rows:
        scored_mask = ~df['is_template_placeholder']
        scored = df[scored_mask]
        print(f"Total samples: {total_all}  (scored after template exclusion: {len(scored)})")
    else:
        scored = df
        print(f"Total samples: {total_all}")

    # Avoid issues if some columns weren’t overwritten
    em = scored['exact_match'] if 'exact_match' in scored.columns else pd.Series(dtype=float)
    cer = scored['character_error_rate'] if 'character_error_rate' in scored.columns else pd.Series(dtype=float)
    wer = scored['word_error_rate'] if 'word_error_rate' in scored.columns else pd.Series(dtype=float)
    anls = scored['anls'] if 'anls' in scored.columns else pd.Series(dtype=float)

    if len(em):
        print(f"Exact matches: {int(pd.to_numeric(em, errors='coerce').fillna(0).sum())} "
              f"({100 * pd.to_numeric(em, errors='coerce').mean():.1f}%)")
    if len(cer):
        print(f"Mean CER: {pd.to_numeric(cer, errors='coerce').mean():.3f}")
        print(f"Median CER: {pd.to_numeric(cer, errors='coerce').median():.3f}")
    if len(wer):
        print(f"Mean WER: {pd.to_numeric(wer, errors='coerce').mean():.3f}")
        print(f"Median WER: {pd.to_numeric(wer, errors='coerce').median():.3f}")
    if len(anls):
        print(f"Mean ANLS: {pd.to_numeric(anls, errors='coerce').mean():.3f}")
        print(f"Median ANLS: {pd.to_numeric(anls, errors='coerce').median():.3f}")

    print(f"\nMetric columns added/updated (respecting overwrite): {metric_columns}")
    kept_note = "kept" if not drop_template_rows else "dropped"
    print(f"Template-placeholder rows were {kept_note}.")
    print(f"Backup saved to: {backup_path}", flush=True)


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Score text reading predictions from CSV")
    parser.add_argument("csv_path", help="Path to CSV file with predictions")

    parser.add_argument("--tokenizer", default=None,
                        help="[DEPRECATED] Tokenizer no longer used - efficient word-based splitting used instead")
    parser.add_argument("--expected-col", default="expected_answer",
                        help="Column name containing expected text")
    parser.add_argument("--predicted-col", default="predicted_answer",
                        help="Column name containing predicted text/JSON")
    parser.add_argument("--question-col", default="question",
                        help="Column name containing question text (default: question)")

    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing metric columns (by default, only missing columns are filled)")

    parser.add_argument("--normalize-whitespace", action="store_true",
                        help="Force whitespace normalization (auto-detected by default)")
    parser.add_argument("--no-normalize-whitespace", action="store_true",
                        help="Force no whitespace normalization (auto-detected by default)")

    parser.add_argument("--unicode-norm", action="store_true",
                        help="Force Unicode NFKC+casefold normalization (auto-detected by default)")
    parser.add_argument("--no-unicode-norm", action="store_true",
                        help="Force NO Unicode normalization (auto-detected by default)")

    parser.add_argument("--drop-template-rows", action="store_true",
                        help="Physically drop rows with template placeholders ({text}) instead of keeping and skipping them")

    args = parser.parse_args()

    # Resolve normalization flags
    normalize_whitespace = None
    if args.normalize_whitespace and args.no_normalize_whitespace:
        parser.error("Cannot specify both --normalize-whitespace and --no-normalize-whitespace")
    elif args.normalize_whitespace:
        normalize_whitespace = True
    elif args.no_normalize_whitespace:
        normalize_whitespace = False

    unicode_normalize = None
    if args.unicode_norm and args.no_unicode_norm:
        parser.error("Cannot specify both --unicode-norm and --no-unicode-norm")
    elif args.unicode_norm:
        unicode_normalize = True
    elif args.no_unicode_norm:
        unicode_normalize = False

    try:
        score_text_reading_predictions(
            csv_path=args.csv_path,
            tokenizer_name=args.tokenizer,
            expected_col=args.expected_col,
            predicted_col=args.predicted_col,
            overwrite=args.overwrite,
            normalize_whitespace=normalize_whitespace,
            question_col=args.question_col,
            unicode_normalize=unicode_normalize,
            drop_template_rows=args.drop_template_rows
        )
        print("\nScoring completed successfully!", flush=True)

    except Exception as e:
        print(f"Error: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
