import logging
from typing import Sequence

import Levenshtein
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def calculate_anls(expected: str, predicted: str, threshold: float = 0.5) -> float:
    """
    Calculate Average Normalized Levenshtein Similarity (ANLS).
    
    ANLS is commonly used in OCR evaluation. It's defined as:
    ANLS = 1 - NL if NL < threshold, else 0
    where NL = Levenshtein distance / max(len(expected), len(predicted))
    
    Args:
        expected: Ground truth text
        predicted: Predicted text
        threshold: Threshold below which similarity is considered 0
        
    Returns:
        ANLS score between 0 and 1
    """
    if not expected and not predicted:
        return 1.0
    
    if not expected or not predicted:
        return 0.0
    
    edit_distance = Levenshtein.distance(expected, predicted)
    max_len = max(len(expected), len(predicted))
    
    if max_len == 0:
        return 1.0
    
    normalized_distance = edit_distance / max_len
    
    if normalized_distance < threshold:
        return 1.0 - normalized_distance
    else:
        return 0.0


def calculate_character_error_rate(expected: str, predicted: str) -> float:
    """
    CER = levenshtein(expected, predicted) / max(len(expected), len(predicted))

    Returns a value in [0, 1] when len(expected) > 0.
    If expected is empty: 0.0 if both empty else 1.0 (conventional choice).
    Uses max length for normalization to ensure bounded metric [0,1].
    """
    if not expected and not predicted:
        return 0.0
    if not expected:
        # undefined mathematically; use 1.0 so metric stays bounded
        return 1.0

    edit_distance = Levenshtein.distance(expected, predicted)
    return edit_distance / max(len(expected), len(predicted))


def _sequence_edit_distance(a: Sequence, b: Sequence) -> int:
    """
    Levenshtein edit distance over sequences (tokens), O(len(a)*len(b)).
    """
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    # DP over two rows for memory efficiency
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost  # substitution
            )
        prev = curr
    return prev[-1]

def calculate_word_error_rate(expected: str, predicted: str) -> float:
    """
    Fast WER using python-Levenshtein on token-encoded strings.
    Each unique word -> one Unicode code point; Levenshtein over those symbols
    equals word-level edit distance. Normalized by max(#words).
    """
    e_words = expected.split() if expected else []
    p_words = predicted.split() if predicted else []

    if not e_words and not p_words:
        return 0.0
    if not e_words:
        return 1.0
    if e_words == p_words:
        return 0.0

    # Map each unique token to a unique code point.
    vocab = {}
    next_cp = 0x10000  # start in Supplementary Plane for tons of room
    def encode(words):
        nonlocal next_cp
        out = []
        for w in words:
            cp = vocab.get(w)
            if cp is None:
                cp = chr(next_cp)
                vocab[w] = cp
                next_cp += 1
            out.append(cp)
        return ''.join(out)

    se = encode(e_words)
    sp = encode(p_words)

    dist = Levenshtein.distance(se, sp)
    return dist / max(len(e_words), len(p_words))

def calculate_comprehensive_text_metrics(
    expected: str, 
    predicted: str, 
    tokenizer: PreTrainedTokenizerBase | None = None,
    anls_threshold: float = 0.5
) -> dict[str, float]:
    """
    Calculate comprehensive text evaluation metrics for unstructured text.
    
    Args:
        expected: Ground truth text
        predicted: Predicted text
        tokenizer: Optional tokenizer for token-level metrics
        anls_threshold: Threshold for ANLS calculation
        
    Returns:
        Dictionary containing:
        - exact_match: Binary exact match
        - anls: Average Normalized Levenshtein Similarity 
        - character_error_rate: Character Error Rate
        - token_error_rate: Token Error Rate (if tokenizer provided)
    """
    import re
    
    # Clean texts for comparison - normalize whitespace
    exp_clean = expected.strip()
    pred_clean = predicted.strip()
    
    # Normalize whitespace - collapse multiple spaces/newlines to single space
    exp_clean = re.sub(r'\s+', ' ', exp_clean)
    pred_clean = re.sub(r'\s+', ' ', pred_clean)

    metrics = {}
    
    # Exact match
    metrics['exact_match'] = float(exp_clean == pred_clean)
    
    # ANLS (Average Normalized Levenshtein Similarity)
    metrics['anls'] = calculate_anls(exp_clean, pred_clean, threshold=anls_threshold)
    
    # Character Error Rate
    metrics['character_error_rate'] = calculate_character_error_rate(exp_clean, pred_clean)
    
    # Word Error Rate (efficient whitespace-based splitting)
    try:
        metrics['word_error_rate'] = calculate_word_error_rate(exp_clean, pred_clean)
        # Keep token_error_rate for backward compatibility
        metrics['token_error_rate'] = metrics['word_error_rate']
    except Exception as e:
        logger.warning(f"Failed to calculate word error rate: {e}")
        metrics['word_error_rate'] = float('nan')
        metrics['token_error_rate'] = float('nan')
    
    return metrics
