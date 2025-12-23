import re, json, argparse, html as html_module, unicodedata
from collections import Counter
import Levenshtein
from bs4 import BeautifulSoup
import markdown2

try:
    # Prefer RapidFuzz (fast, pure Python wheels). If not installed, we'll fall back gracefully.
    from rapidfuzz import fuzz as rf_fuzz, process as rf_process
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False

WS_COLLAPSE = re.compile(r"\s+")
HTML_TAG_HINT = re.compile(r"</?(p|div|span|br|h[1-6]|ul|ol|li|table|tr|td|em|strong|a|img|code|pre)\b", re.I)
HTML_ENTITY_HINT = re.compile(r"&[a-zA-Z]+;|&#\d+;")
MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s", re.M)
MD_FENCE = re.compile(r"```")
MD_LINK_IMAGE = re.compile(r"!\[[^\]]*\]\([^\)]+\)|\[[^\]]+\]\([^\)]+\)")
MD_LIST = re.compile(r"^\s{0,3}(\d+\.\s+|[-*+]\s+)", re.M)
MD_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s+", re.M)

def is_html_like(text: str) -> bool:
    if not text:
        return False
    if "<" not in text or ">" not in text:
        return bool(HTML_ENTITY_HINT.search(text))
    return bool(HTML_TAG_HINT.search(text)) or bool(HTML_ENTITY_HINT.search(text))

def is_markdown_like(text: str) -> bool:
    if not text:
        return False
    return any([
        MD_HEADING.search(text),
        len(MD_FENCE.findall(text)) >= 2,
        MD_LINK_IMAGE.search(text),
        MD_LIST.search(text),
        MD_BLOCKQUOTE.search(text),
        text.count("`") >= 2,
    ])

def strip_html(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script","style"]):
        tag.decompose()
    extracted = soup.get_text(separator=" ")
    extracted = html_module.unescape(extracted).replace("\xa0"," ")
    return WS_COLLAPSE.sub(" ", extracted).strip()

def strip_markdown(text: str) -> str:
    html = markdown2.markdown(text)
    return strip_html(html)

def normalize_text(x: str) -> str:
    if is_html_like(x):
        x = strip_html(x)
    elif is_markdown_like(x):
        x = strip_markdown(x)
    # Unicode NFC + whitespace collapse; lowercase optional (enable if your GT is case-insensitive)
    x = unicodedata.normalize("NFC", x)
    x = WS_COLLAPSE.sub(" ", x.strip())
    return x

def extract_text_from_json_like(s: str) -> str:
    try:
        obj = json.loads(s)
    except Exception:
        return s
    bag = []
    def walk(v):
        if isinstance(v, dict):
            for k, val in v.items():
                # Only harvest likely text-bearing keys
                if isinstance(val, (str, list, dict)) and k.lower() in {"text","value","content","answer","output"}:
                    walk(val)
        elif isinstance(v, list):
            for it in v:
                walk(it)
        elif isinstance(v, str):
            bag.append(v)
    walk(obj)
    return " ".join(bag) if bag else s

# Tokenizer: try Unicode-aware; fallback to ASCII-ish
try:
    import regex as uni_re
    TOKEN_RE = uni_re.compile(r"(?:\p{L}+(?:['’-]\p{L}+)*)|(?:\p{N}+(?:[.,]\p{N}+)*)", uni_re.U)
    def tokenize(t): return TOKEN_RE.findall(t)
except Exception:
    TOKEN_RE = re.compile(r"(?:[A-Za-z]+(?:['-][A-Za-z]+)*)|(?:\d+(?:[.,]\d+)*)")
    def tokenize(t): return TOKEN_RE.findall(t)

def multiset_overlap_counts(gt_tokens, pred_tokens):
    G, P = Counter(gt_tokens), Counter(pred_tokens)
    m = sum(min(G[w], P[w]) for w in set(G)|set(P))
    gsize, psize = sum(G.values()), sum(P.values())
    u = sum(max(G[w], P[w]) for w in set(G)|set(P))  # for multiset Jaccard
    return m, gsize, psize, u

def prf_from_overlap(m, gsize, psize):
    recall = m / gsize if gsize else 0.0
    precision = m / psize if psize else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    return precision, recall, f1

def soft_match_prf(gt_tokens, pred_tokens, tau=0.85):
    if not HAVE_RAPIDFUZZ:
        # fall back to exact
        m, gsize, psize, _ = multiset_overlap_counts(gt_tokens, pred_tokens)
        return prf_from_overlap(m, gsize, psize)
    # Greedy one-to-one matching by best similarity above tau
    used_pred = [False]*len(pred_tokens)
    M = 0.0
    for g in gt_tokens:
        best_sim, best_j = 0.0, -1
        for j, p in enumerate(pred_tokens):
            if used_pred[j]: 
                continue
            s = rf_fuzz.token_sort_ratio(g, p) / 100.0  # simple & robust; or use ratio
            if s > best_sim:
                best_sim, best_j = s, j
        if best_sim >= tau and best_j >= 0:
            used_pred[best_j] = True
            M += best_sim
    gsize, psize = len(gt_tokens), len(pred_tokens)
    recall = M / gsize if gsize else 0.0
    precision = M / psize if psize else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    return precision, recall, f1

def compute_metrics(pred_raw, gt_raw, ned_norm="max", tau=0.85):
    pred = extract_text_from_json_like(pred_raw) if pred_raw.lstrip().startswith(("{","[")) else pred_raw
    pred = normalize_text(pred)
    gt = normalize_text(gt_raw)

    # Character NED
    ed = Levenshtein.distance(pred, gt)
    denom = max(len(pred), len(gt)) if ned_norm=="max" else (len(gt) or 1)
    ned = ed / denom

    # Tokens
    gt_tok = tokenize(gt)
    pr_tok = tokenize(pred)

    m, gsize, psize, u = multiset_overlap_counts(gt_tok, pr_tok)
    p_exact, r_exact, f1_exact = prf_from_overlap(m, gsize, psize)
    jaccard_ms = m / u if u else 0.0

    p_soft, r_soft, f1_soft = soft_match_prf(gt_tok, pr_tok, tau=tau)

    return {
        "ned_char": ned,
        "tok_precision": p_exact, "tok_recall": r_exact, "tok_f1": f1_exact,
        "tok_ms_jaccard": jaccard_ms,
        "tok_precision_soft": p_soft, "tok_recall_soft": r_soft, "tok_f1_soft": f1_soft
    }

def doc_text_eval(predict_root_):
    """Evaluate predictions using only edit distance for maximum speed."""
    predicts = json.load(open(predict_root_, encoding='utf-8'))
    
    total_ned = 0.0
    total_tok_f1 = 0.0
    total_tok_f1_soft = 0.0
    num_samples = len(predicts)
    
    # Process all samples and accumulate metrics
    for ann in predicts:
        metrics = compute_metrics(ann["answer"], ann["label"])
        total_ned += metrics["ned_char"]
        total_tok_f1 += metrics["tok_f1"]
        total_tok_f1_soft += metrics["tok_f1_soft"]
    
    # Calculate and output results
    mean_dict = {
        "eval question num": num_samples,
        "ned_char": total_ned / num_samples if num_samples > 0 else 0.0,
        "tok_f1": total_tok_f1 / num_samples if num_samples > 0 else 0.0,
        "tok_f1_soft": total_tok_f1_soft / num_samples if num_samples > 0 else 0.0
    }
    
    print(json.dumps(mean_dict, indent=4))

    # also write to a file replacing .json with _eval.json
    out_file = predict_root_.replace(".json", "_eval.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(mean_dict, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_file", type=str, required=True)
    args = parser.parse_args()
    doc_text_eval(args.out_file)