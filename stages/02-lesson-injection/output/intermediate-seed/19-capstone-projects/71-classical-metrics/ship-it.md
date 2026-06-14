## Ship It

Write a metrics report function that takes ground truth and predictions, returns everything as a dictionary, and prints a formatted summary. This is what you attach to any model evaluation PR — classification or generation.

```python
import re
import math
import json
from collections import Counter

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

def tokenize(text):
    return TOKEN_RE.findall(text.lower())

def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def lcs_length(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def metrics_report(y_true=None, y_pred=None, candidate_text=None, reference_texts=None):
    report = {}
    
    if y_true is not None and y_pred is not None:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        cm = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
        
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
        
        report["classification"] = {
            "confusion_matrix": cm,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "specificity": round(spec, 4),
            "accuracy": round(acc, 4),
        }
    
    if candidate_text is not None and reference_texts is not None:
        candidate = tokenize(candidate_text)
        references = [tokenize(rt) for rt in reference_texts]
        
        bp = brevity_penalty_impl(candidate, references)
        precisions = []
        for n in range(1, 5):
            mp = modified_precision_impl(candidate, references, n)
            precisions.append(mp if mp > 0 else 1e-10)
        bleu = bp * math.exp(sum(math.log(p) for p in precisions) / 4)
        
        best_rouge = 0.0
        for ref in references:
            if not candidate or not ref:
                continue
            lcs = lcs_length(candidate, ref)
            if lcs == 0:
                continue
            r_lcs = lcs / len(ref)
            p_lcs = lcs / len(candidate)
            f_lcs = (2 * p_lcs * r_lcs) / (p_lcs + r_lcs) if (p_lcs + r_lcs) else 0.0
            best_rouge = max(best_rouge, f_lcs)
        
        report["generation"] = {
            "bleu4": round(bleu, 4),
            "rouge_l": round(best_rouge, 4),
        }
    
    return report

def brevity_penalty_impl(candidate, references):
    cand_len = len(candidate)
    ref_lens = [len(r) for r in references]
    closest = min(ref_lens, key=lambda l: (abs(l - cand_len), l))
    if cand_len > closest:
        return 1.0
    return math.exp(1 - closest / cand_len) if cand_len else 0.0

def modified_precision_impl(candidate, references, n):
    cand_ngrams = Counter(get_ngrams(candidate, n))
    max_ref = Counter()
    for ref in references:
        ref_counts = Counter(get_ngrams(ref, n))
        for gram, count in ref_counts.items():
            max_ref[gram] = max(max_ref[gram], count)
    clipped = sum(min(c, max_ref.get(g, 0)) for g, c in cand_ngrams.items())
    total = sum(cand_ngrams.values())
    return clipped / total if total else 0.0

y_true = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
cand = "our platform reduces deployment time by forty percent"
refs = ["our platform cuts deployment time by forty percent", "deployment time is reduced by forty percent with our platform"]

report = metrics_report(y_true=y_true, y_pred=y_pred, candidate_text=cand, reference_texts=refs)
print(json.dumps(report, indent=2))
```

The function dispatches on what you pass it — classification arrays, generation strings, or both. Attach the JSON output to your PR and reviewers can see exactly what moved. No libraries beyond stdlib, so the numbers are reproducible without `pip install`.