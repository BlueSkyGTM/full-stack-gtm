## Ship It

A coordination benchmark gate in production means: no model or prompt ships without passing the benchmark. Here's a self-contained evaluation harness that creates a golden dataset, runs inference, computes accuracy and per-class precision/recall/F1, and enforces a coordination gate — if any class F1 drops below 0.7, the script exits with code 1. Wire this into CI as a pre-merge check on any change to the classifier or its prompt.

```python
import sys
import json
from collections import defaultdict

GOLDEN_DATASET = [
    {"text": "this looks great, let's talk next week", "label": "interested"},
    {"text": "send me a calendar invite", "label": "interested"},
    {"text": "very interested, what are next steps", "label": "interested"},
    {"text": "let's get this done", "label": "interested"},
    {"text": "out of office until monday", "label": "out_of_office"},
    {"text": "ooo, returning wednesday", "label": "out_of_office"},
    {"text": "i'm away from the office", "label": "out_of_office"},
    {"text": "currently out of office", "label": "out_of_office"},
    {"text": "not interested, remove me from your list", "label": "not_interested"},
    {"text": "we already have a vendor", "label": "not_interested"},
    {"text": "no budget this year", "label": "not_interested"},
    {"text": "stop emailing me", "label": "not_interested"},
    {"text": "talk to my colleague sarah about this", "label": "referral"},
    {"text": "mike on my team handles this", "label": "referral"},
    {"text": "forward this to jennifer", "label": "referral"},
    {"text": "you should reach out to david", "label": "referral"},
]

def classify(text):
    t = text.lower()
    if any(p in t for p in ["out of office", "ooo", "away from the office"]):
        return "out_of_office"
    if any(p in t for p in ["not interested", "remove me", "no budget", "stop emailing", "already have a vendor"]):
        return "not_interested"
    if any(p in t for p in ["colleague", "talk to", "forward this", "reach out to", "handles this"]):
        return "referral"
    return "interested"

def compute_metrics(records, predictions):
    truths = [r["label"] for r in records]
    classes = sorted(set(truths))
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    correct = 0
    for truth, pred in zip(truths, predictions):
        if pred == truth:
            stats[truth]["tp"] += 1
            correct += 1
        else:
            stats[pred]["fp"] += 1
            stats[truth]["fn"] += 1
    accuracy = correct / len(records)
    per_class = {}
    for cls in classes:
        s = stats[cls]
        p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0.0
        r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[cls] = {"precision": p, "recall": r, "f1": f1}
    return accuracy, per_class

predictions = [classify(r["text"]) for r in GOLDEN_DATASET]
accuracy, per_class = compute_metrics(GOLDEN_DATASET, predictions)

F1_THRESHOLD = 0.7
print("=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"Records: {len(GOLDEN_DATASET)}")
print(f"Overall accuracy: {accuracy:.2%}")
print()
print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Status'}")
print("-" * 72)

all_pass = True
for cls, m in per_class.items():
    status = "PASS" if m["f1"] >= F1_THRESHOLD else "FAIL"
    if m["f1"] < F1_THRESHOLD:
        all_pass = False
    print(f"{cls:<20} {m['precision']:<12.2%} {m['recall']:<12.2%} {m['f1']:<12.2%} {status}")

print()
if all_pass:
    print(f"COORDINATION GATE: PASS (all class F1 >= {F1_THRESHOLD})")
    sys.exit(0)
else:
    print(f"COORDINATION GATE: FAIL (one or more class F1 < {F1_THRESHOLD})")
    sys.exit(1)
```

Run this and you'll see a per-class breakdown with pass/fail per metric and a coordination gate decision at the end. The exit code is what your CI pipeline checks: `exit 0` means the change is safe to merge, `exit 1` means it's not. To wire this into CI, save the script as `eval_gate.py`, add a golden dataset JSONL loader if your dataset outgrows inline, and add a step to your GitHub Actions or GitLab CI config that runs `python eval_gate.py` on every PR touching the classifier.

The distributed systems analogy holds here just as it does for enrichment waterfalls: multiple independent scorers running in parallel, each producing a signal, with a coordination protocol (the conjunction gate) that requires agreement before proceeding. The same pattern — parallel workers, independent failure modes, a join barrier — appears whether you're coordinating enrichment API calls or coordinating evaluation metrics across model classes.