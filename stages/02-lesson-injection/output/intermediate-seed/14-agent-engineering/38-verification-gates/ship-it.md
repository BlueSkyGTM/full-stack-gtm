## Ship It

Shipping verification gates to production means answering two questions on an ongoing basis: are the gates catching real errors, and are they rejecting good data? The first question is about recall — if you disable the gates, how much garbage enters your CRM? The second is about precision — are your gates so strict that legitimate enrichment results are being rejected and sent to more expensive fallback providers unnecessarily?

A gate audit gives you both numbers. Run every gate against a labeled dataset of enrichment results where you know the ground truth (manually verified records). Count true positives (correctly rejected bad data), false positives (incorrectly rejected good data), true negatives (correctly accepted good data), and false negatives (incorrectly accepted bad data). The false negative rate tells you which failure modes your gates are missing — maybe you need a new plausibility bound or a new consistency check. The false positive rate tells you which gates are too strict — maybe your email regex rejects valid plus-addressing or your employee count ceiling is too low for enterprise companies.

```python
labeled_data = [
    {"record": {"email": "valid@company.com", "employee_count": 500,
                "domain": "company.com", "company_name": "Company"},
     "ground_truth": "good"},
    {"record": {"email": "bad-email", "employee_count": 500,
                "domain": "company.com", "company_name": "Company"},
     "ground_truth": "bad"},
    {"record": {"email": "valid@company.com", "employee_count": -10,
                "domain": "company.com", "company_name": "Company"},
     "ground_truth": "bad"},
    {"record": {"email": "valid@company.com", "employee_count": 500,
                "domain": "other.com", "company_name": "Company"},
     "ground_truth": "bad"},
    {"record": {"email": "user+tag@company.com", "employee_count": 3,
                "domain": "company.com", "company_name": "Company"},
     "ground_truth": "good"},
    {"record": {"email": "valid@company.com", "employee_count": 0,
                "domain": "company.com", "company_name": "Company"},
     "ground_truth": "bad"},
]

gates = [format_gate_email, plausibility_gate_employees, consistency_gate_domain]

tp = fp = tn = fn = 0
details = {"true_positive": [], "false_positive": [],
           "true_negative": [], "false_negative": []}

for item in labeled_data:
    record = item["record"]
    truth = item["ground_truth"]
    passed, reason = composite_gate(record, gates)
    predicted = "good" if passed else "bad"

    if predicted == "bad" and truth == "bad":
        tp += 1
        details["true_positive"].append((record.get("email", "?"), reason))
    elif predicted == "bad" and truth == "good":
        fp += 1
        details["false_positive"].append((record.get("email", "?"), reason))
    elif predicted == "good" and truth == "good":
        tn += 1
        details["true_negative"].append((record.get("email", "?"),))
    else:
        fn += 1
        details["false_negative"].append((record.get("email", "?"), reason))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print("=" * 60)
print("GATE AUDIT REPORT")
print("=" * 60)
print(f"Total records audited:  {len(labeled_data)}")
print(f"True positives (caught bad):  {tp}")
print(f"False positives (rejected good): {fp}")
print(f"True negatives (accepted good): {tn}")
print(f"False negatives (let bad through): {fn}")
print(f"\nGate precision: {precision:.1%} (of rejections, how many were truly bad)")
print(f"Gate recall:    {recall:.1%} (of bad records, how many were caught)")

if fp > 0:
    print(f"\n--- FALSE POSITIVES (good data rejected) ---")
    for email, reason in details["false_positive"]:
        print(f"  {email}: {reason}")

if fn > 0:
    print(f"\n--- FALSE NEGATIVES (bad data accepted) ---")
    for email, reason in details["false_negative"]:
        print(f"  {email}: {reason}")
print("=" * 60)
```

The audit reveals a false positive: `user+tag@company.com` is a valid email (plus-addressing is RFC-compliant), but the format gate's regex rejects it because the `+` character is not in the character class. This is a gate calibration problem — the regex needs to be widened. The audit also confirms the plausibility gate catches the zero-employee-count record (a common provider artifact for companies with no reported headcount) and the consistency gate catches the mismatched domain. Running this audit on a sample of 100-200 manually verified records before deploying gates to production will surface calibration issues before they affect 10,000 records.

In a Clay production workflow, you ship gates by adding them as formula columns that evaluate enrichment output, then referencing those columns in the conditional run settings of subsequent waterfall steps. The formula column is version-controlled in your Clay table schema. When you change a gate — widening the email regex, adjusting the employee count ceiling — you are modifying the predicate that controls your enrichment pipeline. Treat gate changes with the same review discipline as database migrations: test against labeled data first, deploy to a small batch, verify the audit numbers, then scale.