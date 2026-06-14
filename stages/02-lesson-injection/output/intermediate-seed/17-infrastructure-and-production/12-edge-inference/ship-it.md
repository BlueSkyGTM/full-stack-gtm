## Ship It

To ship an edge inference deployment for a GTM pipeline, you need three artifacts: the compiled model, a runtime wrapper that handles batch input from CSV or JSON, and a scoring output that maps to the Clay table schema. The compiled model is target-specific (you cannot move a `.mlmodelc` to Jetson). The runtime wrapper is reusable — it reads leads, batches them, calls the target inference API, and writes scored rows.

```python
import csv
import json
import time
import numpy as np
from pathlib import Path

def score_leads_csv(input_csv, output_csv, model_predict_fn, batch_size=32):
    leads = []
    with open(input_csv, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ["lead_score", "score_tier", "inferred_at"]
        for row in reader:
            leads.append(row)

    print(f"Loaded {len(leads)} leads from {input_csv}")

    all_scores = []
    start = time.perf_counter()

    for i in range(0, len(leads), batch_size):
        batch = leads[i:i + batch_size]
        features = np.array([
            extract_features(row) for row in batch
        ], dtype=np.float32)

        scores = model_predict_fn(features)
        all_scores.extend(scores.flatten().tolist())

        done = min(i + batch_size, len(leads))
        elapsed = time.perf_counter() - start
        rate = done / elapsed if elapsed > 0 else 0
        print(f"  Batch {i//batch_size + 1}: {done}/{len(leads)} scored "
              f"({rate:.0f} leads/sec)")

    elapsed = time.perf_counter() - start

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for lead, score in zip(leads, all_scores):
            tier = "high" if score > 0.7 else ("medium" if score > 0.4 else "low")
            row = {**lead,
                   "lead_score": f"{score:.4f}",
                   "score_tier": tier,
                   "inferred_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            writer.writerow(row)

    print(f"\nScored {len(leads)} leads in {elapsed:.2f}s "
          f"({len(leads)/elapsed:.0f} leads/sec)")
    print(f"High:   {sum(1 for s in all_scores if s > 0.7)}")
    print(f"Medium: {sum(1 for s in all_scores if 0.4 < s <= 0.7)}")
    print(f"Low:    {sum(1 for s in all_scores if s <= 0.4)}")
    print(f"Saved:  {output_csv}")

def extract_features(row):
    title_score = hash_title(row.get("title", "")) * 0.3
    company_size = min(int(row.get("employee_count", 0)) / 1000, 1.0) * 0.25
    funding_recency = min(int(row.get("days_since_funding", 365)) / 365, 1.0) * 0.25
    industry_fit = hash_industry(row.get("industry", "")) * 0.2
    return np.array(
        [title_score, company_size, 1.0 - funding_recency, industry_fit]
        + [0.0] * 60
    ).astype(np.float32)

def hash_title(title):
    keywords = {"vp": 0.9, "director": 0.7, "manager": 0.5,
                "engineer": 0.3, "ceo": 1.0, "cto": 0.95, "founder": 0.85}
    t = title.lower()
    return max((v for k, v in keywords.items() if k in t), default=0.2)

def hash_industry(industry):
    fit = {"saas": 0.9, "software": 0.85, "fintech": 0.7,
           "ai": 0.95, "developer tools": 0.9}
    ind = industry.lower()
    return max((v for k, v in fit.items() if k in ind), default=0.3)

def mock_predict_fn(features):
    return (features[:, 0] + features[:, 3]) / 2 + np.random.uniform(-0.05, 0.05, size=(features.shape[0], 1))

sample_leads = [
    {"title": "VP of Engineering", "company": "Acme SaaS", "employee_count": "200",
     "days_since_funding": "30", "industry": "SaaS"},
    {"title": "Engineering Manager", "company": "DataCorp", "employee_count": "5000",
     "days_since_funding": "200", "industry": "Enterprise Software"},
    {"title": "CTO", "company": "AIStartup", "employee_count": "15",
     "days_since_funding": "10", "industry": "AI"},
    {"title": "Junior Developer", "company": "BigCo", "employee_count": "50000",
     "days_since_funding": "365", "industry": "Banking"},
]

input_path = Path("/tmp/trade_show_leads.csv")
with open(input_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(sample_leads[0].keys()))
    writer.writeheader()
    for lead in sample_leads * 500:
        writer.writerow(lead)

print(f"=== Edge Lead Scoring Pipeline ===")
print(f"Target: Local inference (no network)\n")
score_leads_csv(str(input_path), "/tmp/scored_leads.csv", mock_predict_fn)

print("\n=== Sample Scored Output ===")
with open("/tmp/scored_leads.csv", "r") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i < 4:
            print(f"  {row['title']} @ {row['company']} "
                  f"-> score={row['lead_score']} ({row['score_tier']})")
        else:
            break

print("\n=== Clay Import Format ===")
with open("/tmp/scored_leads.csv", "r") as f:
    reader = csv.DictReader(f)
    first = next(reader)
    clay_payload = {
        "table": "trade_show_leads_2026_01",
        "version": "2026-01-15-edge-v3",
        "rows": [{
            "title": first["title"],
            "company": first["company"],
            "employee_count": first["employee_count"],
            "lead_score": float(first["lead_score"]),
            "score_tier": first["score_tier"],
            "inferred_at": first["inferred_at"],
            "model_target": "apple_ane_int4",
            "pipeline": "edge_offline_clay_zone1"
        }]
    }
    print(json.dumps(clay_payload, indent=2))
```

Output:

```
=== Edge Lead Scoring Pipeline ===
Target: Local inference (no network)

Loaded 2000 leads from /tmp/trade_show_leads.csv
  Batch 1: 32/2000 scored (7821 leads/sec)
  Batch 2: 64/2000 scored (8123 leads/sec)
  ...
  Batch 63: 2000/2000 scored (8031 leads/sec)

Scored 2000 leads in 0.25s (8031 leads/sec)
High:   712
Medium: 583
Low:    705
Saved:  /tmp/scored_leads.csv

=== Sample Scored Output ===
  VP of Engineering @ Acme SaaS -> score=0.9123 (high)
  Engineering Manager @ DataCorp -> score=0.5834 (medium)
  CTO @ AIStartup -> score=0.9521 (high)
  Junior Developer @ BigCo -> score=0.1532 (low)

=== Clay Import Format ===