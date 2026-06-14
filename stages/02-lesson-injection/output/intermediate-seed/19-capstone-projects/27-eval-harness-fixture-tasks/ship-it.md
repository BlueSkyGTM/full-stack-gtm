## Ship It

Store fixtures in version control alongside your pipeline code — same repository, same PR review process. When somebody changes a prompt template, a parser, or a provider configuration, the harness runs in CI against the full fixture set. If the aggregate score drops below a threshold you define, the build fails. This is the same pattern as running unit tests on a PR that touches application code.

Write results to a structured format — JSON or CSV — on every run. This gives you a time series. When a customer reports bad enrichment data two weeks from now, you pull the historical results and see whether the regression appeared in a specific commit. Without this log, you are reconstructing history from memory.

Here is a harness wrapper that exports to CSV and enforces a CI gate:

```python
import csv
import json
from datetime import datetime, timezone

def run_harness_with_export(fixtures, pipeline, scorers, threshold=0.95, output_path=None):
    report = run_harness(fixtures, pipeline, scorers, verbose=False)
    
    timestamp = datetime.now(timezone.utc).isoformat()
    
    if output_path:
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "fixture_id", "expected", "actual", "score", "passed"])
            for r in report["results"]:
                writer.writerow([timestamp, r["id"], r["expected"], r["actual"], r["score"], r["score"] >= 1.0])
        print(f"Results exported to {output_path}")
    
    print(f"\nCI Gate: mean score {report['mean_score']:.3f} vs threshold {threshold}")
    if report["mean_score"] >= threshold:
        print("GATE PASSED")
        return True
    else:
        print("GATE FAILED — blocking merge")
        return False

passed = run_harness_with_export(
    fixtures,
    route_pipeline,
    SCORERS,
    threshold=0.95,
    output_path="eval_results.csv"
)

print(f"\nCI decision: {'PASS' if passed else 'FAIL'}")
```

Output:

```
Results exported to eval_results.csv

CI Gate: mean score 1.000 vs threshold 0.95
GATE PASSED

CI decision: PASS
```

Run the same gate against the broken pipeline to see it block:

```python
passed_broken = run_harness_with_export(
    fixtures,
    broken_route_pipeline,
    SCORERS,
    threshold=0.95,
    output_path="eval_results_broken.csv"
)

print(f"\nCI decision: {'PASS' if passed_broken else 'FAIL'}")
```

Output:

```
CI Gate: mean score 0.600 vs threshold 0.95
GATE FAILED — blocking merge

CI decision: FAIL
```

For multi-scorer breakdown — the pattern you use when a fixture needs both structural and semantic validation:

```python
def run_multi_scorer_harness(fixtures, pipeline, scorer_map, verbose=True):
    all_results = []
    for fixture in fixtures:
        actual = pipeline(fixture["input"])
        fixture_scores = {}
        for scorer_name, scorer_fn in scorer_map.items():
            score = scorer_fn(fixture["expected"], actual)
            fixture_scores[scorer_name] = score
        all_results.append({"id": fixture["id"], "scores": fixture_scores, "actual": actual})
        if verbose:
            score_str = " ".join(f"{k}={v:.1f}" for k, v in fixture_scores.items())
            print(f"  {fixture['id']}: {score_str}")
    
    print(f"\n{'='*50}")
    print("Per-scorer breakdown:")
    for scorer_name in scorer_map:
        scores = [r["scores"][scorer_name] for r in all_results]
        mean = sum(scores) / len(scores)
        pass_rate = sum(1 for s in scores if s >= 1.0) / len(scores)
        print(f"  {scorer_name:20s}  mean={mean:.3f}  pass_rate={pass_rate:.1%}")
    print(f"{'='*50}")
    return all_results

multi_fixtures = [
    {"id": "multi-001", "input": {"company_name": "Stripe"},
     "expected": "stripe.com", "scorer": "exact_match"},
    {"id": "multi-002", "input": {"company_name": "GitHub"},
     "expected": "github.com", "scorer": "exact_match"},
    {"id": "multi-003", "input": {"company_name": "Notion"},
     "expected": "notion.so", "scorer": "exact_match"},
]

multi_scorer_map = {
    "exact_match": lambda exp, act: 1.0 if exp == act else 0.0,
    "is_non_empty": lambda exp, act: 1.0 if act and len(str(act)) > 0 else 0.0,
    "ends_with_com": lambda exp, act: 1.0 if str(act).endswith(".com") else 0.0,
}

results = run_multi_scorer_harness(multi_fixtures, route_pipeline, multi_scorer_map)
```

Output:

```
  multi-001: exact_match=1.0 is_non_empty=1.0 ends_with_com=1.0
  multi-002: exact_match=1.0 is_non_empty=1.0 ends_with_com=1.0
  multi-003: exact_match=1.0 is_non_empty=1.0 ends_with_com=0.0

==================================================
Per-scorer breakdown:
  exact_match           mean=1.000  pass_rate=100.0%
  is_non_empty          mean=1.000  pass_rate=100.0%
  ends_with_com         mean=0.667  pass_rate=66.7%
==================================================
```

The `ends_with_com` scorer flags that Notion uses `.so`, not `.com`. This is the kind of structural insight a single scorer misses — exact match passes, but the `.com` assumption baked into downstream code (say, a URL builder that appends paths) would silently break for `.so`, `.io`, or `.ai` domains. Multiple scorers surface different categories of failure from the same fixture set.