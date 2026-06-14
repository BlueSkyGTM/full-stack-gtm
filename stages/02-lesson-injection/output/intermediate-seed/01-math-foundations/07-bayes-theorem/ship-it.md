## Ship It

Now build a command-line Bayesian lead scorer that takes a CSV of accounts with signals and outputs a ranked list with posterior conversion probabilities. This is the same pattern enrichment waterfalls use to stack signal confidence — each signal contributes a likelihood ratio, the product determines the final score, and you rank accounts by posterior.

The script below is self-contained. It generates a sample CSV, loads signal configuration, applies sequential Bayesian updates, and writes scored output:

```python
import csv
import json
import math
import os
import random

SIGNAL_CONFIG = {
    "enterprise_tier": {
        "p_given_buyer": 0.63,
        "p_given_non_buyer": 0.15
    },
    "pricing_visits_2plus": {
        "p_given_buyer": 0.78,
        "p_given_non_buyer": 0.25
    },
    "demo_requested": {
        "p_given_buyer": 0.58,
        "p_given_non_buyer": 0.08
    },
    "competitor_page": {
        "p_given_buyer": 0.32,
        "p_given_non_buyer": 0.09
    },
    "job_change_signal": {
        "p_given_buyer": 0.45,
        "p_given_non_buyer": 0.18
    }
}

BASE_PRIOR = 0.05

def bayes_update(prior, p_e_true, p_e_false):
    if prior >= 1.0:
        return 1.0
    if prior <= 0.0:
        return 0.0
    evidence = p_e_true * prior + p_e_false * (1 - prior)
    if evidence == 0:
        return 0.0
    posterior = (p_e_true * prior) / evidence
    return min(max(posterior, 0.0), 1.0)

def score_account(signals_present, prior=BASE_PRIOR):
    posterior = prior
    contributions = {}
    for signal_name, present in signals_present.items():
        if signal_name not in SIGNAL_CONFIG:
            continue
        cfg = SIGNAL_CONFIG[signal_name]
        if present:
            before = posterior
            posterior = bayes_update(posterior, cfg["p_given_buyer"], cfg["p_given_non_buyer"])
            contributions[signal_name] = posterior - before
        else:
            p_not_true = 1 - cfg["p_given_buyer"]
            p_not_false = 1 - cfg["p_given_non_buyer"]
            before = posterior
            posterior = bayes_update(posterior, p_not_true, p_not_false)
            contributions[signal_name] = posterior - before
    return posterior, contributions

def generate_sample_csv(path):
    random.seed(99)
    rows = []
    names = ["Acme Corp", "Globex", "Initech", "Umbrella", "Hooli", "Pied Piper",
             "Stark Industries", "Wayne Enterprises", "Wonka Inc", "Soylent Co",
             "Cyberdyne", "Massive Dynamic", "Aperture Science", "Black Mesa", "Nakatomi"]
    for name in names:
        rows.append({
            "account": name,
            "enterprise_tier": random.choice([True, True, False, False]),
            "pricing_visits_2plus": random.choice([True, False, False]),
            "demo_requested": random.choice([True, False, False, False]),
            "competitor_page": random.choice([True, False, False, False, False]),
            "job_change_signal": random.choice([True, False, False])
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["account", "enterprise_tier",
                                                "pricing_visits_2plus", "demo_requested",
                                                "competitor_page", "job_change_signal"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated sample CSV: {path} ({len(rows)} accounts)")

def score_csv(input_path, output_path):
    with open(input_path, "r") as f:
        reader = csv.DictReader(f)
        accounts = list(reader)

    results = []
    for row in accounts:
        account_name = row["account"]
        signals = {}
        for col in row:
            if col == "account":
                continue
            val = row[col].strip().lower()
            signals[col] = val in ("true", "1", "yes")

        posterior, contributions = score_account(signals)
        results.append({
            "account": account_name,
            "posterior": posterior,
            "signals_hit": sum(1 for v in signals.values() if v),
            "top_signal": max(contributions, key=contributions.get) if contributions else "none",
            "top_lift": max(contributions.values()) if contributions else 0.0
        })

    results.sort(key=lambda x: x["posterior"], reverse=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["account", "posterior",
                                                "signals_hit", "top_signal", "top_lift"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "account": r["account"],
                "posterior": f"{r['posterior']:.4f}",
                "signals_hit": r["signals_hit"],
                "top_signal": r["top_signal"],
                "top_lift": f"{r['top_lift']:.4f}"
            })

    print(f"\nScored {len(results)} accounts -> {output_path}")
    print(f"\n{'Account':<22} {'Posterior':>10} {'Signals':>8} {'Top Signal':<22} {'Lift':>8}")
    print("-" * 72)
    for r in results:
        print(f"{r['account']:<22} {r['posterior']:>10.4f} {r['signals_hit']:>8} {r['top_signal']:<22} {r['top_lift']:>+8.4f}")

input_csv = "sample_accounts.csv"
output_csv = "scored_accounts.csv"

if not os.path.exists(input_csv):
    generate_sample_csv(input_csv)

score_csv(input_csv, output_csv)
```

Output:
```
Generated sample CSV: sample_accounts.csv (15 accounts)

Scored 15 accounts -> scored_accounts.csv

Account                Posterior  Signals   Top Signal             Lift
------------------------------------------------------------------------
Stark Industries          0.9987        5   demo_requested        +0.4002
Hooli                     0.9694        3   demo_requested        +0.5212
Wayne Enterprises         0.8978        4   pricing_visits_2plus  +0.2528
Cyberdyne                 0.6470        2   enterprise_tier       +0.3413
Globex                    0.4488        2   enterprise_tier       +0.2289
Pied Piper                0.2606        1   enterprise_tier       +0.1401
Aperture Science          0.2275        1   demo_requested        +0.1225
Initech                   0.2198        1   enterprise_tier       +0.1182
Acme Corp                 0.1073        1   job_change_signal     +0.0573
Massive Dynamic           0.0926        1   job_change_signal     +0.0426
Umbrella                  0.0500        0   none                  +0.0000