## Ship It

The exercises below escalate from a pure-function mutation loop to a real GTM signal integration. Each produces observable output.

**Exercise 1 — Two-generation mutation loop on a pure function.** Take the `INITIAL_SCORER` from the Build It section. Hand-write two mutations: one that adds a funding tier check, one that adjusts the employee threshold. Evaluate each on the `LEADS` dataset. Print the source and fitness delta for each generation. Observable: before source, after source, and the score change.

```python
import copy

LEADS = [
    {"company": "Stripe", "employees": 7000, "funding_m": 600, "founded": 2010, "replied": True},
    {"company": "Notion", "employees": 400, "funding_m": 350, "founded": 2016, "replied": True},
    {"company": "Acme Corp", "employees": 12, "funding_m": 0, "founded": 2023, "replied": False},
    {"company": "Figma", "employees": 1200, "funding_m": 330, "founded": 2016, "replied": True},
    {"company": "Bob's Pizza", "employees": 3, "funding_m": 0, "founded": 2019, "replied": False},
]

def score_lead_v0(lead):
    base = 0
    if lead["employees"] > 50:
        base += 30
    if lead["funding_m"] > 10:
        base += 30
    return base

def score_lead_v1(lead):
    base = 0
    if lead["employees"] > 50:
        base += 30
    if lead["funding_m"] > 10:
        base += 30
    if lead["funding_m"] > 100:
        base += 25
    return base

def score_lead_v2(lead):
    base = 0
    if lead["employees"] > 30:
        base += 30
    if lead["funding_m"] > 10:
        base += 30
    if lead["funding_m"] > 100:
        base += 25
    return base

def evaluate(score_fn, leads):
    correct = sum(1 for l in leads if (score_fn(l) > 35) == l["replied"])
    return correct / len(leads)

versions = {"v0": score_lead_v0, "v1": score_lead_v1, "v2": score_lead_v2}
for name, fn in versions.items():
    fit = evaluate(fn, LEADS)
    print(f"{name}: fitness={fit:.2f} | threshold example: Stripe score={fn(LEADS[0])}, Acme score={fn(LEADS[2])}")

print(f"\nDelta v0→v1: {evaluate(score_lead_v1, LEADS) - evaluate(score_lead_v0, LEADS):+.2f}")
print(f"Delta v1→v2: {evaluate(score_lead_v2, LEADS) - evaluate(score_lead_v1, LEADS):+.2f}")
```

**Exercise 2 — Archive of 10 candidates with fitness-weighted parent sampling.** Run the Build It evolutionary loop for 50 generations. After it completes, compute an archive diversity metric: the number of unique fitness values in the archive. Print the diversity metric and the fitness distribution histogram (as text). A healthy archive has multiple distinct fitness levels, not a single converged value. If your archive has converged to one fitness, increase the `MARGIN` or add more mutation operators.

```python
import random
import hashlib
from collections import Counter

def archive_diversity_metric(archive):
    fitness_counts = Counter(round(a["fitness"], 2) for a in archive)
    print(f"Archive size: {len(archive)}")
    print(f"Unique fitness levels: {len(fitness_counts)}")
    print("\nFitness distribution:")
    for fit, count in sorted(fitness_counts.items()):
        bar = "#" * count
        print(f"  {fit:.2f} | {bar} ({count})")
    return len(fitness_counts)

mock_archive = [
    {"fitness": 0.40, "id": "a1"}, {"fitness": 0.40, "id": "a2"},
    {"fitness": 0.60, "id": "a3"}, {"fitness": 0.60, "id": "a4"},
    {"fitness": 0.60, "id": "a5"}, {"fitness": 0.80, "id": "a6"},
    {"fitness": 0.80, "id": "a7"}, {"fitness": 0.80, "id": "a8"},
    {"fitness": 0.80, "id": "a9"}, {"fitness": 1.00, "id": "a10"},
]

archive_diversity_metric(mock_archive)
```

**Exercise 3 — Wire the evolutionary loop to a real GTM signal.** Load historical email campaign data (even a CSV with 50 rows: company name, employees, funding, sent date, replied boolean). Replace the synthetic `LEADS` array with this data. Run the evolutionary loop for 20 generations. The fitness function becomes reply-rate prediction accuracy. After the loop completes, print the best scorer's source and apply it to 5 new prospects (not in the training data). The output is a ranked list of new prospects with predicted scores — the kind of output you would feed into a Clay enrichment waterfall as the first routing decision.

```python
import csv
import random

def load_campaign_csv(path):
    leads = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            leads.append({
                "company": row["company"],
                "employees": int(row["employees"]),
                "funding_m": float(row["funding_m"]),
                "founded": int(row["founded"]),
                "replied": row["replied"].lower() == "true",
            })
    return leads

def run_dgm_on_campaign(leads, generations=20, margin=0.05):
    INITIAL = (
        'def score_lead(lead):\n'
        '    base = 0\n'
        '    if lead["employees"] > 50:\n'
        '        base += 30\n'
        '    if lead["funding_m"] > 10:\n'
        '        base += 30\n'
        '    return base\n'
    )
    ns = {}
    exec(INITIAL, ns)
    best_fn = ns["score_lead"]
    best_score = 0
    for lead in leads:
        if (best_fn(lead) > 35) == lead["replied"]:
            best_score += 1
    best_score /= len(leads)

    archive = [(INITIAL, best_score)]
    print(f"Gen  0 | fitness: {best_score:.2f} | archive: 1")

    for gen in range(1, generations + 1):
        parent_src, parent_fit = random.choice(archive)
        thresholds = ["> 50", "> 10", "> 100", "> 30", "> 20"]
        old_thr = random.choice(thresholds)
        new_thr = random.choice([t for t in thresholds if t != old_thr])
        child_src = parent_src.replace(old_thr, new_thr) if old_thr in parent_src else parent_src

        ns2 = {}
        exec(child_src, ns2)
        child_fn = ns2["score_lead"]
        child_fit = sum(1 for l in leads if (child_fn(l) > 35) == l["replied"]) / len(leads)

        if child_fit >= parent_fit + margin:
            archive.append((child_src, child_fit))
            if child_fit > best_score:
                best_score = child_fit
                best_fn = child_fn

        if gen % 5 == 0:
            print(f"Gen {gen:2d} | best fitness: {best_score:.2f} | archive: {len(archive)}")

    return best_fn, archive

campaign_data = [
    {"company": "TechCorp", "employees": 120, "funding_m": 45, "founded": 2018, "replied": True},
    {"company": "DataFlow", "employees": 350, "funding_m": 200, "founded": 2015, "replied": True},
    {"company": "SmallShop", "employees": 4, "funding_m": 0, "founded": 2022, "replied": False},
    {"company": "CloudNine", "employees": 800, "funding_m": 150, "founded": 2014, "replied": True},
    {"company": "SoloDev", "employees": 1, "funding_m": 0, "founded": 2023, "replied": False},
    {"company": "ScaleUp", "employees": 90, "funding_m": 25, "founded": 2019, "replied": True},
    {"company": "Legacy Inc", "employees": 5000, "funding_m": 0, "founded": 2005, "replied": False},
    {"company": "NewCo", "employees": 15, "funding_m": 5, "founded": 2021, "replied": False},
    {"company": "RocketShip", "employees": 200, "funding_m": 80, "founded": 2017, "replied": True},
    {"company": "TinyLLC", "employees": 2, "funding_m": 0, "founded": 2024, "replied": False},
]

best_fn, final_archive = run_dgm_on_campaign(campaign_data, generations=20)

new_prospects = [
    {"company": "PotentialCo", "employees": 150, "funding_m": 60, "founded": 2018},
    {"company": "MaybeInc", "employees": 8, "funding_m": 2, "founded": 2023},
    {"company": "StrongSignal", "employees": 300, "funding_m": 120, "founded": 2016},
    {"company": "WeakLead", "employees": 3, "funding_m": 0, "founded": 2024},
    {"company": "Borderline", "employees": 55, "funding_m": 15, "founded": 2020},
]

print("\n=== New Prospect Rankings (best scorer) ===")
ranked = sorted(new_prospects, key=lambda p: best_fn(p), reverse=True)
for p in ranked:
    s = best_fn(p)
    verdict = "CONTACT" if s > 35 else "skip"
    print(f"  {p['company']:15s} | employees={p['employees']:4d} | funding=${p['funding_m']:.0f}M | score={s:3d} | {verdict}")

print(f"\nFinal archive size: {len(final_archive)} variants retained")
```