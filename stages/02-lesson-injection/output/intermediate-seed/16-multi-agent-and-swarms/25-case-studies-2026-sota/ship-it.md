## Ship It

**Easy:** Index your existing case study library with metadata tags and run retrieval against 10 target accounts. The code above handles five case studies and one prospect — extend the `case_studies` list with your real content, create 10 prospect profiles, and loop retrieval across all of them. Print the matched case study per account.

**Medium:** Add a reranking step that filters case studies where the prospect's reported metric already exceeds the case study outcome. Log every rejection with a reason.

```python
import json

rerank_log = []

def rerank_with_logging(scored, prospect, top_k=3):
    results = []
    for sim, cs in scored[:top_k]:
        rejection = None
        if prospect.get("industry") and prospect["industry"] != cs.industry:
            rejection = f"industry mismatch: prospect={prospect['industry']}, case_study={cs.industry}"
        elif prospect.get("use_case") and prospect["use_case"] != cs.use_case:
            rejection = f"use_case mismatch: prospect={prospect['use_case']}, case_study={cs.use_case}"
        elif prospect.get("baseline"):
            for metric, current_val in prospect["baseline"].items():
                if cs.outcome_label == metric and "_reduction" in metric:
                    projected = current_val * (1 - cs.outcome_value / 100)
                    if projected >= current_val:
                        rejection = f"baseline already exceeds outcome: {current_val} -> {projected:.1f}"
        
        entry = {
            "customer": cs.customer,
            "similarity": round(sim, 4),
            "outcome": f"{cs.outcome_label}={cs.outcome_value}",
            "decision": "REJECT" if rejection else "KEEP",
            "reason": rejection or "passed all filters",
        }
        rerank_log.append(entry)
        if not rejection:
            results.append((sim, cs))
    return results

prospect_a = {
    "company": "Stark Corp", "industry": "logistics", "employee_count": "500-1000",
    "use_case": "lead_routing",
    "baseline": {"response_time_hours": 2},
}

prospect_b = {
    "company": "Wayne Tech", "industry": "saas", "employee_count": "100-200",
    "use_case": "churn_prediction",
    "baseline": {"churn_rate_pct": 5},
}

prospects = [prospect_a, prospect_b]

for p in prospects:
    query = embed(f"{p['industry']} {p['employee_count']} {p['use_case']}")
    scored = sorted(
        [(cosine(query, embed(f"{cs.industry} {cs.employee_count} {cs.use_case}")), cs) for cs in case_studies],
        key=lambda x: x[0], reverse=True
    )
    kept = rerank_with_logging(scored, p)
    picked = kept[0][1] if kept else scored[0][1]
    print(f"\n{p['company']}: matched -> {picked.customer} ({picked.url})")

print("\n=== RERANK LOG ===")
print(json.dumps(rerank_log, indent=2))
```

Prospect A (Stark Corp) has a baseline response time of 2 hours — Acme Logistics' case study projects 4.5 hours from a 14-hour baseline, which is worse than Stark's current state. The reranker should reject it. Prospect B (Wayne Tech) matches Initech SaaS on industry, size, and use case. The log captures every decision and its reason.

**Hard:** Build an agentic evaluator that reads the drafted email and the retrieved case study, then decides whether inclusion strengthens or weakens the message. The full CLI tool takes a domain, constructs the prospect profile from firmographic data, retrieves and reranks case studies, and runs the agentic evaluation before printing the final draft.

```python
import sys
import hashlib
import math
import json
from dataclasses import dataclass

@dataclass
class CaseStudy:
    customer: str
    industry: str
    employee_count: str
    acv_range: str
    use_case: str
    outcome_label: str
    outcome_value: float
    url: str
    narrative: str

def embed(text, dims=128):
    tokens = text.lower().split()
    vec = [0.0] * dims
    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        for i in range(dims):
            vec[i] += math.sin(h * (i + 1) * 0.001)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]

def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))

DB = [
    CaseStudy("Acme Logistics", "logistics", "500-1000", "50k-100k",
              "lead_routing", "response_time_reduction_pct", 68.0,
              "https://acme.io/case", "Acme cut response time 14h to 4.5h via automated routing."),
    CaseStudy("Zenith Retail", "retail", "1000-5000", "100k-250k",
              "signal_based_outbound", "pipeline_growth_multiplier", 3.2,
              "https://zenith.io/case", "Zenith achieved 3.2x pipeline growth with signal-based outbound."),
    CaseStudy("Northwind Health", "healthcare", "200-500", "25k-50k",
              "lead_scoring", "conversion_rate_lift_pct", 42.0,
              "https://northwind.io/case", "Northwind lifted conversion 42% with AI lead scoring."),
    CaseStudy("Globex Mfg", "manufacturing", "5000-10000", "250k-500k",
              "workflow_automation", "cycle_time_reduction_pct", 55.0,
              "https://globex.io/case", "Globex cut sales cycle 55% with quoting automation."),
    CaseStudy("Initech SaaS", "saas", "100-200", "25k-50k",
              "churn_prediction", "churn_reduction_pct", 31.0,
              "https://initech.io/case", "Initech reduced churn 31% with prediction models."),
]

FIRMOGRAPHICS = {
    "vandelay.com": {"company": "Vandelay Industries", "industry": "logistics",
                     "employee_count": "500-1000", "use_case": "lead_routing",
                     "baseline": {"response_time_reduction_pct": 14}},
    "hooli.com": {"company": "Hooli", "industry": "saas",
                  "employee_count": "100-200", "use_case": "churn_prediction",
                  "baseline": {"churn_rate_pct": 8}},
    "piedpiper.com": {"company": "Pied Piper", "industry": "saas",
                      "employee_count": "50-100", "use_case": "lead_scoring",
                      "baseline": {"conversion_rate_pct": 12}},
}

def lookup_firm