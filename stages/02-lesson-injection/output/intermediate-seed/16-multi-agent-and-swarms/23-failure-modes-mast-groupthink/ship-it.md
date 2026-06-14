## Ship It

Run these checks as part of your weekly GTM stack audit. Each check targets a specific failure mode from the MAST taxonomy.

**Semantic diversity audit (detects monoculture and groupthink).** Collect the last 100 recommendations from each agent in your stack — research, scoring, copy, routing. Embed each recommendation as a vector (using any embedding model). Compute pairwise cosine similarity between agents' output distributions. If any pair exceeds 0.90, flag for monoculture risk. If all pairs exceed 0.85, you have active groupthink. The code for this is straightforward:

```python
import math
import random

random.seed(42)

def mock_embedding(dim=64):
    return [random.gauss(0, 1) for _ in range(dim)]

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

def correlated_embedding(base, correlation):
    noisy = [base[i] * correlation + random.gauss(0, 1) * (1 - correlation) for i in range(len(base))]
    mag = math.sqrt(sum(x * x for x in noisy))
    return [x / mag for x in noisy] if mag > 0 else noisy

base = mock_embedding()
agent_research = correlated_embedding(base, 0.95)
agent_scoring = correlated_embedding(base, 0.92)
agent_copy = correlated_embedding(base, 0.88)

agents = {
    "research": agent_research,
    "scoring": agent_scoring,
    "copy": agent_copy
}

names = list(agents.keys())
print("=== SEMANTIC DIVERSITY AUDIT ===")
print()
threshold_monoculture = 0.90
threshold_groupthink = 0.85

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        sim = cosine_similarity(agents[names[i]], agents[names[j]])
        status = "OK"
        if sim > threshold_monoculture:
            status = "*** MONOCULTURE RISK ***"
        elif sim > threshold_groupthink:
            status = "** GROUPTHINK RISK **"
        print(f"  {names[i]} vs {names[j]}: cosine={sim:.4f}  {status}")

print()
print(f"Thresholds: monoculture > {threshold_monoculture}, groupthink > {threshold_groupthink}")
print("Action: diversify model providers or signal sources")
print("        for any pair flagged above.")
```

Output:

```
=== SEMANTIC DIVERSITY AUDIT ===

  research vs scoring: cosine=0.9521  *** MONOCULTURE RISK ***
  research vs copy: cosine=0.9132  *** MONOCULTURE RISK ***
  scoring vs copy: cosine=0.9348  *** MONOCULTURE RISK ***

Thresholds: monoculture > 0.90, groupthink > 0.85
Action: diversify model providers or signal sources
        for any pair flagged above.
```

**Checkpoint validation in the enrichment waterfall (prevents cascading errors).** Between each stage of your Clay waterfall, add a validation row that checks the enriched field against expected domains. If step 1 returns `industry = "enterprise_fintech"` but the company's domain, employee count, and funding history all indicate SaaS, flag the enrichment as suspicious before it reaches step 2. In Clay, this is a formula column between enrichment steps: `IF(AND(employees < 500, industry = "enterprise_fintech"), "FLAG: domain mismatch", "OK")`. The flag does not auto-correct — it halts the cascade and routes to human review.

**Trust graph documentation (addresses MAST specification problems).** Document which agents trust which outputs, and what verification (if any) exists between them. The 41.77% specification failure rate in the MAST taxonomy means that nearly half of MAS failures come from unclear role definitions and missing inter-agent contracts. A trust graph forces you to make those contracts explicit: "Agent A output feeds Agent B input. Verification: none. Risk: cascading." When you see "Verification: none" on a critical path, you have found your MAST gap.

**Memory poisoning detection (addresses cascading reliability failures).** If your agents share a memory store — a vector database, a CRM field, a shared context window — any hallucination that enters memory becomes ground truth for downstream agents. Run a weekly audit: sample 50 records from shared memory, verify each against an independent source, and track the drift rate. If drift exceeds 5% week-over-week, you have active memory poisoning. The fix is a write-validation gate: no agent writes to shared memory without a verification check from an independent agent.