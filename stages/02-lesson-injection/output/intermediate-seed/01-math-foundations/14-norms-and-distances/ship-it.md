## Ship It

Now we build a reusable `nearest_neighbors.py` that loads named entities with embeddings from a JSON file, accepts a query vector, and returns ranked results. This script is the backbone of any "find similar companies" or "score this lead against historical wins" workflow.

```python
import json
import math
import sys

def l1_norm(v):
    return sum(abs(x) for x in v)

def l2_norm(v):
    return math.sqrt(sum(x ** 2 for x in v))

def euclidean(u, v):
    return math.sqrt(sum((u[i] - v[i]) ** 2 for i in range(len(u))))

def manhattan(u, v):
    return sum(abs(u[i] - v[i]) for i in range(len(u)))

def cosine(u, v):
    dot = sum(u[i] * v[i] for i in range(len(u)))
    return 1.0 - dot / (l2_norm(u) * l2_norm(v))

METRICS = {
    "euclidean": euclidean,
    "manhattan": manhattan,
    "cosine": cosine,
}

def nearest_neighbors(query, candidates, metric="cosine", k=5):
    dist_fn = METRICS[metric]
    scored = []
    for candidate in candidates:
        d = dist_fn(query, candidate["embedding"])
        scored.append((candidate["name"], d, candidate.get("description", "")))
    scored.sort(key=lambda x: x[1])
    return scored[:k]

sample_data = [
    {"name": "Stripe", "description": "Payment infrastructure", "embedding": [0.82, 0.10, 0.03, 0.02, 0.01, 0.02]},
    {"name": "Plaid", "description": "Financial data API", "embedding": [0.71, 0.18, 0.05, 0.03, 0.02, 0.01]},
    {"name": "Square", "description": "Payment processing hardware", "embedding": [0.68, 0.20, 0.04, 0.03, 0.03, 0.02]},
    {"name": "Snowflake", "description": "Cloud data warehouse", "embedding": [0.08, 0.05, 0.79, 0.04, 0.02, 0.02]},
    {"name": "Databricks", "description": "Unified analytics platform", "embedding": [0.12, 0.08, 0.71, 0.05, 0.02, 0.02]},
    {"name": "Vercel", "description": "Frontend cloud platform", "embedding": [0.25, 0.15, 0.10, 0.72, 0.08, 0.03]},
]

with open("companies.json", "w") as f:
    json.dump(sample_data, f, indent=2)

with open("companies.json") as f:
    companies = json.load(f)

query = companies[0]["embedding"]
query_name = companies[0]["name"]
candidates = companies[1:]

print(f"Query: {query_name}")
print(f"Embedding: {query}")
print()

for metric in ["cosine", "euclidean", "manhattan"]:
    results = nearest_neighbors(query, candidates, metric=metric, k=3)
    print(f"--- Top 3 by {metric} distance ---")
    for rank, (name, dist, desc) in enumerate(results, 1):
        print(f"  {rank}. {name:15s}  d={dist:.4f}  ({desc})")
    print()
```

Run this script and compare the rankings across metrics. For the Stripe query, cosine distance should rank Plaid and Square at the top — both are fintech. Euclidean may produce a similar ranking here because the vectors are low-dimensional and well-separated, but in production with 384-dimensional embeddings from real text, the divergence between metrics becomes pronounced. The script accepts `metric` as a parameter so you can A/B test which metric produces rankings that align with your GTM team's qualitative judgment of "these companies are similar."

To use this with real embeddings, replace the `sample_data` block with a call to your embedding API or model, write the results to `companies.json`, and pass the query vector from your target account. The distance functions do not change — only the input vectors get larger.

---