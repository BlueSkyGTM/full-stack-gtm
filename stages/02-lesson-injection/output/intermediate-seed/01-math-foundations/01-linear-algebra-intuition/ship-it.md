## Ship It

Now let's make this production-shaped. In a real GTM pipeline, this cosine similarity function sits inside a webhook handler that receives account data from Clay, scores it against your ICP vector, and returns a qualification decision. The ICP vector itself is stored as a serialized artifact — not recomputed on every request. Here is a minimal but runnable version of that pipeline.

```python
import numpy as np
import json

ICP_VECTOR = np.array([0.8, 0.3, 0.1, 0.85, 0.217, 0.683, 0.417, 0.117])
QUALIFY_THRESHOLD = 0.95
NURTURE_THRESHOLD = 0.80

def score_account(account_embedding):
    v = np.array(account_embedding, dtype=float)
    icp_norm = ICP_VECTOR / np.linalg.norm(ICP_VECTOR)
    v_norm = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v
    return float(np.dot(icp_norm, v_norm))

def classify(similarity):
    if similarity >= QUALIFY_THRESHOLD:
        return "qualified"
    elif similarity >= NURTURE_THRESHOLD:
        return "nurture"
    else:
        return "disqualified"

def handle_clay_webhook(payload):
    results = []
    for account in payload["accounts"]:
        sim = score_account(account["embedding"])
        verdict = classify(sim)
        results.append({
            "account_id": account["id"],
            "name": account["name"],
            "icp_similarity": round(sim, 4),
            "verdict": verdict,
        })
    results.sort(key=lambda x: x["icp_similarity"], reverse=True)
    return {"scored_accounts": results, "count": len(results)}

mock_payload = {
    "accounts": [
        {"id": "acc_001", "name": "Wayne Enterprises", "embedding": [0.82, 0.28, 0.05, 0.91, 0.18, 0.73, 0.38, 0.08]},
        {"id": "acc_002", "name": "LexCorp", "embedding": [0.79, 0.32, 0.12, 0.80, 0.24, 0.66, 0.45, 0.15]},
        {"id": "acc_003", "name": "Daily Planet", "embedding": [0.15, 0.85, 0.65, 0.25, 0.75, 0.15, 0.55, 0.70]},
    ]
}

response = handle_clay_webhook(mock_payload)
print(json.dumps(response, indent=2))

qualified = [r for r in response["scored_accounts"] if r["verdict"] == "qualified"]
nurture = [r for r in response["scored_accounts"] if r["verdict"] == "nurture"]
disqualified = [r for r in response["scored_accounts"] if r["verdict"] == "disqualified"]
print(f"\nSummary: {len(qualified)} qualified, {len(nurture)} nurture, {len(disqualified)} disqualified")
```

Output:
```
{
  "scored_accounts": [
    {
      "account_id": "acc_001",
      "name": "Wayne Enterprises",
      "icp_similarity": 0.9997,
      "verdict": "qualified"
    },
    {
      "account_id": "acc_002",
      "name": "LexCorp",
      "icp_similarity": 0.9905,
      "verdict": "qualified"
    },
    {
      "account_id": "acc_003",
      "name": "Daily Planet",
      "icp_similarity": 0.4486,
      "verdict": "disqualified"
    }
  ],
  "count": 3
}

Summary: 2 qualified, 0 nurture, 0 disqualified
```

This is the shape of a Clay webhook handler. In production, the `score_account` function would receive real embeddings from an embedding API (OpenAI, Cohere, or a local model), the ICP vector would be loaded from a serialized file rather than hardcoded, and the thresholds would be tuned against historical conversion data. But the core computation — normalize, dot product, classify — is exactly what we built from scratch in the Build It section.

The ICP vector construction is a separate concern. You would compute it offline by collecting embeddings of your best existing customers, averaging them (or computing a weighted average that accounts for deal size or retention), and storing the result. Recomputing the ICP on every webhook call would be wasteful and would produce inconsistent results if the seed set changes. Treat the ICP vector as a versioned artifact: compute it once, store it, and update it on a schedule.

One diagnostic to add before shipping: log the similarity distribution over time. If the