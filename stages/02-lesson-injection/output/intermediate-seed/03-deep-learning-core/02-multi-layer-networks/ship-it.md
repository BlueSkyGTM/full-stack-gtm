## Ship It

To deploy a forward-pass scoring model in a GTM pipeline, you need to connect the matrix operations to live data sources. The input vector in the code above maps directly to enriched fields in a tool like Clay: funding stage from Crunchbase enrichment, technology stack from BuiltWith or Wappalyzer, hiring velocity from LinkedIn job posting scrapes. The weight matrices would be learned offline from historical conversion data (which leads closed-won), then exported as serialized NumPy arrays or ONNX format for inference.

The forward pass itself is cheap — a few matrix multiplications for a single lead. The engineering work is in the pipeline: scraping signals in real time (Zone 03 in the GTM curriculum), normalizing them into the `[0, 1]` range the model expects, batching them into the input matrix `X`, and writing the scores back to your CRM or outreach tool. The handbook positions real-time signal detection — RSS feeds for funding announcements, scrapers for job postings — as the input layer for these scoring systems [CITATION NEEDED — concept: signal detection as input to scoring pipelines, Source: 80/20 GTM Engineer Handbook, Growth Lead LLC].

A practical integration pattern: run the scraper (from the Signal Machine cluster), normalize the output, pass it through the forward pass, and route leads above a score threshold into an outreach sequence. Leads below the threshold go to a nurture pool. The forward pass is the decision boundary — the same mechanism as the XOR solution, applied to firmographic data instead of Boolean logic.

```python
import numpy as np
import json

np.random.seed(42)

model = {
    "W1": np.random.randn(4, 6) * 0.4,
    "b1": np.zeros((6, 1)),
    "W2": np.random.randn(6, 1) * 0.4,
    "b2": np.zeros((1, 1)),
}

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def score_leads(leads, model, threshold=0.5):
    results = []
    for lead in leads:
        x = np.array([
            [lead["funding_normalized"]],
            [lead["tech_match_normalized"]],
            [lead["hiring_velocity_normalized"]],
            [lead["engagement_normalized"]],
        ])

        z1 = model["W1"].T @ x + model["b1"]
        a1 = relu(z1)
        z2 = model["W2"].T @ a1 + model["b2"]
        score = sigmoid(z2)[0, 0]

        action = "route_to_outbound" if score >= threshold else "route_to_nurture"
        results.append({
            "company": lead["company"],
            "score": round(float(score), 4),
            "action": action,
        })

    return results

sample_leads = [
    {"company": "Acme Corp", "funding_normalized": 0.85, "tech_match_normalized": 0.70, "hiring_velocity_normalized": 0.80, "engagement_normalized": 0.60},
    {"company": "Globex Inc", "funding_normalized": 0.20, "tech_match_normalized": 0.90, "hiring_velocity_normalized": 0.10, "engagement_normalized": 0.30},
    {"company": "Initech LLC", "funding_normalized": 0.55, "tech_match_normalized": 0.45, "hiring_velocity_normalized": 0.50, "engagement_normalized": 0.75},
    {"company": "Umbrella Corp", "funding_normalized": 0.90, "tech_match_normalized": 0.85, "hiring_velocity_normalized": 0.95, "engagement_normalized": 0.80},
]

scored = score_leads(sample_leads, model, threshold=0.5)

print(json.dumps(scored, indent=2))

outbound = [r for r in scored if r["action"] == "route_to_outbound"]
nurture = [r for r in scored if r["action"] == "route_to_nurture"]
print(f"\nOutbound queue: {len(outbound)} leads")
print(f"Nurture pool: {len(nurture)} leads")
```

The output is a JSON payload you could write directly to a Clay table, a CRM webhook, or a Slack notification. The forward pass ran four matrix multiplications per lead and produced a routing decision. That is the shipped form of a multi-layer network in a GTM context.