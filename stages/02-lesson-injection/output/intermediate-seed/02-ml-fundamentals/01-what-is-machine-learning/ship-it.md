## Ship It

Build a production-shaped lead scorer. Generate 200 mock leads with three features (normalized employee count, pages visited, email opens) and a binary conversion label derived from a weighted signal with 10% label noise to simulate real-world messiness. Split 80/20 into train and test sets, evaluate accuracy on held-out data, then score five new inbound leads and emit each as a JSON object.

```python
import json
import random
import math
from collections import Counter

random.seed(42)

def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def knn_proba(train, query, k=5):
    distances = []
    for features, label in train:
        dist = euclidean(features, query)
        distances.append((dist, label))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    positive = sum(1 for _, label in neighbors if label == 1)
    return positive / k

leads = []
for i in range(200):
    emp = random.randint(5, 5000)
    pages = random.randint(1, 25)
    opens = random.randint(0, 10)

    signal = (emp / 5000) * 0.3 + (pages / 25) * 0.4 + (opens / 10) * 0.3
    converted = 1 if signal > 0.45 else 0
    if random.random() < 0.1:
        converted = 1 - converted

    features = [emp / 5000, pages / 25, opens / 10]
    meta = {"company": f"Company_{i:03d}", "employee_count": emp,
            "pages_visited": pages, "email_opens": opens}
    leads.append((features, converted, meta))

split = int(len(leads) * 0.8)
train_set = [(f, l) for f, l, _ in leads[:split]]
test_set = leads[split:]

correct = 0
for features, true_label, _ in test_set:
    prob = knn_proba(train_set, features, k=5)
    pred = 1 if prob >= 0.5 else 0
    if pred == true_label:
        correct += 1

print(f"Model accuracy on {len(test_set)} held-out leads: {correct/len(test_set):.1%}")
print()

inbound_leads = [
    ([0.62, 0.80, 0.70], "Acme Corp", 3100, 20, 7),
    ([0.12, 0.20, 0.10], "Beta LLC", 600, 5, 1),
    ([0.84, 0.92, 0.80], "Gamma Inc", 4200, 23, 8),
    ([0.30, 0.44, 0.30], "Delta Co", 1500, 11, 3),
    ([0.50, 0.60, 0.50], "Epsilon Ltd", 2500, 15, 5),
]

scored = []
for features, company, emp, pages, opens in inbound_leads:
    prob = knn_proba(train_set, features, k=5)
    scored.append({
        "company": company,
        "employee_count": emp,
        "pages_visited": pages,
        "email_opens": opens,
        "conversion_probability": round(prob, 3),
        "action": "prioritize" if prob >= 0.6 else "nurture" if prob >= 0.3 else "disqualify",
    })

scored.sort(key=lambda x: x["conversion_probability"], reverse=True)

print("--- Ranked Inbound Leads (JSON) ---")
for s in scored:
    print(json.dumps(s))
```

The output is five JSON objects ranked by conversion probability, each with an action recommendation. That JSON is what an enrichment waterfall consumes — the model runs after enrichment populates firmographic and behavioral features, and the score determines whether the lead enters an outreach sequence or gets deprioritized. The accuracy number tells you whether the model is trustworthy enough to act on. If accuracy is below ~65% on the held-out set, the features are not informative enough or the dataset is too noisy, and you need better signals before shipping the scorer into a production workflow.

---