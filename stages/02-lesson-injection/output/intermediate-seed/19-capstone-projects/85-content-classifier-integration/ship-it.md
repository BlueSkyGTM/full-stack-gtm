## Ship It

Deploy the classifier endpoint to a persistent environment — a containerized FastAPI service behind a load balancer, or a serverless function with a cold-start budget that fits your latency requirements. The endpoint accepts POST requests with a JSON body containing the content payload, runs inference, applies thresholds, and returns the routing decision synchronously if latency allows or emits it to a queue for asynchronous processing.

Monitoring catches what training accuracy does not. Track three signals over time: classification distribution (the percentage of payloads in each category), average confidence per category, and routing volume per destination. If a category's average confidence drops below its threshold for 50 consecutive classifications, that indicates either concept drift — the inbound content has shifted away from the distribution the model was trained on — or a data quality problem where malformed or truncated payloads are reaching the classifier. Either case requires intervention: retraining on fresh data, fixing the upstream payload parser, or temporarily tightening the threshold to push more traffic into human review while the root cause is addressed.

```python
from collections import deque
import statistics

confidence_history = {
    "sales": deque(maxlen=50),
    "support": deque(maxlen=50),
    "nurture": deque(maxlen=50),
    "discard": deque(maxlen=50),
}

drift_thresholds = {cat: thresholds[cat] for cat in thresholds}

def log_and_check_drift(category, confidence):
    if category not in confidence_history:
        return None
    confidence_history[category].append(confidence)
    window = list(confidence_history[category])
    if len(window) < 50:
        return {"category": category, "status": "warming up", "samples": len(window)}

    avg_conf = statistics.mean(window)
    thresh = drift_thresholds[category]
    if avg_conf < thresh:
        return {
            "category": category,
            "status": "DRIFT ALERT",
            "avg_confidence": round(avg_conf, 4),
            "threshold": thresh,
            "samples": len(window),
            "action": "tighten threshold or retrain model",
        }
    return {
        "category": category,
        "status": "healthy",
        "avg_confidence": round(avg_conf, 4),
        "threshold": thresh,
        "samples": len(window),
    }

simulated_traffic = [
    ("pricing question", "sales", 0.62),
    ("login error", "support", 0.71),
    ("just looking", "nurture", 0.58),
] * 17

for _, cat, conf in simulated_traffic:
    result = log_and_check_drift(cat, conf)

for cat in confidence_history:
    result = log_and_check_drift(cat, 0.40)
    if result:
        print(f"[{result['status']}] {result['category']}: avg_conf={result.get('avg_confidence', 'N/A')}, threshold={result.get('threshold', 'N/A')}, samples={result['samples']}")
```

Production classifiers degrade silently because the inputs do not announce that they have changed. A model trained on Q1 inbound traffic will encounter Q3 traffic that reflects a new product launch, a pricing change, or a shifted campaign strategy — the same words start appearing in different contexts, and the probability distribution drifts. The monitoring layer is the only signal you have that this is happening before a human notices misrouted content in a queue they check weekly.