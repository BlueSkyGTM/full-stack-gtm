## Ship It

Production routing has three concerns the prototype does not address: drift, confidence escalation, and cost accounting. The classifier's routing decisions shift over time as input distributions change. A router trained on Q1 inbound signals may misclassify Q3 signals because the product, the market, and the vocabulary have moved. The online quality gate is the monitoring loop that catches this.

The pattern is to sample a percentage of routed outputs, score them against a quality metric, and alert when the score crosses below a threshold. In a GTM context, the "quality metric" is downstream: did the lead convert, did they respond positively, did they book a meeting? Those signals arrive days or weeks later, so you also need a proxy metric — a strong model reviewing the routed response, or a human spot-check on 5% of routed conversations.

```python
import random

random.seed(42)

QUALITY_BY_TIER = {"simple": 0.97, "medium": 0.94, "complex": 0.96}
MISROUTE_QUALITY_DROP = 0.12

def simulate_drift(days=90, daily_queries=1000, initial_cheap_ratio=0.65, drift_per_day=0.004):
    log = []
    cheap_ratio = initial_cheap_ratio
    for day in range(days):
        n_cheap = int(daily_queries * cheap_ratio)
        n_rest = daily_queries - n_cheap
        quality_samples = []

        for _ in range(n_cheap):
            is_correct = random.random() < 0.88
            if is_correct:
                quality_samples.append(QUALITY_BY_TIER["simple"] + random.gauss(0, 0.01))
            else:
                dropped = QUALITY_BY_TIER["simple"] - MISROUTE_QUALITY_DROP
                quality_samples.append(dropped + random.gauss(0, 0.01))

        for _ in range(n_rest):
            tier = random.choice(["medium", "complex"])
            is_correct = random.random() < 0.92
            if is_correct:
                quality_samples.append(QUALITY_BY_TIER[tier] + random.gauss(0, 0.01))
            else:
                dropped = QUALITY_BY_TIER[tier] - MISROUTE_QUALITY_DROP
                quality_samples.append(dropped + random.gauss(0, 0.01))

        avg_quality = sum(quality_samples) / len(quality_samples)
        log.append({
            "day": day,
            "cheap_ratio": cheap_ratio,
            "quality": avg_quality,
            "alert": avg_quality < 0.92
        })
        cheap_ratio = min(0.95, cheap_ratio + drift_per_day)

    return log

results = simulate_drift()

print(f"{'DAY':<6} {'CHEAP %':<10} {'QUALITY':<10} {'STATUS':<10} {'COST @ 1M tok/day'}")
print("-" * 52)

checkpoints = [0, 7, 14, 21, 30, 45, 60,