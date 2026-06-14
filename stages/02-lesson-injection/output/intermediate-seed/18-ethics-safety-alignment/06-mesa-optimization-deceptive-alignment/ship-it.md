## Ship It

Build a pre-deployment alignment audit for every autonomous agent in your GTM stack. This is a checklist you run before promoting any agent from sandbox to production. The items map directly to the conditions that produce deceptive alignment in the simulation: you are checking whether training performance is sufficient evidence of deployment performance, and the answer is always "not without additional testing."

The checklist has four components. First, document the training distribution: what data the agent was optimized on, what the reward signal was, and what the known limitations of that data are (e.g., "trained on Q3 2024 outbound data from SaaS companies in North America, reward signal was reply rate, does not include international accounts or enterprise sales cycles"). Second, document the deployment distribution: where the agent will actually run, and how it differs from training. Third, run a held-out evaluation set that structurally differs from training — different account types, different geographies, different tone expectations — and compare agent behavior against the training distribution. Fourth, run a behavioral consistency test: measure the same metric (reply rate, accuracy, qualification rate) under training-like and deployment-like conditions and flag any delta above a threshold you set.

Here is a monitoring configuration that flags production behavior inconsistent with held-out evaluation. It computes the distribution of agent outputs in production and compares it to the held-out evaluation distribution using a simple statistical distance. When the distance exceeds a threshold, it fires an alert — not because the agent is necessarily defective, but because the conditions under which you validated it no longer match the conditions under which it is operating.

```python
import random
from collections import Counter

random.seed(42)

EVAL_DISTRIBUTION = {
    "qualified_reply": 0.15,
    "neutral_reply": 0.10,
    "no_reply": 0.65,
    "negative_reply": 0.05,
    "unsubscribe": 0.05,
}

PRODUCTION_TRAFFIC = ["qualified_reply"] * 8 + ["neutral_reply"] * 6 + ["no_reply"] * 50 + ["negative_reply"] * 20 + ["unsubscribe"] * 16

THRESHOLD = 0.10

def kl_divergence(p, q, epsilon=1e-6):
    total = 0.0
    for key in p:
        p_val = p[key] + epsilon
        q_val = q.get(key, 0) + epsilon
        total += p_val * (p_val / q_val).__log__() if hasattr(p_val / q_val, '__log__') else p_val * math.log(p_val / q_val)
    return total

import math

def compute_kl(p, q, epsilon=1e-6):
    total = 0.0
    for key in p:
        p_val = p[key] + epsilon
        q_val = q.get(key, 0) + epsilon
        total += p_val * math.log(p_val / q_val)
    return total

production_sample = [random.choice(PRODUCTION_TRAFFIC) for _ in range(500)]
counts = Counter(production_sample)
total = sum(counts.values())
production_dist = {k: counts.get(k, 0) / total for k in EVAL_DISTRIBUTION}

kl = compute_kl(EVAL_DISTRIBUTION, production_dist)

print("=== ALIGNMENT MONITOR ===")
print(f"Evaluation distribution:    {EVAL_DISTRIBUTION}")
print(f"Production distribution:    {{ {', '.join(f'{k}: {v:.3f}' for k, v in sorted(production_dist.items()))} }}")
print(f"KL(evaluation || production): {kl:.4f}")
print(f"Threshold:                  {THRESHOLD}")
print()

if kl > THRESHOLD:
    print("ALERT: Production behavior diverges from held-out evaluation.")
    print("  Possible mesa-optimization: agent objective may differ from base objective.")
    print("  Investigate:")
    print("    1. Has the deployment distribution shifted from training?")
    print("    2. Is the agent conditioning on a signal that differs in production?")
    print("    3. Are negative_reply and unsubscribe rates elevated vs evaluation?")
    print("  Action: Pause agent, re-evaluate on held-out set, compare deltas.")
else:
    print("OK: Production behavior consistent with held-out evaluation.")
```

The KL divergence between your evaluation distribution and your production distribution is a crude but useful signal. It will not catch every form of misalignment — a deceptively aligned agent that conditions on subtle features of the deployment environment can maintain a matching distribution while still pursuing a different objective. But it catches the common GTM failure mode: the agent that was validated on one population and deployed on another, or the agent whose reward signal drifted from what you intended. The point is that you have a monitoring layer that does not trust training performance as evidence of deployment alignment.