## Ship It

When you move from a notebook to a production enrichment system, the TD framing gives you three concrete design decisions.

First, **what is your state representation?** In the gridworld, state is `(x, y)` — discrete and small. In a GTM enrichment pipeline, state is the combination of record attributes that affect provider success: company size, industry, geographic region, data freshness. If you treat every company as a distinct state, your Q-table is empty for most of them and you never learn. If you bucket companies into meaningful segments (e.g., "US-based SaaS, 10-50 employees, founded after 2020"), you get generalization — each provider's success rate is learned across similar companies. This is the same generalization problem that drives function approximation in deep RL, but at the pipeline level you solve it with segment-based bucketing rather than neural networks.

Second, **what is your reward signal?** In the gridworld, reward is `-1` per step and `+10` at the goal. In enrichment, reward is not just "did we get data" — it is the net value of the data minus the cost of acquiring it. A provider that returns data 80% of the time but costs $0.50 per call has a different expected value than one that returns data 90% of the time at $5.00 per call. Your reward function should be `value_of_usable_record - cost_of_enrichment`, and your Q-table updates should use this net reward. If you only track success rate without cost, you will converge on the most expensive provider.

Third, **how do you handle non-stationarity?** Provider match rates change over time. Clearbit's coverage of your ICP today is not the same as six months ago. A constant learning rate `α = 0.1` means your Q-table tracks recent performance and forgets old data — this is a feature, not a bug, in a non-stationary environment. If you decay `α` to zero (as pure convergence theory demands), your waterfall ordering freezes and becomes stale. In production, keep `α` bounded away from zero. This is the pragmatic deviation from Robbins-Monro that every production RL system makes.

Here is a pattern for a production-ready waterfall that learns from its own execution history:

```python
from collections import defaultdict
from datetime import datetime, timedelta

class LearningWaterfall:
    def __init__(self, providers, alpha=0.1, gamma=0.9, window_days=30):
        self.providers = providers
        self.alpha = alpha
        self.gamma = gamma
        self.window_days = window_days
        self.q_table = defaultdict(dict)
        self.history = []

    def _recent_outcomes(self, segment, provider):
        cutoff = datetime.now() - timedelta(days=self.window_days)
        relevant = [
            h for h in self.history
            if h["segment"] == segment
            and h["provider"] == provider
            and h["timestamp"] > cutoff
        ]
        if not relevant:
            return 0.5
        successes = sum(1 for h in relevant if h["success"])
        return successes / len(relevant)

    def order_for_segment(self, segment):
        scored = []
        for provider in self.providers:
            rate = self._recent_outcomes(segment, provider)
            cost = provider["cost_cents"] / 100
            q_value = rate * (provider["value_dollars"] - cost)
            scored.append((provider["name"], q_value, rate, cost))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def record_outcome(self, segment, provider_name, success, timestamp=None):
        self.history.append({
            "segment": segment,
            "provider": provider_name,
            "success": success,
            "timestamp": timestamp or datetime.now()
        })

providers = [
    {"name": "clearbit", "cost_cents": 2, "value_dollars": 5.0},
    {"name": "apollo", "cost_cents": 8, "value_dollars": 5.0},
    {"name": "manual", "cost_cents": 50, "value_dollars": 5.0},
]

wf = LearningWaterfall(providers)

wf.record_outcome("us_saas_seed", "clearbit", True)
wf.record_outcome("us_saas_seed", "clearbit", False)
wf.record_outcome("us_saas_seed", "clearbit", False)
wf.record_outcome("us_saas_seed", "apollo", True)
wf.record_outcome("us_saas_seed", "apollo", True)

wf.record_outcome("eu_enterprise", "clearbit", False)
wf.record_outcome("eu_enterprise", "clearbit", False)
wf.record_outcome("eu_enterprise", "apollo", False)
wf.record_outcome("eu_enterprise", "manual", True)

print("=== WATERFALL ORDER: us_saas_seed ===")
for name, q, rate, cost in wf.order_for_segment("us_saas_seed"):
    print(f"  {name:12s}  Q={q:.2f}  rate={rate:.0%}  cost=${cost:.2f}")

print("\n=== WATERFALL ORDER: eu_enterprise ===")
for name, q, rate, cost in wf.order_for_segment("eu_enterprise"):
    print(f"  {name:12s}  Q={q:.2f}  rate={rate:.0%}  cost=${cost:.2f}")

print("\nNote: different segments get different optimal orderings.")
print("This is segment-conditional policy — the enrichment analogue of state-conditional Q-values.")
```

The output shows that the US SaaS seed segment gets a different provider ordering than the EU enterprise segment — because the measured success rates differ. This is the enrichment analogue of a state-conditional Q-table: the "state" is the segment, the "actions" are the providers, and the "reward" is net value. The waterfall reorders itself as new outcomes are recorded, which is online TD learning applied to revenue infrastructure. The same loop — act, observe reward, update value estimate, reorder — is what your Q-learning gridworld does, just mapped to a different domain.

[CITATION NEEDED — concept: segment-based provider success rate measurement driving Clay waterfall reordering]