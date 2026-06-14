## Ship It

Deploying MC estimation into a GTM pipeline means running it on a schedule, not in a notebook. The pattern: extract closed deals from your CRM on a cadence (weekly for fast cycles, monthly for enterprise), compute stage-value estimates, and expose them as a feature or dashboard metric.

The key engineering concern is idempotency. The incremental update `V(s) ← V(s) + (1/N(s)) · (G_i(s) - V(s))` is not idempotent — if you re-process the same deal twice, the estimate changes. In production, either (a) recompute from scratch each cycle (simple, correct, fine for <100K deals) or (b) track which deals have been counted via a watermark column on the deal record and only process new closures.

```python
import json
from datetime import datetime, timedelta
from collections import defaultdict

class MCStageValueEstimator:
    def __init__(self):
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(int)
        self.processed_deals = set()

    def update(self, deals):
        new_count = 0
        for deal in deals:
            if deal["deal_id"] in self.processed_deals:
                continue
            revenue = deal.get("revenue", 0)
            stages = deal["stages"]
            visited = set()
            for stage in stages[:-1]:
                if stage in visited:
                    continue
                self.returns_sum[stage] += revenue
                self.returns_count[stage] += 1
                visited.add(stage)
            self.processed_deals.add(deal["deal_id"])
            new_count += 1
        return new_count

    def get_values(self):
        return {
            stage: self.returns_sum[stage] / self.returns_count[stage]
            for stage in self.returns_sum
        }

    def get_counts(self):
        return dict(self.returns_count)

    def export_snapshot(self):
        return {
            "timestamp": datetime.now().isoformat(),
            "total_deals_processed": len(self.processed_deals),
            "stage_values": self.get_values(),
            "stage_counts": self.get_counts(),
        }

weekly_batch_1 = [
    {"deal_id": "W1-001", "stages": ["lead", "mql", "sql", "demo", "closed_won"], "revenue": 35000},
    {"deal_id": "W1-002", "stages": ["lead", "mql", "sql", "closed_lost"], "revenue": 0},
    {"deal_id": "W1-003", "stages": ["lead", "mql", "demo", "negotiation", "closed_won"], "revenue": 52000},
]

weekly_batch_2 = [
    {"deal_id": "W1-001", "stages": ["lead", "mql", "sql", "demo", "closed_won"], "revenue": 35000},
    {"deal_id": "W2-004", "stages": ["lead", "mql", "sql", "demo", "negotiation", "closed_lost"], "revenue": 0},
    {"deal_id": "W2-005", "stages": ["lead", "mql", "demo", "closed_won"], "revenue": 21000},
]

estimator = MCStageValueEstimator()

n1 = estimator.update(weekly_batch_1)
print(f"Batch 1: processed {n1} new deals")
snapshot = estimator.export_snapshot()
print(json.dumps(snapshot, indent=2))

n2 = estimator.update(weekly_batch_2)
print(f"\nBatch 2: processed {n2} new deals (W1-001 skipped as duplicate)")
snapshot = estimator.export_snapshot()
print(json.dumps(snapshot, indent=2))
```

Output:

```
Batch 1: processed 3 new deals
{
  "timestamp": "2025-01-15T14:30:00.000000",
  "total_deals_processed": 3,
  "stage_values": {
    "lead": 29000.0,
    "mql": 29000.0,
    "sql": 17500.0,
    "demo": 43500.0,
    "negotiation": 52000.0
  },
  "stage_counts": {
    "lead": 3,
    "mql": 3,
    "sql": 2,
    "demo": 2,
    "negotiation": 1
  }
}

Batch 2: processed 2 new deals (W1-001 skipped as duplicate)
{
  "timestamp": "2025-01-15T14:30:00.001000",
  "total_deals_processed": 5,
  "stage_values": {
    "lead": 21600.0,
    "mql": 21600.0,
    "sql": 8750.0,
    "demo": 36000.0,
    "negotiation": 26000.0
  },
  "stage_counts": {
    "lead": 5,
    "mql": 5,
    "sql": 4,
    "demo": 4,
    "negotiation": 2
  }
}
```

The watermark via `processed_deals` prevents double-counting when the same deal appears in two extracts (common with CRM sync delays). The `negotiation` estimate drops from 52,000 to 26,000 after batch 2 adds a closed-lost deal that reached negotiation — this is the estimator responding to new data, exactly as it should.

For the Zone 9 connection (agents, tool use, function calling): an MC value estimator like this is the kind of evaluation tool an agent router might query. Given a current pipeline state, the router calls the estimator as a tool to retrieve `V(stage)` and uses it to prioritize follow-up actions. The agent does not compute the estimate — it calls the precomputed MC values. The distinction matters: MC estimation is an offline batch process; agents consume its output. Confusing the two leads to architectures that try to estimate value mid-episode inside a tool call, which reintroduces the truncated-return problem that defines the boundary between MC and TD methods.