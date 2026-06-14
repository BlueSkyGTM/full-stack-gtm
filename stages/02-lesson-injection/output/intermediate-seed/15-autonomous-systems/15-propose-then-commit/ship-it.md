## Ship It

Production deployment requires four components beyond the state machine itself. First, durable storage. The in-memory store in the build example is for learning. Production needs PostgreSQL or equivalent — LangGraph's checkpointing uses a Postgres table with the proposal ID, serialized state, and timestamp. If the review service restarts, pending proposals must survive. The idempotency key must be a database unique constraint, not just a set in memory.

Second, audit logging that satisfies compliance requirements. Every state transition is logged: who proposed, who reviewed, what was committed, what was rejected, and why. The log is append-only and timestamped. For GDPR-regulated data, the log itself may need retention limits — you cannot keep prospect data in audit logs indefinitely. The practical pattern is to log the proposal ID and metadata but redact PII from the audit entry after a retention window.

Third, batch review workflows. Reviewing proposals one at a time does not scale — a RevOps lead facing 200 pending tier upgrades needs a queue view with filtering, sorting by confidence, and bulk approve/reject. The `batch_review` method in the build example demonstrates the primitive. In production, this is a UI that groups proposals by type (all tier upgrades together, all industry reclassifications together), shows the blast radius prominently, and requires an explicit checklist confirmation before batch commit. The checklist is the anti-rubber-stamp mechanism: instead of one "Approve" button, the reviewer must acknowledge "I have reviewed 47 industry reclassifications and confirmed the sample audit" before the batch commits.

Fourth, monitoring and feedback loops. The approval rate is your primary signal. If 95% of proposals are approved, either your model is very good (raise the auto-commit threshold and remove humans from the loop for those cases) or your review process is theater (the reviewers are not actually reviewing). Track rejection reasons as structured data — not free text. "Headcount signal stale," "industry misclassified," "personalization tone too aggressive" are actionable categories that feed back into prompt engineering and data quality checks. If 40% of rejections cite stale data, the fix is upstream data freshness, not better review tooling.

Latency impact is real and must be measured. A review gate adds wall-clock time between enrichment and CRM update. For SLA-bound workflows (e.g., "new MQL must be enriched and routed within 5 minutes"), the review gate may be incompatible. The solution is tiered routing: high-confidence proposals auto-commit and meet the SLA; low-confidence proposals enter the review queue and miss the SLA but are flagged for follow-up. This is the confidence-thresholded auto-commit from the build example, applied to a production SLA constraint.

```python
import json
from datetime import datetime, timedelta

class ProductionMonitor:
    def __init__(self):
        self.rejections = []
        self.approval_latencies = []
        self.hourly_counts = {}

    def record_review(self, proposal, decision, review_latency_seconds):
        hour_key = datetime.now().strftime('%Y-%m-%d %H:00')
        if hour_key not in self.hourly_counts:
            self.hourly_counts[hour_key] = {'approved': 0, 'rejected': 0, 'auto_committed': 0}

        if proposal.status.value == 'committed' and proposal.reviewed_by is None:
            self.hourly_counts[hour_key]['auto_committed'] += 1
            return

        if decision == 'approve':
            self.hourly_counts[hour_key]['approved'] += 1
            self.approval_latencies.append(review_latency_seconds)
        elif decision == 'reject':
            self.hourly_counts[hour_key]['rejected'] += 1
            self.rejections.append({
                'proposal_id': proposal.id,
                'intent': proposal.intent,
                'reason': proposal.rejection_reason,
                'confidence': proposal.confidence,
                'timestamp': datetime.now().isoformat()
            })

    def rubber_stamp_check(self, min_reviews=20, rubber_stamp_threshold=0.90):
        total = sum(h['approved'] + h['rejected'] for h in self.hourly_counts.values())
        if total < min_reviews:
            print(f"RUBBER-STAMP CHECK: Insufficient data ({total} reviews, need {min_reviews}). Cannot assess.")
            return None

        approved = sum(h['approved'] for h in self.hourly_counts.values())
        rate = approved / total
        verdict = "RUBBER-STAMP LIKELY" if rate > rubber_stamp_threshold else "HEALTHY"
        print(f"RUBBER-STAMP CHECK: Approval rate {rate:.1%} over {total} reviews → {verdict}")
        if rate > rubber_stamp_threshold:
            print(f"  ACTION: Raise auto-commit threshold or tighten review checklist.")
        return rate

    def rejection_breakdown(self):
        if not self.rejections:
            print("REJECTION BREAKDOWN: No rejections recorded.")
            return {}

        reasons = {}
        for r in self.rejections:
            category = r['reason'].split(':')[0] if ':' in r['reason'] else r['reason']
            reasons[category] = reasons.get(category, 0) + 1

        total = len(self.rejections)
        print(f"\nREJECTION BREAKDOWN ({total} total):")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason:40s}: {count:4d} ({count/total:.0%})")
        top = max(reasons, key=reasons.get)
        print(f"  TOP REJECTION DRIVER: '{top}' → investigate upstream fix")
        return reasons

    def latency_report(self):
        if not self.approval_latencies:
            print("LATENCY: No manual review latencies recorded.")
            return
        latencies = sorted(self.approval_latencies)
        p50 = latencies[len(latencies)//2]
        p95 = latencies[int(len(latencies)*0.95)]
        avg = sum(latencies)/len(latencies)
        print(f"\nMANUAL REVIEW LATENCY:")
        print(f"  avg: {avg:.0f}s  p50: {p50:.0f}s  p95: {p95:.0f}s  count: {len(latencies)}")
        if p95 > 3600:
            print(f"  WARNING: p95 exceeds 1 hour. Review queue may be understaffed.")

    def full_report(self):
        print(f"\n{'='*60}")
        print(f"PRODUCTION HITL MONITORING REPORT")
        print(f"{'='*60}")
        for hour, counts in sorted(self.hourly_counts.items()):
            total = counts['approved'] + counts['rejected']
            auto = counts['auto_committed']
            print(f"  {hour}: manual={total} (approved={counts['approved']}, "
                  f"rejected={counts['rejected']}) auto_committed={auto}")
        self.rubber_stamp_check()
        self.rejection_breakdown()
        self.latency_report()
        print(f"{'='*60}\n")


monitor = ProductionMonitor()

class FakeProposal:
    def __init__(self, pid, intent, confidence, status, reviewed_by, rejection_reason=None):
        self.id = pid
        self.intent = intent
        self.confidence = confidence
        self.status = type('S', (), {'value': status})()
        self.reviewed_by = reviewed_by
        self.rejection_reason = rejection_reason

rejection_categories = [
    "stale_data:headcount_signal",
    "stale_data:funding_round",
    "misclassification:industry",
    "misclassification:industry",
    "tone:too_aggressive",
    "stale_data:headcount_signal",
    "wrong_segment:excluded_list",
    "misclassification:industry",
    "stale_data:headcount_signal",
    "tone:too_aggressive",
]

for i in range(40):
    p = FakeProposal(f"p{i:03d}", f"Update account {i}", 0.80 + i*0.001,
                     "committed", "reviewer@corp.com")
    monitor.record_review(p, "approve", review_latency_seconds=120 + i*5)

for j, reason in enumerate(rejection_categories):
    p = FakeProposal(f"r{j:03d}", f"Update account {j+40}",
                     0.70 + j*0.01, "rejected", "reviewer@corp.com",
                     rejection_reason=reason)
    monitor.record_review(p, "reject", review_latency_seconds=300 + j*10)

for k in range(15):
    p = FakeProposal(f"a{k:03d}", f"Auto-fill field {k}", 0.97,
                     "committed", None)
    monitor.record_review(p, "approve", review_latency_seconds=0)

monitor.full_report()
```

This monitoring code simulates 50 manual reviews (40 approved, 10 rejected) plus 15 auto-commits and produces a report showing the approval rate, rejection reason breakdown, and latency percentiles. The rubber-stamp check flags whether the approval rate is suspiciously high. The rejection breakdown identifies the top upstream fix — in this case, stale data signals, which means the enrichment provider data freshness is the problem, not the review process.