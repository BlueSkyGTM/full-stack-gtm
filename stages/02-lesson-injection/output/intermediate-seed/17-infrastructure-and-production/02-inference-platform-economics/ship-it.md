## Ship It

The deliverable is a cost projection script that takes a real GTM workload and outputs a deployment recommendation. Below is the production version — it accepts a JSON workload profile, runs the comparison, and outputs a markdown report with a specific platform recommendation and the economic reasoning behind it.

```python
import json
import sys
from pathlib import Path

def generate_gtm_report(profiles: list[WorkloadProfile]) -> str:
    lines = ["# GTM Inference Cost Projection\n"]
    lines.append(f"Generated with pricing from {Path(__file__).name}\n")

    total_monthly = {}
    for wp in profiles:
        results = compare_all(wp)
        lines.append(f"## {wp.name}\n")
        lines.append(f"- QPS: {wp.qps} | In: {wp.avg_input_tokens} tok | Out: {wp.avg_output_tokens} tok")
        lines.append(f"- Schedule: {wp.hours_per_day}h/day × {wp.days_per_month} days/month")
        lines.append(f"- Latency budget: {wp.latency_budget_ms}ms | Cold start tolerance: {wp.cold_start_tolerance_sec}s\n")

        lines.append("| Platform | Billing | Monthly Cost | Cost/1K Req |")
        lines.append("|----------|---------|-------------|-------------|")
        for r in results:
            lines.append(
                f"| {r['platform']} | {r['billing_model']} | "
                f"${r['monthly_cost']:,.2f} | ${r['cost_per_1k_requests']:.4f} |"
            )

        cheapest = results[0]
        lines.append(f"\n**Recommended: {cheapest['platform']}** — ${cheapest['monthly_cost']:,.2f}/mo\n")

        for key in PRICING:
            total_monthly[key] = total_monthly.get(key, 0) + cheapest["monthly_cost"] if key == cheapest["platform"] else total_monthly.get(key, 0)

    lines.append("## Aggregate Recommendation\n")
    best_platform = min(total_monthly, key=total_monthly.get) if total_monthly else "none"
    lines.append(f"If using a single platform across all workloads, total spend on cheapest-per-workload mix: ${sum(total_monthly.values()):,.2f}/mo\n")

    lines.append("### Decision Rules Applied\n")
    lines.append("1. Batch enrichment (latency-tolerant, QPS > 1): per-token pricing favored")
    lines.append("2. Real-time scoring (latency < 1s): warm instance required, per-GPU-second if always-on")
    lines.append("3. Low-volume research (< 0.1 QPS): per-token with serverless, avoid dedicated GPU")
    lines.append("4. If workload mix includes both batch and real-time: split across two platforms\n")

    return "\n".join(lines)


gtm_workload_profiles = [
    WorkloadProfile(
        name="Batch Account Enrichment (Zone 2)",
        qps=3.0,
        avg_input_tokens=800,
        avg_output_tokens=200,
        hours_per_day=6,
        days_per_month=20,
        latency_budget_ms=5000,
        cold_start_tolerance_sec=30,
    ),
    WorkloadProfile(
        name="Real-Time Lead Scoring (Zone 3)",
        qps=0.3,
        avg_input_tokens=400,
        avg_output_tokens=80,
        hours_per_day=10,
        days_per_month=22,
        latency_budget_ms=800,
        cold_start_tolerance_sec=2,
    ),
    WorkloadProfile(
        name="Message Personalization (Zone 3)",
        qps=1.0,
        avg_input_tokens=600,
        avg_output_tokens=150,
        hours_per_day=8,
        days_per_month=22,
        latency_budget_ms=2000,
        cold_start_tolerance_sec=5,
    ),
]

report = generate_gtm_report(gtm_workload_profiles)
print(report)

output_path = Path("gtm_inference_cost_projection.md")
output_path.write_text(report)
print(f"\nReport saved to {output_path.resolve()}")
```

This script produces a markdown file you can hand to a finance team or attach to a deployment decision document. The report includes per-workload recommendations, aggregate spend across a mixed workload, and the decision rules applied so the reasoning is auditable. The pricing constants are clearly dated — update them quarterly, because inference pricing drops faster than any other infrastructure line item.

The skeptical defaults baked into the calculator: model load time is included for Baseten (3 minutes per cold deployment), Replicate's 60-second minimum per prediction is included, and cold-start sensitivity forces always-on billing when the workload's cold-start tolerance is below the platform's documented cold-start time. These are conservative — actual cold starts may be faster on warm pools — but they prevent underestimation, which is the failure mode that gets you an unexpected bill.