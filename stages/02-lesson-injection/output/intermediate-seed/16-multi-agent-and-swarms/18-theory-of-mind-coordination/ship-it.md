## Ship It

The production deployment of ToM-aware GTM is the two-prompt belief pipeline above, wired into your enrichment and outreach tooling. The first prompt models the buyer's belief state from firmographic and technographic signals. The second prompt generates outreach calibrated to that belief state. The output is a belief-conditioned email that adjusts language based on whether the prospect has category awareness.

The critical production question is whether the belief model is accurate. A false-belief pipeline that *misclassifies* the buyer's awareness is worse than no pipeline — it either dumbs down outreach to a sophisticated buyer or uses jargon with a naive one. Validate the belief model against historical reply data: prospects tagged "awareness: none" who received jargon-free copy should reply at higher rates than those who received jargon. If the delta is zero or negative, the belief model is not tracking reality.

```python
import json

VALIDATION_FRAMEWORK = {
    "metric": "reply_rate_by_belief_segment",
    "segments": ["awareness_none", "awareness_vague", "awareness_active"],
    "hypothesis": "Prospects tagged 'awareness_none' who receive jargon-free copy reply at >= 1.5x the rate of those who receive category jargon.",
    "minimum_sample": 200,
    "test_window_days": 14,
    "failure_action": "If delta < 1.2x, the belief model is not calibrated. Re-examine signals used for awareness classification.",
}

PRODUCTION_CHECKLIST = {
    "belief_model_validation": "Reply rate delta >= 1.5x between calibrated and non-calibrated outreach",
    "second_order_check": "For enterprise deals, model the buyer's perception of their boss's priorities — if the LLM can't pass second-order ToM, don't rely on it for multi-threaded strategy",
    "coordination_protocol": "For multi-agent enrichment chains, use a deterministic waterfall (Clay's model) instead of agent negotiation unless ToM prompting is explicitly included",
    "fallback": "If the belief model returns 'unknown' for awareness, default to jargon-free copy — false negative is cheaper than false positive",
    "monitoring": [
        "Track belief_model distribution weekly — if 90%+ of prospects are classified as 'active_research', the model is collapsed",
        "Log outreach variants alongside belief segments for A/B analysis",
        "Flag prospects where belief model disagrees with sales team's manual assessment",
    ],
}

def generate_production_config():
    print("=" * 60)
    print("PRODUCTION CONFIG: ToM-Aware Outreach Pipeline")
    print("=" * 60)
    
    print("\n1. BELIEF MODEL")
    print(f"   Metric: {VALIDATION_FRAMEWORK['metric']}")
    print(f"   Min sample: {VALIDATION_FRAMEWORK['minimum_sample']}")
    print(f"   Window: {VALIDATION_FRAMEWORK['test_window_days']} days")
    print(f"   Threshold: {VALIDATION_FRAMEWORK['hypothesis']}")
    print(f"   On failure: {VALIDATION_FRAMEWORK['failure_action']}")
    
    print("\n2. PRODUCTION CHECKLIST")
    for key, val in PRODUCTION_CHECKLIST.items():
        if isinstance(val, list):
            print(f"   {key}:")
            for item in val:
                print(f"     - {item}")
        else:
            print(f"   {key}: {val}")
    
    print("\n3. DEPLOYMENT DECISION TREE")
    print("   IF belief model validated (reply delta >= 1.5x):")
    print("     → Deploy ToM-calibrated outreach to full pipeline")
    print("   IF belief model not validated (delta < 1.2x):")
    print("     → Fall back to static segmentation (firmographic only)")
    print("     → Do NOT use LLM-generated belief states for personalization")
    print("   IF second-order ToM accuracy < 60%:")
    print("     → Restrict to first-order personalization only")
    print("     → Do NOT attempt multi-threaded / VP-aware messaging")
    print()
    
    config = {
        "validation": VALIDATION_FRAMEWORK,
        "checklist": PRODUCTION_CHECKLIST,
    }
    print("4. CONFIG OUTPUT (for pipeline ingestion):")
    print(json.dumps(config, indent=2))

if __name__ == "__main__":
    generate_production_config()
```

For multi-agent enrichment specifically, the production pattern is: use a deterministic waterfall, not agent negotiation. The waterfall is a protocol that replaces ToM-dependent coordination with a fixed sequence. This is Zone 16 (distributed systems) applied to GTM: your enrichment chain is a distributed system with parallel requests, rate-limit backpressure, and idempotent retries. The "coordination" is the waterfall order, not agents reasoning about each other. If you must use multi-agent negotiation (e.g., for complex account research where tasks are interdependent), include ToM prompting explicitly — and measure whether coordination degrades when the prompt is removed, per the Riedl protocol.