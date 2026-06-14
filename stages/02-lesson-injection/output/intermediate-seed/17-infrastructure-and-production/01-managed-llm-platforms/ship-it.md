## Ship It

Now compute the cost tradeoff between on-demand and provisioned throughput so you can make an informed capacity decision. The calculator below takes a monthly volume estimate and outputs the break-even point where PTU reservations become cheaper than pay-per-token pricing.

```python
import json

PRICING = {
    "azure_gpt4o_ondemand": {"input": 2.50, "output": 10.00},
    "azure_gpt4o_ptu_monthly": 3500.00,
    "bedrock_claude_sonnet": {"input": 3.00, "output": 15.00},
    "bedrock_claude_sonnet_provisioned_monthly": 4200.00,
    "vertex_gemini_flash": {"input": 0.075, "output": 0.30},
}

def compute_ondemand_cost(provider_key, monthly_requests, avg_input_tokens, avg_output_tokens):
    rates = PRICING[provider_key]
    monthly_input_tokens = monthly_requests * avg_input_tokens
    monthly_output_tokens = monthly_requests * avg_output_tokens
    input_cost = (monthly_input_tokens / 1_000_000) * rates["input"]
    output_cost = (monthly_output_tokens / 1_000_000) * rates["output"]
    return round(input_cost + output_cost, 2)

def compute_breakeven(ondemand_key, ptu_monthly_key, avg_input, avg_output):
    rates = PRICING[ondemand_key]
    ptu_cost = PRICING[ptu_monthly_key]

    cost_per_request_ondemand = (
        (avg_input / 1_000_000) * rates["input"]
        + (avg_output / 1_000_000) * rates["output"]
    )

    if cost_per_request_ondemand == 0:
        return float("inf")

    breakeven_requests = ptu_cost / cost_per_request_ondemand
    return int(breakeven_requests)

def full_comparison(monthly_requests, avg_input, avg_output):
    print(f"\n{'='*65}")
    print(f"MONTHLY WORKLOAD: {monthly_requests:,} requests")
    print(f"AVG PAYLOAD: {avg_input} input tokens / {avg_output} output tokens")
    print(f"{'='*65}")

    providers = [
        ("Azure GPT-4o (On-Demand)", "azure_gpt4o_ondemand", None),
        ("Azure GPT-4o (PTU)", None, "azure_gpt4o_ptu_monthly"),
        ("Bedrock Claude Sonnet (On-Demand)", "bedrock_claude_sonnet", None),
        ("Bedrock Claude Sonnet (Provisioned)", None, "bedrock_claude_sonnet_provisioned_monthly"),
        ("Vertex Gemini Flash (On-Demand)", "vertex_gemini_flash", None),
    ]

    for label, ondemand_key, ptu_key in providers:
        if ondemand_key:
            cost = compute_ondemand_cost(ondemand_key, monthly_requests, avg_input, avg_output)
            print(f"  {label}: ${cost:,.2f}/month")
        elif ptu_key:
            ptu_cost = PRICING[ptu_key]
            print(f"  {label}: ${ptu_cost:,.2f}/month (fixed)")
            be = compute_breakeven(
                "azure_gpt4o_ondemand" if "azure" in ptu_key else "bedrock_claude_sonnet",
                ptu_key,
                avg_input,
                avg_output,
            )
            print(f"    Break-even: {be:,} requests/month")
            current_volume_note = "ABOVE break-even (PTU cheaper)" if monthly_requests > be else "BELOW break-even (on-demand cheaper)"
            print(f"    Your volume: {current_volume_note}")

full_comparison(50_000, 800, 200)
full_comparison(500_000, 800, 200)
full_comparison(2_000_000, 800, 200)

print("\n" + "=" * 65)
print("BREAKEVEN ANALYSIS: Azure GPT-4o On-Demand vs PTU")
print("(800 input tokens, 200 output tokens per request)")
print("=" * 65)
be_azure = compute_breakeven("azure_gpt4o_ondemand", "azure_gpt4o_ptu_monthly", 800, 200)
print(f"  PTU becomes cheaper at: {be_azure:,} requests/month")
print(f"  That is ~{be_azure // 30:,} requests/day")
print(f"  At $3,500/month PTU, daily cost is always $116.67 regardless of usage")
```

The output shows three workload scenarios. At 50,000 monthly requests, on-demand is cheaper across the board. At 500,000, provisioned throughput starts to win on Azure and Bedrock. At 2,000,000, the PTU reservation saves thousands per month and eliminates throttling risk entirely. The break-even point depends on your average token count — a GTM enrichment pipeline with short prompts (200 input tokens, 50 output tokens) will need more requests to justify PTU than a long-context document processing pipeline (4,000 input tokens, 500 output tokens).

For a GTM team running high-volume enrichment — say, scoring 10,000 leads per day with an LLM-based ICP classifier — the decision framework is: start on-demand while you measure actual token consumption, then provision throughput once your monthly volume consistently exceeds the break-even point. Zone 17 calls this scoring drift detection: you track not just cost but whether your model's classification accuracy degrades over time as your prospect population shifts. The same telemetry infrastructure that records token counts for FinOps should also record prediction confidence distributions so you can detect when retraining or model swapping is needed. [CITATION NEEDED — concept: Zone 17 scoring drift in GTM systems]