## Ship It

Build a cost-performance calculator that takes your API budget, per-token costs for two model tiers, and scaling-law-derived performance estimates, then outputs the break-even point where the higher-cost model justifies its spend. This is what you run before approving any AI GTM tooling budget — whether you are choosing between GPT-4o-mini and GPT-4o for enrichment waterfalls, or comparing a self-hosted fine-tune against an API model for personalized outbound.

The break-even computation works as follows. You have a fixed budget B. Each call costs `avg_tokens × cost_per_token`. Within budget, the cheaper model can make more calls. But the expensive model has higher per-call accuracy. The question is: what is the minimum value of a correct output that makes the expensive model's higher accuracy worth the lower call volume?

```python
import numpy as np

budget_usd = 5000.0
avg_tokens_per_call = 800

model_a = {
    "name": "GPT-4o-mini",
    "input_cost_per_1m": 0.15,
    "output_cost_per_1m": 0.60,
    "accuracy": 0.82,
    "avg_output_tokens": 200,
}

model_b = {
    "name": "GPT-4o",
    "input_cost_per_1m": 2.50,
    "output_cost_per_1m": 10.00,
    "accuracy": 0.91,
    "avg_output_tokens": 200,
}

def cost_per_call(spec, input_tokens):
    input_cost = (input_tokens / 1_000_000) * spec["input_cost_per_1m"]
    output_cost = (spec["avg_output_tokens"] / 1_000_000) * spec["output_cost_per_1m"]
    return input_cost + output_cost

cost_a = cost_per_call(model_a, avg_tokens_per_call)
cost_b = cost_per_call(model_b, avg_tokens_per_call)

calls_a = int(budget_usd / cost_a)
calls_b = int(budget_usd / cost_b)

correct_a = int(calls_a * model_a["accuracy"])
correct_b = int(calls_b * model_b["accuracy"])

break_even_value = (cost_b - cost_a) / (model_b["accuracy"] - model_a["accuracy"])

total_cost_for_target_a = 10000 * cost_a
total_cost_for_target_b = 10000 * cost_b

print("=" * 60)
print("COST-PER