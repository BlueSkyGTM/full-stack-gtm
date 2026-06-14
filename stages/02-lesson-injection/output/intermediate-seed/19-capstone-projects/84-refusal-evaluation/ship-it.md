## Ship It

In production, the mock LLM gets replaced with a real API call. The evaluation harness does not change—the `evaluate` function accepts any callable that takes a prompt string and returns a response string. You swap `mock_llm` for a function that calls your model endpoint, and the metrics compute the same way.

The production deployment pattern is a nightly evaluation job that runs the full test suite against the current model configuration and writes the metrics to a dashboard. If over-refusal rate crosses a threshold—say 5% on benign prospecting prompts—the job alerts the team. This catches regressions from system prompt changes, model version updates, or RAG context modifications that shift the refusal boundary. In a Zone 19 RAG pipeline, adding new case studies or product docs to the retrieval index can change what context the model sees, which changes what it refuses. The nightly eval catches this before reps notice.

Here is the production harness wrapper that compares two models and reports the diff—the pattern you use when evaluating whether a model upgrade is safe to ship:

```python
def model_a(prompt: str) -> str:
    return mock_llm(prompt)


def model_b(prompt: str) -> str:
    lowered = prompt.lower()
    if "defense" in lowered or "pharmaceutical" in lowered or "cannabis" in lowered:
        return "Sure, here's a draft. Hi there, I wanted to reach out about..."
    if "hack" in lowered or "fake reviews" in lowered or "pretending" in lowered:
        return "I cannot assist with that request."
    return mock_llm(prompt)


suite = build_test_suite() + GTTM_TEST_SUITE if 'GTTM_TEST_SUITE' in dir() else build_test_suite()

results_a = evaluate(suite, model_a)
results_b = evaluate(suite, model_b)
metrics_a = compute_metrics(results_a)
metrics_b = compute_metrics(results_b)

print("MODEL COMPARISON: Refusal Behavior Diff")
print(f"{'Metric':<25} {'Model A':>10} {'Model B':>10} {'Delta':>10}")
print("-" * 55)
for key in ["over_refusal_rate", "under_refusal_rate", "refusal_precision", "refusal_recall", "refusal_f1"]:
    a = metrics_a[key]
    b = metrics_b[key]
    delta = b - a
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
    print(f"{key:<25} {a:>10.1%} {b:>10.1%} {delta:>+9.1%} {arrow}")

print(f"\nModel A over-refusals: {metrics_a['false_positives']}")
print(f"Model B over-refusals: {metrics_b['false_positives']}")
```

The diff output tells you exactly what changes when you swap models. If Model B drops the over-refusal rate from 40% to 0% but also drops under-refusal rate on policy-violating prompts, you have a tradeoff to evaluate. The numbers make the tradeoff visible instead of anecdotal.

For a Clay-specific application: Clay's waterfall enrichment passes prospect data through multiple data providers. If you add an AI step that drafts personalized copy from enriched data, that step inherits the model's refusal boundary. A prospect with a title like "Compliance Officer at a weapons manufacturer" might trigger a refusal that silently drops the personalization, and the rep sees a generic template instead. Running the refusal evaluation harness against your Clay-enriched prospect samples before activating the AI step prevents this. Tag your prospect test cases by industry and role, run them through the harness, and any over-refusal shows up as a false positive in the per-category breakdown.

---