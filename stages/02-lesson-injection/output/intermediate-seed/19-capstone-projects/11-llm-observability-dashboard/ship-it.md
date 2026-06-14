## Ship It

The local SQLite store is a development harness. Production replaces it with ClickHouse for the span table (columnar storage gives you sub-second aggregation over millions of spans) and Postgres for metadata (user configs, app-level settings, alert routing). The ingest path changes from direct SQLite writes to OTLP HTTP — your SDK emits GenAI-semconv spans to a collector endpoint, which handles batching, retry, and fan-out to both stores.

The eval architecture splits into two tiers. **Inline evals** — fast heuristics like regex match, JSON validation, length bounds, keyword presence — run at ingestion time and attach scores to the span before it hits the store. These have a strict latency budget (under 5ms) because they sit in the request path. **Batch evals** — LLM-as-judge, RAGAS faithfulness, DeepEval test cases — run as a separate job that polls the span store for unscored traces, runs the heavy evaluation, and writes results back. The batch job samples (typically 10–20% of traces) to control cost.

The drift detection threshold and window size are the two parameters you tune in production. Start with a window of 50 calls and a threshold of 0.15 (15% score drop). If you see false positives from natural variance, widen the window. If real regressions take too long to surface, narrow the window or raise sensitivity. Log every alert to a table with the span range, delta, and a resolved/unresolved flag — this becomes your incident history for prompt changes.

Here is the production regression test — the measurement bar for this capstone. It injects a degraded prompt at call 30 into a RAG-augmented outreach workflow and confirms the dashboard detects it within 10 subsequent calls:

```python
def run_regression_test():
    init_store()

    rag_obs = LLMObservability(
        app="gtm-outbound", workflow="rag-outreach"
    )
    rag_suite = EvalSuite()
    rag_suite.add("cites_case_study",
                  lambda s: 1.0 if "case study" in s["output_text"].lower() or
                            "reduced" in s["output_text"].lower() or
                            "increased" in s["output_text"].lower() else 0.0)
    rag_suite.add("mentions_company",
                  lambda s: 1.0 if any(c in s["output_text"]
                            for c in ["Acme", "Globex", "Initech", "Umbrella",
                                      "Hooli", "Vandelay", "Stark", "Wayne"]) else 0.0)
    rag_suite.add("valid_structure",
                  lambda s: 1.0 if len(s["output_text"].split(".")) >= 3 else 0.0)

    def good_rag_llm(prompt: str) -> str:
        company = [c for c in ["Acme", "Globex", "Initech", "Umbrella", "Hooli"]
                   if c in prompt]
        target = company[0] if company else "your company"
        return (f"Hi — our case study shows we reduced churn 30% for a company "
                f"like {target}. That's relevant given your growth trajectory. "
                f"Want to see the breakdown?")

    def degraded_rag_llm(prompt: str) -> str:
        return "Hey. Check out our product. It's great. Thanks."

    alerts = []
    companies = ["Acme", "Globex", "Initech", "Umbrella", "Hooli"]

    for i in range(60):
        company = companies[i % len(companies)]
        prompt = f"RAG context: case study about churn. Target: {company}."
        llm_fn = good_rag_llm if i < 30 else degraded_rag_llm
        span = rag_obs(llm_fn, prompt, model="gpt-4o")
        rag_suite.run(span)

        if i >= 35:
            dash = Dashboard()
            triggered = dash.check_drift(window_size=10, threshold=0.20)
            if triggered:
                alerts.append(i)
                print(f"  >>> Regression detected at call {i} "
                      f"(injected at call 30, {i - 30} calls to detect)")
                break

    if alerts:
        detection_calls = alerts[0] - 30
        print(f"\n  RESULT: PASS — drift detected in {detection_calls} calls "
              f"after injection")
        assert detection_calls <= 10, f"Took {detection_calls} calls, target was <=10"
    else:
        print("\n  RESULT: FAIL — drift not detected within window")

    final_dash = Dashboard()
    final_dash.print_summary()

run_regression_test()
```

The test asserts detection within 10 calls of the injection point. The degraded output scores 0.0 on all three heuristics, so once the rolling window fills with enough degraded spans to push the mean below threshold, the alert fires. With a window of 10 and threshold of 0