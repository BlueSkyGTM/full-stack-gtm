# Code Exec Metric

## Learning Objectives

- Build a subprocess-isolated code runner that captures wall-clock duration, exit status, and error type for every execution.
- Implement a metrics collector that aggregates success rate, latency percentiles, and error distributions across multiple runs.
- Score code-exec tasks by extracting fenced code blocks, running assertion harnesses, and computing pass-at-k across sampled generations.
- Detect enrichment pipeline degradation by applying execution metrics to GTM workflow steps and comparing against alert thresholds.
- Compare the four observable dimensions — duration, outcome, error class, throughput — and select which to instrument for a given pipeline stage.

## The Problem

You built a Clay waterfall with three enrichment steps. Two silently failed yesterday. You didn't find out until a rep complained about missing data at the pipeline review. The enrichment API returned 429s for six hours, your script swallowed the exceptions, and the table kept populating with empty cells. Nobody noticed because nobody was measuring.

This is the same problem in code evaluation. You generate 200 candidate functions from a model, half of them crash on edge cases, and your eval harness reports "model accuracy: 73%" without telling you *why* the other 27% failed. Was it a syntax error? A timeout? An infinite loop? A wrong import? Without structured execution metrics, you're reading stdout logs one by one — which is fine for ten samples and impossible for ten thousand.

Code execution metrics — runtime, success rate, error signatures — are the difference between catching degradation at 9:02 AM and discovering it in a pipeline review a week later. The mechanism is the same whether the code is a Clay enrichment function or a generated candidate in a benchmark: instrument once, aggregate many.

## The Concept

**Code execution metrics** are structured observations emitted by a running program: wall-clock duration, exit codes, exception types, and domain-specific counters like "API calls made" or "rows written." These metrics let you detect degradation, attribute failures,