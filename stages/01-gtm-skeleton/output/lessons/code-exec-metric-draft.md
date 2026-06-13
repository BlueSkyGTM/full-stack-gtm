# Code Exec Metric

## Hook

You built a Clay waterfall with three enrichment steps. Two silently failed yesterday. You didn't find out until a rep complained about missing data. Code execution metrics — runtime, success rate, error signatures — are the difference between catching that at 9:02 AM and discovering it in a pipeline review a week later.

## Concept

**Code execution metrics** are structured observations emitted by a running program: wall-clock duration, exit codes, exception types, and domain-specific counters (e.g., "API calls made," "rows written"). These metrics let you detect degradation, attribute failures, and set alerts — without reading logs line by line.

The key distinction: **logs** tell you *what happened* (event stream); **metrics** tell you *how often and how fast* (aggregated signal). Both come from the same runtime, but metrics are pre-aggregated into a shape that supports comparison over time windows.

GTM cluster: **Zone 02 — Signal Capture & Enrichment**. Every enrichment step in a Clay table, every webhook handler, every bulk-processing script is code that *executes*. If you can't measure it, you can't trust the signal it produces.

## Mechanism

### What gets measured

A code execution runtime exposes (or can be instrumented to emit) four observable dimensions:

| Dimension | Example | Type |
|---|---|---|
| Duration | `elapsed_ms` | histogram |
| Outcome | `exit_code`, `success` | counter (0 or 1) |
| Error class | `error_type = "TimeoutError"` | counter by label |
| Throughput | `rows_processed` | counter |

### How metrics are emitted

The algorithm:

1. **Start a timer** at function entry.
2. **Wrap the function body** in a `try/except` (or equivalent).
3. **On success**: record `elapsed`, increment `success` counter.
4. **On failure**: record `elapsed`, increment `error` counter with the exception class as a label.
5. **In all cases**: record domain-specific counters (rows, API calls, bytes).
6. **Flush** metrics to a collector (stdout, file, or metrics endpoint) at the end of the run.

This is the same pattern used by OpenTelemetry, Prometheus client libraries, and every observability SDK. The mechanism is: *instrument once, aggregate many*.

### What can go wrong

- **High-cardinality labels** (e.g., tagging every `user_id`) explode storage. Restrict labels to bounded sets (error types, endpoint names, table names).
- **Unhandled exceptions** bypass your instrumentation. A top-level handler that always emits a metric is mandatory.
- **Clock skew** in distributed systems means `elapsed_ms` is only trustworthy within a single process.

### Observable output

Running this script prints a structured metric snapshot to stdout — no external collector required.

```python
import time
import random
from dataclasses import dataclass, field
from collections import Counter

@dataclass
class CodeExecMetrics:
    function_name: str
    calls: int = 0
    successes: int = 0
    failures: int = 0
    error_types: Counter = field(default_factory=Counter)
    total_elapsed_ms: float = 0.0
    domain_counters: dict = field(default_factory=dict)

    def record_success(self, elapsed_ms: float, extra: dict = None):
        self.calls += 1
        self.successes += 1
        self.total_elapsed_ms += elapsed_ms
        if extra:
            for k, v in extra.items():
                self.domain_counters[k] = self.domain_counters.get(k, 0) + v

    def record_failure(self, elapsed_ms: float, error_type: str, extra: dict = None):
        self.calls += 1
        self.failures += 1
        self.error_types[error_type] += 1
        self.total_elapsed_ms += elapsed_ms
        if extra:
            for k, v in extra.items():
                self.domain_counters[k] = self.domain_counters.get(k, 0) + v

    def snapshot(self) -> dict:
        avg_ms = self.total_elapsed_ms / self.calls if self.calls else 0
        success_rate = self.successes / self.calls if self.calls else 0
        return {
            "function": self.function_name,
            "calls": self.calls,
            "success_rate": round(success_rate, 4),
            "failures": self.failures,
            "error_types": dict(self.error_types),
            "avg_elapsed_ms": round(avg_ms, 2),
            "total_elapsed_ms": round(self.total_elapsed_ms, 2),
            "domain_counters": self.domain_counters,
        }


def enrich_company(domain: str) -> dict:
    start = time.monotonic()
    metrics = enrich_company.metrics
    roll = random.random()
    try:
        time.sleep(random.uniform(0.01, 0.05))
        if roll < 0.2:
            raise TimeoutError("API timeout fetching company data")
        if roll < 0.3:
            raise ValueError("Invalid domain format")
        elapsed = (time.monotonic() - start) * 1000
        metrics.record_success(elapsed, extra={"companies_enriched": 1})
        return {"domain": domain, "name": domain.split(".")[0].title(), "employees": random.randint(10, 5000)}
    except (TimeoutError, ValueError) as e:
        elapsed = (time.monotonic() - start) * 1000
        metrics.record_failure(elapsed, type(e).__name__)
        return {"domain": domain, "error": str(e)}


enrich_company.metrics = CodeExecMetrics("enrich_company")

domains = ["acme.com", "globex.io", "initech.com", "wayne.ent", "stark.ind", "umbrella.corp", "cyberdyne.sys", "oscorp.com", "wonka.com", "tyrell.corp"]

results = [enrich_company(d) for d in domains]

import json
print(json.dumps(enrich_company.metrics.snapshot(), indent=2))
```

Typical output (will vary per run due to randomness):

```
{
  "function": "enrich_company",
  "calls": 10,
  "success_rate": 0.7,
  "failures": 3,
  "error_types": {"TimeoutError": 2, "ValueError": 1},
  "avg_elapsed_ms": 28.41,
  "total_elapsed_ms": 284.12,
  "domain_counters": {"companies_enriched": 7}
}
```

The snapshot tells you: 70% success rate, `TimeoutError` dominates failures, average latency ~28ms. That is a *metric* — you can compare it to yesterday's run and detect regression.

## Use It

**Zone 02 — Signal Capture & Enrichment.** Every Clay enrichment column, every webhook handler that writes to a table, and every bulk-processing script in your GTM stack is a code execution point. If `enrich_company` drops from 90% to 60% success rate, your ICP scoring is now operating on incomplete data. Code exec metrics make that degradation visible *before* a rep asks why their territory has no accounts.

Concrete applications:

1. **Clay table scripts**: Wrap each Clay JavaScript formula in a timing + error counter. Export via Clay's HTTP API or log to a Clay table column for dashboarding.
2. **Webhook receivers**: Instrument every `/webhook` handler. Track acceptance rate, processing time, and payload size. A spike in `422` errors means upstream changed their schema.
3. **Bulk enrichment jobs**: When running a CSV of 5,000 domains through an enrichment API, track per-row success/failure and cumulative throughput. Stop the job if error rate exceeds a threshold.

**Exercise hooks:**

- **Easy**: Add a `timeout_count` domain counter to the existing metric class. Print it in the snapshot. Run the script and confirm the count matches `error_types["TimeoutError"]`.
- **Medium**: Refactor the decorator pattern. Write a `@track_exec` decorator that attaches a `CodeExecMetrics` instance to any function automatically — no manual `try/except` required. Test it on two different functions and print both snapshots.
- **Hard**: Implement a `compare()` method on `CodeExecMetrics` that takes a second snapshot dict and returns a diff (e.g., `success_rate_delta: -0.15`, `avg_elapsed_ms_delta: +12.3`). Simulate two runs and print the diff to detect regression.

## Ship It

**Deliverable**: A standalone `metrics.py` module that you import into any GTM script — Clay formulas, webhook handlers, enrichment jobs. It must:

1. Provide a `CodeExecMetrics` class with `record_success`, `record_failure`, and `snapshot`.
2. Provide a `@track_exec` decorator that instruments any function automatically.
3. Print a JSON snapshot to stdout when the script exits (via `atexit` or explicit call).
4. Support a `--metrics-threshold` CLI flag that exits non-zero if success_rate drops below the given value — usable in CI or cron monitoring.

**GTM redirect**: This is the instrumentation layer for every enrichment pipeline in **Zone 02**. Ship it, import it, and every script in your GTM stack becomes observable without adding print statements.

## Evaluate

**Objectives** (each testable):

1. **Implement** a `CodeExecMetrics` class that tracks call count, success count, failure count, error types, and elapsed time.
2. **Apply** the `@track_exec` decorator to instrument an arbitrary function's execution without modifying its internals.
3. **Detect** degradation by comparing two metric snapshots and identifying a drop in success rate or spike in average elapsed time.
4. **Configure** a threshold alert that exits non-zero when success rate falls below a configurable bound.
5. **Explain** why high-cardinality labels (e.g., per-user-id) make metric storage unbounded and how to restrict labels to bounded sets.

**Assessment alignment**: Each objective maps to a quiz question in the corresponding `docs/en.md`. No question will use generic ML trivia — all questions reference the `CodeExecMetrics` class, the decorator pattern, or the GTM enrichment scenario described in this lesson.

---

*CITATION CHECK: The concept of code execution metrics as described here (timer at entry, try/except wrapper, labeled counters, histogram for duration) is standard software engineering practice and does not require a specific external citation. The mapping to Clay enrichment workflows is based on the Clay platform's JavaScript formula execution model.*