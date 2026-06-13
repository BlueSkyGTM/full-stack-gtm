# End-to-End Eval Runner

## Hook It
A single LLM call failure in production is a bug. A systematic drift in response quality across thousands of calls is a business failure. The eval runner is the system that catches the second one before revenue notices.

## Ground It
Define the eval runner pattern: a harness that feeds inputs to a system-under-test, collects outputs, scores them against rubrics, and aggregates results into actionable metrics. Distinguish from unit tests (deterministic assertions) and from benchmarks (static datasets). The eval runner is a pipeline, not a test suite.

## Show It
Build a minimal eval runner in Python that: loads a dataset of input-output pairs, runs each through a scorer function (exact match, contains, LLM-as-judge), and prints a summary table with pass rates per scorer. All code runs in terminal, outputs a formatted results table.

## Build It
Extend the minimal runner into a configurable pipeline: add YAML-based eval configuration (model, dataset path, scorers), implement three scorer types (string match, JSON schema validation, LLM-grade with rubric), and produce a JSON report file. Exercise hooks: easy — add a new string-match scorer; medium — implement JSON schema validation scorer; hard — wire an LLM-as-judge scorer that grades on a 1-5 rubric and thresholds at 3.

## Use It
GTM redirect: Clay's AI field enrichment generates personalized data points at scale. An eval runner validates that enrichment quality holds across ICP segments before a campaign launches. Run evals against a holdout set of accounts with known-correct enrichments to catch regression when prompts change. [CITATION NEEDED — concept: Clay AI field enrichment eval methodology]

## Ship It
Wire the eval runner into CI: run on every prompt change, fail the pipeline if any scorer drops below threshold, and archive reports with git SHA tags. The eval runner becomes a gate, not a report you read once. Exercise hooks: easy — write a shell script that runs the eval runner and exits non-zero on failure; medium — generate a markdown report from the JSON output and commit it; hard — build a simple trending view that compares current run metrics against the last 5 runs.