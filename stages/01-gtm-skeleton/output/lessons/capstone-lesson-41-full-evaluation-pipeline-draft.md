# Lesson 41: Full Evaluation Pipeline

## Hook

You've built individual eval components across the course — LLM-as-judge, heuristic checks, output schema validation. Now you wire them into a single pipeline that runs end-to-end, produces a scored report, and fails the build when quality drops below threshold.

## Concept

Define the evaluation pipeline as a DAG of evaluators that each produce a score in [0, 1], combined through a weighted aggregation into a single pass/fail decision. Cover three mechanisms: (1) evaluator independence — no evaluator sees another evaluator's output, (2) threshold calibration — setting per-evaluator cutoffs from a golden dataset, (3) regression detection — comparing current run scores against a rolling baseline. The pipeline is not a loop; it terminates once all evaluators have scored and the aggregator returns a decision.

## Demo

Build a working pipeline in a single Python file that: loads a test dataset of 20 inputs, runs three evaluators (exact-match heuristic, schema validator, LLM-as-judge), aggregates scores with configurable weights, prints a per-evaluator breakdown table, and exits with code 1 if any evaluator's mean score falls below its threshold. All output is terminal-only. Code runs unmodified in Claude Code Desktop.

## Use It

Map the pipeline to GTM enrichment validation. When you run a Clay waterfall that enriches company data through multiple providers, you need to verify output quality at scale — not spot-check by hand. The evaluation pipeline structure applies directly: each enrichment field gets an evaluator (format check for URLs, domain validation for emails, recency check for revenue figures), and the aggregator determines whether the enrichment run is clean enough to push to CRM. GTM cluster: **Zone 2 — Enrichment & Scoring**. [CITATION NEEDED — concept: enrichment quality gates in CRM push workflows]

## Ship It

**Easy:** Add a fourth evaluator to the demo pipeline that checks output length is within a configurable range, wire it into the aggregator, and confirm the exit code changes when threshold is violated.

**Medium:** Replace the hard-coded golden dataset with a JSONL file loaded from disk, add a `--calibrate` flag that computes threshold values from that file and writes them to a config file, then run the pipeline using the calibrated thresholds.

**Hard:** Implement regression detection — store each pipeline run's scores in a SQLite database, add a `--baseline` flag that computes the rolling mean of the last N runs, and fail the build if the current run's aggregate score drops more than 10% below the baseline. Print the baseline comparison table.

## Evaluate

Three quiz questions grounded in pipeline mechanics: (1) Why must evaluators not see each other's outputs — what failure mode does this prevent? (2) Given a pipeline with three evaluators scoring 0.9, 0.85, and 0.4 with weights 0.5, 0.3, 0.2 and a threshold of 0.7, does the run pass or fail? (3) What happens to regression detection if the rolling baseline window is set to 1? Each question maps to a specific objective in `docs/en.md`.