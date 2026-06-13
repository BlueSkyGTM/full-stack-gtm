# Lesson 27: Eval Harness with Fixture Tasks

## Hook

You've built agents, chained prompts, and wired enrichments. Now you need to know if they actually work—repeatedly, mechanically, without manual inspection. An eval harness automates judgment against fixture tasks: fixed inputs with known-correct outputs. Without this, you're eyeballing quality and hoping for the best.

---

## Concept

**Fixture-based evaluation** is the same pattern as unit testing, applied to AI outputs. A fixture is a frozen input-output pair. The harness runs your pipeline against every fixture, compares actual output to expected output using a scoring function, and aggregates results. The scoring function can be exact match, substring match, JSON schema validation, or an LLM-as-judge rubric. The mechanism is: load fixtures → run pipeline → score → report. Anything else is ad-hoc.

Key terms:
- **Fixture task**: a `(input, expected_output)` pair, stored as JSON or YAML
- **Harness**: the orchestrator that iterates fixtures, calls the pipeline, applies the scorer
- **Scorer**: a function `(expected, actual) → float` between 0.0 and 1.0
- **Aggregation**: mean, median, or pass@k across all fixture scores

---

## Build

Implement a minimal eval harness from scratch. Define fixture tasks as a JSON list. Build a harness loop that loads fixtures, runs a callable pipeline against each, applies a configurable scorer, and prints a summary table. No framework dependency—just functions and loops.

**Exercise (Easy):** Write three fixture tasks for a company-name-to-domain pipeline and score with exact match.

**Exercise (Medium):** Implement a `contains_key_scorer` that checks whether a specific key exists in JSON output and returns 1.0 or 0.0.

**Exercise (Hard):** Build a harness that accepts multiple scorers per fixture and reports a per-scorer breakdown table with pass rates.

---

## Use It

In GTM, every enrichment pipeline—company lookup, lead scoring, ICP classification—needs regression-proofing. Ship a new prompt version, run the harness against your fixture set, and see if scores hold or drop. This is the evaluation backbone for Zone 2 enrichment and Zone 3 routing workflows. If you're using Clay's waterfall enrichment, your fixture tasks are the known test companies and their expected enrichment results.

**GTM Redirect:** Zone 2 enrichment quality assurance. Fixture tasks represent known companies, contacts, or accounts with verified enrichment data. The harness catches regressions when prompts change or providers swap.

---

## Ship It

Store fixtures in version control alongside your pipeline code. Run the harness in CI on every PR that touches prompt templates or parsing logic. Fail the build if aggregate score drops below a threshold you set. Log per-fixture results to a file or database so you can track regressions over time. This is not optional infrastructure if you're running GTM AI in production.

**Exercise (Easy):** Export harness results to a CSV file with columns: fixture_id, score, passed, timestamp.

**Exercise (Medium):** Add a CLI argument for score threshold and exit with code 1 if the aggregate score falls below it—ready for CI.

**Exercise (Hard):** Implement fixture versioning. When a fixture's expected output changes, log it as an intentional override with a reason string, not a silent update.

---

## Review

You built an eval harness: a mechanical loop that runs fixed inputs through your pipeline and scores the outputs against known answers. This is how you stop guessing and start measuring. Fixtures are your ground truth. Scorers are your tolerance for variance. Aggregation is your dashboard. Everything else in AI quality assurance builds on this pattern.

**Next lesson:** You'll extend this harness with LLM-as-judge scoring for outputs that can't be checked with exact match—open-ended classifications, email drafts, research summaries. [CITATION NEEDED — concept: LLM-as-judge scorer integration with fixture harness]

---

## Learning Objectives

1. Implement a fixture-based eval harness that loads `(input, expected)` pairs from JSON and runs them through a callable pipeline.
2. Write configurable scorer functions that return normalized scores between 0.0 and 1.0.
3. Aggregate per-fixture scores into a pass rate and identify failing fixtures by ID.
4. Export harness results to CSV for version-controlled regression tracking.
5. Configure a CI-ready harness that exits with a failure code when aggregate score drops below a defined threshold.