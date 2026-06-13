# Lesson 25: Verification Gates and the Observation Budget

## GTM Redirect Rules

Primary GTM cluster: **Zone 3 — Agent Workflows** (specifically: enrichment waterfall quality gates, where each step in a Clay waterfall is a verification gate with an observation cost).

Secondary: **Zone 4 — Orchestration** (multi-agent handoff validation, where an observation budget prevents runaway enrichment loops).

---

## Learning Objectives

1. Implement a verification gate function that evaluates agent outputs against configurable criteria and returns a pass/halt/escalate decision.
2. Calculate and enforce an observation budget across a multi-step agent pipeline.
3. Compare full-revalidation, sampling-based, and heuristic-only verification strategies with measurable trade-offs.
4. Build a budget tracker that logs observations, tallies cost, and terminates execution when the budget is exhausted.
5. Diagnose verification failures by reading structured gate logs and identifying whether the failure is a budget exhaustion, a confidence shortfall, or a criteria mismatch.

---

## Beat 1: Hook

You've built agents that chain calls, enrich records, and score leads. But without verification gates, you're shipping outputs you've never checked — and without an observation budget, you're burning API credits on records that already failed. This lesson formalizes the two controls that keep autonomous workflows honest: a gate that decides *whether to proceed*, and a budget that decides *whether you can afford to check*.

---

## Beat 2: Concept

### Verification Gates

A verification gate is a decision function placed between agent steps. It takes three inputs: the output to verify, a set of acceptance criteria, and the current observation budget remaining. It returns one of three decisions: `PASS` (proceed to next step), `RETRY` (re-run current step with modified inputs), or `HALT` (stop the pipeline and log the failure). The gate itself consumes observation budget — each evaluation costs units, whether the evaluation is an LLM call, a rule check, or a human review.

### Observation Budget

The observation budget is a finite resource counter. Every verification action — running a validation prompt, checking a regex, sampling a population — debits the budget. When the budget reaches zero, the pipeline must either accept the current output unverified or halt. This models the real constraint: you cannot afford to verify everything, so you must decide where to spend your verification attention.

### The Tension

Stronger verification (full LLM-based re-evaluation) costs more budget per gate. Weaker verification (heuristic-only) costs less but misses more failures. The practitioner's job is to allocate the budget across gates so that the total failure rate across the pipeline stays below the threshold the business can tolerate.

---

## Beat 3: Code

Build a working verification gate system with an observation budget tracker. The code will:

1. Define a `VerificationGate` class that accepts criteria functions and returns structured decisions.
2. Define an `ObservationBudget` class that tracks remaining units and raises when exhausted.
3. Run a 3-step enrichment pipeline (extract company name → validate format → score confidence) where each step passes through a gate, and the budget is debited per evaluation.
4. Print structured logs showing: step number, gate decision, budget remaining, and output.

All code runs in terminal with observable print output. No external APIs — the gate logic uses deterministic checks (regex, length, character patterns) to model what would otherwise be LLM calls.

**Exercise hooks:**
- *Easy*: Modify the budget ceiling and observe how many records complete before exhaustion.
- *Medium*: Add a new gate that checks for common company name suffixes (Inc, LLC, Ltd) and log its pass/fail rate.
- *Hard*: Implement a `RETRY` path that re-runs a step with a modified prompt (modeled as a function parameter change) and track how many retries each record requires before pass or halt.

---

## Beat 4: Use It

**GTM application: enrichment waterfall quality gates.**

In a Clay waterfall (Zone 3), each enrichment provider is a step, and between steps you need to decide: is this data good enough to use, or should I try the next provider? That decision is a verification gate. The observation budget is your Clay credit budget — every enrichment call and every validation check costs credits.

The mapping:
- **Gate criteria** → field completeness checks, format validation, deduplication logic.
- **Observation budget** → your Clay credit allocation for this enrichment run.
- **HALT decision** → stop enriching this record and move to the next.
- **RETRY decision** → try the next provider in the waterfall for this field.

[CITATION NEEDED — concept: Clay waterfall credit costs per enrichment step and built-in fallback logic]

**Exercise hooks:**
- *Easy*: Map the 3-step pipeline from Beat 3 to a Clay waterfall with three providers (e.g., Clearbit → Hunter → Apollo) and label each gate with the GTM action it represents.
- *Medium*: Write a budget allocation strategy for a 10,000-record enrichment run where you have enough Clay credits for 3 verification checks per record on average — which steps get full verification and which get heuristic-only?
- *Hard*: Implement a priority queue that sorts records by estimated verification cost (based on data completeness) and processes high-value records first, halting low-value records earlier when budget runs low.

---

## Beat 5: Ship It

Production considerations for verification gates and observation budgets in a live GTM stack:

1. **Persistent budget state**: Store observation budget in a database or KV store so that a pipeline crash doesn't reset the counter. If you restart a 50,000-record enrichment run, you need to know how much budget you already spent.
2. **Gate logging schema**: Define a structured log format (JSON) that captures: timestamp, record ID, step name, gate decision, criteria results, budget remaining. This log is your audit trail and your debugging surface.
3. **Configurable thresholds per pipeline**: Different GTM workflows tolerate different failure rates. A high-value account list needs stricter gates and a larger budget per record than a broad outbound spray.
4. **Alerting on budget exhaustion**: If a pipeline halts because the observation budget hit zero before all records processed, that's a signal — either the budget was too low, or the data quality is worse than expected. Either way, the operator needs to know.

**Exercise hooks:**
- *Easy*: Add JSON-structured logging to the pipeline from Beat 3 and write the output to a file.
- *Medium*: Implement budget persistence using a SQLite table so that restarting the pipeline resumes from the last known budget state.
- *Hard*: Build a CLI tool that reads gate logs from a file, calculates failure rate per step, and recommends which gates to strengthen (full verification) vs. weaken (heuristic-only) to hit a target overall pipeline failure rate within a fixed budget.

---

## Beat 6: Evaluate

### Discussion Questions

1. A pipeline has 4 steps, each with a verification gate. The observation budget allows 2 full verifications per record. Which steps get full verification and which get heuristic-only? What information do you need to decide?
2. Your enrichment waterfall is halting on 30% of records due to budget exhaustion. You cannot increase the budget. What are three things you can change to reduce the exhaustion rate?
3. You notice that Gate 2 passes 98% of records but Gate 3 catches 40% of those as failures. What does this tell you about the relationship between Gate 2's criteria and Gate 3's criteria?

### Applied Prompt

You run a Clay waterfall that enriches 5,000 accounts per week. Each record goes through 4 enrichment steps. Your weekly Clay credit budget allows 15,000 verification checks total. Design a gate strategy that:
- Allocates verification budget across the 4 steps.
- Specifies which steps use full verification vs. heuristic-only.
- Defines the HALT condition for each gate.
- Estimates the expected failure rate at the end of the pipeline.

Submit your strategy as a structured document with a budget allocation table and a justification for each gate's verification level.

---

## Cross-References

- **Lesson 20** (Confidence Calibration): Verification gates consume confidence scores as input. This lesson assumes you can produce calibrated confidence estimates.
- **Lesson 22** (Fallback Chains): The Clay waterfall pattern is a specific implementation of a fallback chain. Verification gates decide *when* to trigger the fallback.
- **Lesson 24** (Cost-Aware Routing): Observation budgets are a specialization of cost-aware routing applied to the verification step itself.