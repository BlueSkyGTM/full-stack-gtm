# Eval-Driven Agent Development

## Learning Objectives

1. Write eval suites that expose specific agent failure modes
2. Implement grading functions using exact match, LLM-as-judge, and structured extraction patterns
3. Run eval-driven iteration cycles and measure improvement between agent versions
4. Diagnose agent regressions using eval trace output
5. Build a CI-runnable eval suite that gates agent deployment

---

## Beat 1: Hook — The Silent Failure Problem

Agents don't crash — they produce plausible garbage. A lead research agent that returns slightly wrong revenue figures looks identical to one that's correct. By the time a human catches the pattern, the agent has already enriched 4,000 accounts. Eval-driven development is the discipline of defining "correct" before you write the agent, then using that definition as the build signal.

---

## Beat 2: Concept — The Eval-First Loop

The mechanism is a feedback loop: define test cases with expected outputs, run the agent against them, score the results, modify the agent, repeat. The eval suite is the specification. The agent is the implementation. You write the eval first because the act of writing test cases forces you to confront ambiguity in the task definition — if you can't write a grading function, the task isn't specified well enough to automate. This is test-driven development applied to non-deterministic systems, with the key difference that grading is probabilistic, not boolean.

---

## Beat 3: Mechanism — Grading Functions and Score Aggregation

Three grading primitives, each with different noise profiles:

**Exact match** — compares agent output to expected string. Zero noise, but only works for tightly constrained outputs (entity extraction, classification). Fragile: any formatting drift causes false failures.

**LLM-as-judge** — a second model evaluates whether the agent output meets criteria. Higher signal for open-ended tasks (summarization, research synthesis). Introduces judge noise: the grader itself is non-deterministic. Mitigate by running 3-5 judge passes and majority-voting.

**Structured extraction + assertion** — parse the agent output into a schema (JSON fields), then assert constraints on individual fields. Example: agent must return a JSON object with `revenue` as a number between 0 and 1 trillion. Catches format errors and value-range violations without needing a reference answer.

Score aggregation: run the full test suite, compute per-case scores (0 or 1 for exact match, 0–1 for judge), then produce a suite-level score. Track this score across agent versions. A version that drops below baseline fails the gate.

[CITATION NEEDED — concept: standard eval frameworks for agent development, beyond ad-hoc scripts]

---

## Beat 4: Code — Building and Running an Eval Suite

Working Python script that:
- Defines an agent function (simple: takes a company name, returns structured data about it)
- Defines a test suite as a list of `{input, expected_output}` cases
- Implements all three grading functions
- Runs the eval suite and prints per-case results and aggregate score
- Introduces a prompt change and re-runs, showing score delta

All output is terminal-printed. No browser. No API keys required for the skeleton (mock the LLM call if needed, or use a deterministic function as the "agent" to keep it runnable).

Exercise hook (easy): Add two new test cases that expose a specific failure mode in the mock agent.  
Exercise hook (medium): Swap the exact-match grader for an LLM-as-judge grader using Claude via the anthropic SDK. Run 3 passes and report variance.  
Exercise hook (hard): Implement a regression gate that compares two agent versions and exits non-zero if the new version scores below the old version on any individual case.

---

## Beat 5: Use It — Evaluating GTM Enrichment Agents

In GTM, enrichment agents research accounts and produce structured data: company size, tech stack, funding, buying signals. The Clay waterfall executes multiple enrichment steps sequentially — each step's output feeds the next. An eval suite here catches the specific failure mode where early-stage errors cascade through the waterfall.

The redirect: this is the evaluation discipline for **ICP Scoring & Enrichment** workflows. When you're running autonomous research across hundreds of accounts, you need a grading function that checks whether the enrichment output matches ground truth for a held-out test set of 20–50 known accounts. Build that eval suite before you run the waterfall at scale. Track the score every time you modify a Clay enrichment prompt. If the score drops, the prompt change doesn't ship.

For agents that draft outreach or personalize messaging, the LLM-as-judge pattern applies directly: define what "good personalization" means in a rubric, encode it as a judge prompt, and score every variant against the rubric.

If the agent task doesn't map to a specific GTM workflow (e.g., general-purpose reasoning), the redirect is: foundational for all agent-based GTM automation.

---

## Beat 6: Ship It — Production Eval Pipeline

Take-home deliverable: a complete eval suite for an agent the practitioner is actually building or maintaining. Requirements:

1. Minimum 15 test cases covering at least 2 distinct failure modes
2. At least 2 of the 3 grading functions implemented
3. A baseline score recorded for the current agent version
4. A script that runs the full suite and prints: per-case pass/fail, aggregate score, and any cases where the judge disagreed across runs
5. A CI stub (GitHub Actions YAML or Makefile target) that runs the eval and exits non-zero on regression

Exercise hook (hard): Instrument the eval runner to capture per-case latency. Add a regression gate for latency (not just accuracy) — flag any case that takes >2x the baseline median time.

---

## GTM Redirect Rules for This Lesson

- **Primary cluster**: ICP Scoring & Enrichment — eval suites for enrichment accuracy
- **Secondary cluster**: Agent Workflows — LLM-as-judge for outreach personalization quality
- **Foundational fallback**: eval-driven development is foundational for all GTM agent deployments; no fabricated specificity