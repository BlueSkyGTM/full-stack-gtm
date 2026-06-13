# METR Time Horizons and External Capability Evaluation

## Set It

Introduces the measurement problem: how long can an autonomous agent sustain coherent task execution before failing? Covers METR's time horizon methodology — the empirical measurement of task-completion probability as a function of autonomous work time — and explains why internal loss metrics don't predict real-world agent reliability. No GTM connection yet; this is foundational evaluation literacy.

## Map It

Defines the core mechanism: time horizon curves plot task-success rate against wall-clock autonomous execution time, producing a characteristic half-life. Distinguishes internal evaluation (loss, perplexity, benchmark accuracy) from external evaluation (task completion in real tool-using environments with real failure modes like context drift, tool misuse, goal drift). Maps the three failure modes that collapse time horizons: context window saturation, compounding tool errors, and goal drift across multi-step reasoning chains.

## Build It

Provides a working Python script that simulates time horizon measurement: generates synthetic task-completion data across varying time budgets, fits an exponential decay curve to produce a time-horizon half-life, and prints the calculated metric. No external APIs; pure computation with observable numeric output.

## Test It

Extends the build script to inject controlled failure modes (context saturation at N tokens, error-compounding cascades after M tool calls) and observe how each shifts the time horizon curve. Outputs a comparison table showing half-life under each degradation condition. Validates that the practitioner can distinguish which failure mode dominated a given decay pattern.

## Use It

**GTM Redirect: AI Agent Reliability in Outbound Sequences** — connects time horizon evaluation to measuring how long an AI agent can sustain coherent, personalized outreach before quality degrades. Maps to the GTM cluster for automated outbound and research workflows. The time horizon half-life predicts the maximum safe autonomous run length for multi-step prospecting agents before human review is required.

## Ship It

Provides a production-pattern checklist for instrumenting any GTM AI agent with time horizon logging: record task start time, each tool-call timestamp, and final success/failure state. Outputs a structured log format compatible with downstream half-life recalculation. Notes the open question: [CITATION NEEDED — concept: standardized METR time horizon benchmark datasets for GTM-specific tasks] and marks where practitioners should calibrate decay curves against their own historical agent execution data.

---

**Exercise Hooks:**

- **Easy:** Given a CSV of 50 task outcomes with start/end times and success flags, calculate the time horizon half-life.
- **Medium:** Modify the build script to accept real execution logs and output a formatted time horizon report with confidence intervals.
- **Hard:** Implement a monitoring function that halts an autonomous agent when its rolling success rate drops below the predicted time horizon threshold, simulating a production safety cutoff.