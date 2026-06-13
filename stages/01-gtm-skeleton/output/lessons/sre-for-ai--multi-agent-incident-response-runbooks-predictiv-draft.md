# Lesson Outline: SRE for AI — Multi-Agent Incident Response, Runbooks, Predictive Detection

---

## Beat 1: Hook

A production AI agent cluster silently degrades — latency spikes on one agent cascade into timeout failures across a multi-step pipeline, and no single metric trips an alert. This is the incident class that traditional SRE doesn't catch: failure modes unique to compound AI systems where one agent's degraded output becomes another agent's poisoned input. Multi-agent incident response coordinates specialized diagnostic agents to isolate the root cause faster than a human paging through dashboards.

---

## Beat 2: Concept

Three mechanisms, each building on the last:

**Multi-Agent Incident Response.** Pattern: incident triggers a coordinator agent that dispatches specialized diagnostic agents (triage, root-cause-analysis, mitigation) in parallel, then synthesizes their outputs into a ranked action plan. The coordinator resolves conflicts between agents using a confidence-weighted voting mechanism — not a fixed priority hierarchy.

**Runbooks as Executable Graphs.** A runbook is a directed acyclic graph where each node is a diagnostic or remediation step, and edges represent conditional transitions based on observed state. AI-generated runbooks are assembled dynamically by matching incident signatures against a library of step templates, then validated against a simulator before execution. Static runbooks rot; generated runbooks adapt but require guardrails to prevent automated action on ambiguous signals.

**Predictive Detection via Telemetry Correlation.** Mechanism: stream telemetry (latency, token usage, error rates, semantic drift scores) into a sliding window, compute anomaly scores per signal, then run causal correlation across signals to surface leading indicators — the precursor pattern that appears minutes before a cascade. This is not forecasting a single metric; it's detecting the multi-variate signature that precedes compound failures.

---

## Beat 3: Guided Exercise

Build a minimal multi-agent incident responder. Given a simulated incident stream (provided as a working script), implement a coordinator that dispatches three diagnostic agents, collects their findings, and outputs a ranked remediation plan.

- **Easy:** Single-agent triage that classifies incident type from a structured alert payload.
- **Medium:** Three-agent system (triage, root-cause, mitigation) with a coordinator that merges outputs using confidence scores.
- **Hard:** Add predictive detection — ingest a telemetry time series, compute anomaly scores, and trigger the incident response pipeline pre-emptively (before the alert fires).

---

## Beat 4: Use It

**GTM Redirect: Zone 2 — AI Operations.** When you run multi-step AI workflows in production GTM (Clay waterfalls, agent-based research sequences, automated enrichment pipelines), a single agent failure silently corrupts downstream outputs — bad enrichment poisons personalization, which tanks reply rates, and the pipeline reports "success" because no step returned an error. Multi-agent incident response monitors the *semantic* output quality, not just HTTP status codes. The runbook graph maps directly to your Clay waterfall stages: each node is a Clay enrichment step, and conditional edges skip or retry based on output validation. Predictive detection on token-usage velocity and latency percentiles catches the "Clay API is about to throttle us" signal before it hits.

---

## Beat 5: Challenge

You receive a 48-hour telemetry dump from a production AI pipeline (provided as JSONL). Three separate incidents occurred. Build a system that: (1) detects all three incidents retrospectively, (2) generates a runbook for each, and (3) identifies the leading-indicator pattern that could have predicted at least two of them. Output a structured post-mortem with timestamps, root causes, and the predictive signal window.

- **Constraint:** No manual inspection of the raw data — your code must surface the findings.

---

## Beat 6: Ship It

Production incident-response loop: wire the predictive detector to a real telemetry source (even a synthetic one emitting at intervals), trigger the multi-agent responder on anomaly, generate an executable runbook, and log the full incident lifecycle to a file. The deliverable is a running process that catches, diagnoses, and documents an incident with zero human input.

- **Checkpoint:** Your system must produce a structured incident report (JSON) containing: detection timestamp, root cause, confidence score, remediation steps, and a post-incident telemetry summary.

---

## Learning Objectives (Draft)

1. **Implement** a coordinator-agent pattern that dispatches parallel diagnostic agents and synthesizes their outputs into a ranked action plan.
2. **Construct** an executable runbook as a DAG with conditional transitions, given a set of incident signatures and remediation steps.
3. **Build** a predictive detection pipeline that computes multi-variate anomaly scores from streaming telemetry and triggers incident response before alert thresholds are breached.
4. **Evaluate** whether an automated remediation is safe to execute by checking runbook guardrails (confidence thresholds, blast-radius limits, rollback paths).
5. **Generate** a structured post-mortem report from incident telemetry, diagnostic agent outputs, and runbook execution logs.

---

## Suggested `docs/en.md` Fragments (for quiz alignment)

- Coordinator conflict-resolution mechanism: confidence-weighted voting
- Runbook DAG validation: cycle detection, guardrail constraints
- Predictive detection: sliding-window anomaly scoring, causal correlation vs. simple threshold
- Safety: automated action criteria, human-in-the-loop triggers