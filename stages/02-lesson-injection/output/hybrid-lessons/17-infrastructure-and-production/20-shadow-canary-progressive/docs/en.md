# Shadow Traffic, Canary Rollout, and Progressive Deployment for LLMs

## Learning Objectives

- Implement a shadow traffic mirror that duplicates production requests and computes offline comparison metrics between model versions.
- Build a canary routing function with deterministic traffic splitting and guardrail-based automatic rollback.
- Simulate a progressive deployment state machine with staged traffic increments, dwell times, and promotion gates.
- Compare LLM-specific failure modes (semantic drift, cost explosion, non-deterministic variance) against traditional service deployment failures.
- Design a rollback path that executes in seconds via policy-flag configuration