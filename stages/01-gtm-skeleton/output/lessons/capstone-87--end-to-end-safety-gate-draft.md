# Capstone 87 — End-to-End Safety Gate

## Hook

You've built individual safety checks in isolation. Now you'll chain them into a single gate that evaluates every LLM call before it reaches a human or a customer. One input enters. It either passes with a signed clearance or gets blocked with a logged reason. No partial states.

## Concept

The safety gate pattern: a sequential pipeline where each stage is a binary pass/fail classifier. Input validation → intent classification → output scanning → policy enforcement. Each stage short-circuits on failure. The gate emits a structured result: `{passed: bool, stage: string, reason: string | null, payload: any}`. You'll implement this as a composable middleware chain, not a monolithic function.

## Demo

Build the gate as a Python module with four stages. Each stage is a callable that receives a dict and returns a verdict dict. The orchestrator runs them in sequence, stops on first failure, and writes a JSONL audit log. Demonstrates with three test inputs: one clean pass, one prompt injection blocked at stage 2, one output toxicity blocked at stage 3. All observable via printed JSONL lines and final verdict.

## Use It

**GTM cluster: Automated Outbound Safety** — When Clay generates personalized outreach at scale, a safety gate between generation and send prevents brand-damaging outputs. The same pattern applies here: prompt in, LLM output generated, gate checks for competitor mentions, PII leakage, and tone policy before the email enters the send queue. Build the gate once, deploy it as a checkpoint in any Clay waterfall that produces customer-facing text.

## Ship It

**Easy:** Add a fifth stage that checks output length against a configurable max.  
**Medium:** Make each stage's pass/fail threshold configurable via a YAML file, and reload without restart.  
**Hard:** Add a Prometheus counter for each stage's pass/fail rates, expose on `:9090/metrics`, and write a curl-based smoke test that confirms the endpoint responds.

## Evaluate

Five quiz questions drawn from the working code: identify which stage blocks a given input, trace the audit log format, predict behavior when a stage raises an exception versus returns a fail verdict, explain why short-circuit ordering matters for latency, and identify the structural difference between a middleware chain and a monolithic if-else block.

---

**Learning Objectives:**
1. Implement a sequential safety gate with composable stages.
2. Enforce short-circuit failure propagation across four classifier stages.
3. Write structured JSONL audit logs for every gate evaluation.
4. Compare middleware-chain vs. monolithic architecture for safety pipelines.
5. Configure stage thresholds via external YAML without code changes.