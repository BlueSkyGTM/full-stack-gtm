# Prompt Engineering: Techniques & Patterns

## Hook It

You've been writing prompts by vibes. There's a better way. This lesson covers the repeatable patterns that turn inconsistent LLM outputs into reliable pipeline steps — the same patterns that power enrichment waterfalls and personalization at scale.

## Map It

Six core prompt patterns: zero-shot, few-shot, chain-of-thought, role assignment, output constraint, and prompt chaining. Each pattern solves a specific failure mode (hallucination, format drift, reasoning collapse). We'll map each pattern to its mechanism and its failure mode so you can diagnose which one to reach for.

## Build It

Working code for each pattern against Claude's API. Every example runs, accepts real input, and prints structured output you can inspect. Starts with a single-prompt classification task, then layers on patterns one at a time — few-shot examples to fix format drift, chain-of-thought to fix reasoning errors, output constraints to force parseable JSON. You see the failure, then the pattern that fixes it.

## Use It

GTM redirect: **Clay waterfall enrichment**. The enrichment waterfall is literally a prompt chain — each step conditions on the prior step's output. Few-shot patterns fix company classification accuracy. Output constraints force structured data that downstream formulas can consume. Role assignment reduces hallucination in niche industry research. You'll implement a two-stage enrichment prompt (research → score) that produces the same shape data Clay expects.

## Ship It

Prompt versioning, failure testing, and cost budgeting. A prompt that works on 10 examples can break at 1,000. You'll write a test harness that runs your prompt against labeled examples, checks output schema compliance, and flags drift. Covers token counting for cost estimation and a simple prompt registry pattern so you're not hardcoding prompts in production scripts.

## Drill It

**Easy:** Write a zero-shot prompt that classifies companies into ICP tiers; print structured results for 5 test inputs.
**Medium:** Build a few-shot prompt chain that enriches a lead record (company summary → relevance score) with JSON output.
**Hard:** Implement a prompt evaluation harness that tests your enrichment chain against 20 labeled examples, reports accuracy and schema compliance, and flags which inputs cause failures.

---

## Learning Objectives

1. Implement zero-shot, few-shot, and chain-of-thought prompts and compare their output quality on the same classification task.
2. Diagnose prompt failure modes (hallucination, format drift, reasoning collapse) and select the correct pattern to fix each.
3. Construct a multi-step prompt chain that produces structured JSON output compatible with downstream data pipelines.
4. Build a test harness that evaluates prompt accuracy and schema compliance across a labeled dataset.
5. Estimate token costs for a prompt chain given input/output length targets.

## GTM Redirect

Cluster: **Enrichment Waterfall** (stages/00-b-gtm-content-mapping/output/gtm-topic-map.md — enrichment and scoring cluster). Prompt engineering is the mechanism that makes waterfall enrichment work. Every enrichment step is a prompt. Every format constraint is what lets the next column's formula parse the result. This is not an analogy — this is the literal implementation.