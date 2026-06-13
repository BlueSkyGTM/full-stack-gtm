# Lesson Outline: LLM Evaluation — RAGAS, DeepEval, G-Eval

---

## Beat 1: Hook

You've built a RAG pipeline that surfaces account intelligence. Leadership asks: "Is it any good?" You stare at 200 generated outputs with no systematic way to answer. This lesson gives you the frameworks to quantify "good" — and catch when your LLM outputs drift from useful to garbage.

---

## Beat 2: Concept

Three evaluation mechanisms, each solving a different measurement problem:

- **RAGAS** (Retrieval Augmented Generation Assessment): Decomposes RAG quality into component metrics — faithfulness (grounded in context?), answer relevance (on-topic?), context precision (retrieved the right chunks?), context recall (retrieved all necessary chunks?). Computes scores using an LLM-as-judge pattern with specific prompting strategies per metric. Designed for RAG pipelines specifically.

- **G-Eval**: Uses chain-of-thought reasoning to generate evaluation criteria, then applies probabilistic scoring (token probabilities of 1-5 ratings) to reduce variance. Framework-agnostic — works on any text generation task. The mechanism: generate evaluation steps via CoT, then score with weighted probabilities rather than raw ratings.

- **DeepEval**: Implements multiple evaluation metrics (including RAGAS-style and G-Eval-style) as test cases with a pytest-like interface. Provides the testing scaffolding: define test cases, attach metrics, run suite, get pass/fail. Solves the integration problem — how to bake evaluation into CI/CD rather than running ad-hoc notebooks.

Key distinction: RAGAS measures retrieval + generation quality. G-Eval measures any generation against custom criteria. DeepEval packages both (and more) into a test framework.

---

## Beat 3: Demonstration

Show each framework producing scores on the same GTM-relevant sample: a RAG pipeline that retrieves account intelligence and generates a research brief.

- RAGAS: Score faithfulness and answer relevance on 5 sample outputs using `ragas` library
- G-Eval: Score "actionability" of the same outputs using custom criteria with `deepeval`'s GEval implementation
- DeepEval: Package all metrics into a test suite that returns pass/fail with threshold configuration

All code runs locally with hardcoded sample data. Observable output: metric scores printed to terminal.

---

## Beat 4: Use It

**GTM Redirect**: Zone 1 (Intelligence Gathering) — specifically the account research and enrichment pipeline.

Application: You're generating account briefs from 10-K filings and news articles via RAG. You need to know if your outputs are faithful to source material and actually useful for SDR outreach.

Exercise hooks:
- **Easy**: Score 3 pre-generated account briefs using RAGAS faithfulness metric. Identify which one hallucinates.
- **Medium**: Write a G-Eval custom metric for "outreach actionability" — does this brief contain specific, usable talking points? Score 5 outputs and compare with human ratings.
- **Hard**: Build a DeepEval test suite that runs on every prompt template change. Configurable thresholds for faithfulness (>0.85) and relevance (>0.80). Pipeline fails CI if scores drop.

---

## Beat 5: Ship It

Production evaluation pipeline pattern:

1. **Offline evaluation**: Run full RAGAS + DeepEval suite on a held-out evaluation set (50-100 curated examples) before deploying any prompt or retrieval changes
2. **Online monitoring**: Sample 5% of production outputs, score with lightweight metrics (answer relevance + custom G-Eval), log to tracking system
3. **Alerting**: If sampled scores drop below threshold for 3 consecutive windows, trigger review

Implementation: Script that loads evaluation dataset from JSONL, runs RAGAS metrics, outputs structured results. No manual steps.

GTM Redirect: This is the quality gate for your Clay enrichment waterfall and AI-powered account research. Without evaluation, you're shipping outreach based on unvalidated LLM outputs.

Exercise hooks:
- **Easy**: Run the provided evaluation script against a sample dataset and interpret results
- **Medium**: Add a custom G-Eval metric to the pipeline for "ICP relevance scoring" — rate whether the generated insight matches your ideal customer profile attributes
- **Hard**: Implement the full offline + online evaluation pattern with configurable thresholds and a JSON output format compatible with your CI system

---

## Beat 6: Review

Three frameworks, three measurement problems:

| Framework | What It Measures | Best For |
|-----------|-----------------|----------|
| RAGAS | RAG pipeline components (retrieval + generation) | Evaluating account intelligence pipelines |
| G-Eval | Any generation against custom criteria | Domain-specific quality (actionability, relevance to ICP) |
| DeepEval | Test framework for all metrics | CI/CD integration, regression testing |

Key takeaways:
- Evaluation must be decomposed into specific, measurable components — "is it good?" is not a metric
- LLM-as-judge introduces its own variance — use probabilistic scoring (G-Eval) or multi-metric aggregation (RAGAS) to reduce noise
- Evaluation sets are assets — curate them, version them, protect them. Garbage in, garbage scores out.

**Common failure modes**: Evaluating on training data (circular), using only one metric (blind spots), ignoring retrieval quality when only measuring generation quality.

**Next lesson**: Evaluation drift detection — how to notice when your production outputs diverge from your evaluation set.

---

## Learning Objectives (for `docs/en.md`)

1. Implement RAGAS faithfulness and answer relevance metrics on a RAG pipeline output and interpret scores
2. Configure a G-Eval metric with custom evaluation criteria for domain-specific content quality
3. Build a DeepEval test suite that runs multiple metrics against a curated evaluation set
4. Compare RAGAS, G-Eval, and DeepEval on measurement scope, variance handling, and integration complexity
5. Detect when an LLM output evaluation pipeline produces unreliable scores due to metric configuration issues

---

## GTM Redirect Rules (Summary)

- **Beat 4**: Zone 1 application — evaluating account research pipeline outputs
- **Beat 5**: Quality gate for Clay enrichment waterfall and AI-powered research — foundational for Zones 1-2
- **No forced connections**: Framework choice depends on what you're measuring, not GTM stage