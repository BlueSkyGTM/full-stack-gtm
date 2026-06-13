# Result Evaluator

## Hook

You've built a pipeline that generates outputs — enrichment data, lead scores, outreach copy. Now you need to answer one question: is any of it good? A Result Evaluator is the automated judge that scores, filters, or reranks AI outputs against defined criteria before they touch a downstream system or a human.

## Concept

A Result Evaluator applies a scoring function to an AI output. That function can be rule-based (regex, length checks, keyword presence), model-based (a classifier, an embedding similarity score), or LLM-as-judge (a second model evaluating the first model's output). The mechanism is always the same: input → output → score → threshold → accept/reject/rerank. Multi-criteria evaluators decompose the score into dimensions (relevance, accuracy, completeness, tone) and aggregate them. The evaluator is not the generator — it is a separate system that creates an adversarial or complementary signal.

## Use It

**GTM Redirect:** In Clay waterfall enrichment and AI outbound workflows, a Result Evaluator is what separates "we enriched 10,000 accounts" from "we enriched 10,000 accounts with usable data." You evaluate enrichment completeness before routing to segmentation. You evaluate AI-generated email copy against brand and compliance criteria before queuing for send. This maps directly to the enrichment validation and output governance stages in the GTM enrichment and outreach clusters.

**Exercise hooks:**
- *Easy:* Write a rule-based evaluator that scores an enriched company description on completeness (has industry, has employee count, has HQ location) and returns accept/reject.
- *Medium:* Implement an LLM-as-judge evaluator that scores AI-generated outreach emails on three criteria (relevance, tone, compliance) using a structured prompt, returning a JSON score object.
- *Hard:* Build a composite evaluator that runs both rule-based and LLM-as-judge scoring, aggregates the results with configurable weights, and routes outputs to "accept," "review," or "reject" buckets with printed rationale.

## Demo

Build a working Result Evaluator in Python that takes a list of AI-generated outreach emails, evaluates each against three criteria using a second LLM call structured as a judge, scores each on a 1–5 scale, and prints the accept/reject decision with scores. No external API keys — use a deterministic mock judge function that simulates the scoring logic so the code runs unmodified and produces observable output.

## Ship It

Deploying a Result Evaluator in production requires deciding: synchronous (block the output until evaluated) vs. asynchronous (evaluate after delivery, log failures for review). You need a scoring schema, a threshold configuration, and a fallback for when the evaluator itself fails. Cover the pattern of storing evaluation results alongside outputs for audit trails, and the anti-pattern of using the same model to generate and evaluate — the evaluator must be independent. [CITATION NEEDED — concept: evaluation latency budgets in GTM enrichment pipelines]

## Assess

Evaluate whether the practitioner can distinguish evaluator architectures (rule-based vs. model-based vs. LLM-as-judge), implement a multi-criteria scoring function with configurable thresholds, and explain where the evaluator sits relative to the generator in a GTM pipeline. Quiz questions trace directly to the mechanism: scoring function → threshold → routing decision.

---

**Learning Objectives:**
1. Implement a multi-criteria Result Evaluator that scores AI outputs against defined rubrics
2. Compare rule-based, model-based, and LLM-as-judge evaluation approaches with trade-offs for each
3. Configure threshold-based routing (accept/review/reject) for evaluated outputs
4. Diagnose evaluator failures and design fallback behavior
5. Position the Result Evaluator correctly in a GTM enrichment or outreach pipeline