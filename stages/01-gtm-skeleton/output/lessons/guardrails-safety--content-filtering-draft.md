# Guardrails, Safety & Content Filtering

---

## Beat 1: Hook — When the Model Goes Off-Script

A single unfiltered output can destroy a sales sequence. This beat opens with a real failure mode: an LLM-generate outbound email that hallucinates a pricing discount, leaks internal data, or produces off-brand language. The practitioner sees the financial cost of unguarded generation and why filtering is a production engineering concern, not a compliance afterthought.

---

## Beat 2: Core Concept — The Three-Layer Filter Stack

Explain the mechanism: input guardrails (validate before the call), output guardrails (classify after the call), and fallback behavior (reject, rewrite, or escalate). Define each layer's responsibility. Introduce the pattern of classification-based filtering where a smaller model or rule set judges the output of a larger model before delivery to the end user.

---

## Beat 3: Code Walkthrough — Build a Classification Guardrail

Working Python script that takes a generated sales email, runs it through a local classifier (regex + keyword rules for PII detection, competitor mentions, and prohibited claims), prints pass/fail for each rule, and outputs a final delivery decision. All output observable in terminal. No external API dependencies.

**Exercise hook (easy):** Add a new rule that flags superlatives ("best," "number one," "#1").
**Exercise hook (medium):** Rewrite the classifier to accept a configurable ruleset from a JSON file.
**Exercise hook (hard):** Implement a two-pass system where flagged outputs are sent back to the LLM with the violation context for self-correction, then reclassified.

---

## Beat 4: Use It — Guardrails in GTM Content Generation

This is the Clay outbound waterfall application. When Clay generates personalized emails at scale, guardrails prevent off-brand sends. The practitioner maps the three-layer filter stack to a Clay workflow: input validation on prospect data, output filtering on generated copy, and fallback to a template when the AI output fails classification. The redirect is specific: "this is how you prevent a Clay-personalized sequence from sending a bad email to 10,000 prospects."

GTM cluster: Zone 02 — Outbound & Prospecting (content generation guardrails)

[CITATION NEEDED — concept: exact Clay waterfall stage where output filtering is applied]

---

## Beat 5: Common Pitfalls — Where Guardrails Fail in Practice

Covers the failure modes practitioners encounter: over-filtering that strangles personalization into generic templates, under-filtering that lets subtle hallucinations through, false positives on legitimate industry terminology, and the latency cost of multi-pass classification. Addresses the tension between speed (batch send) and safety (check every output).

---

## Beat 6: Ship It — Deploying Guardrails to Production

The practitioner implements a production-ready guardrail pipeline: logging every rejection, monitoring filter rates over time (drift detection), and configuring escalation paths. Ends with a deployable script that wraps any LLM call in the filter stack, logs decisions to a local file, and prints aggregate statistics.

**Exercise hook (easy):** Add logging that writes every rejected output to a CSV with the failure reason.
**Exercise hook (medium):** Build a dashboard script that reads the rejection log and prints filter-rate percentages by rule type.
**Exercise hook (hard):** Implement A/B threshold testing — run the same batch through strict and lenient configurations, compare outputs side by side, and print the diff summary.

---

## Learning Objectives

1. Implement a three-layer guardrail system (input, output, fallback) for LLM-generated content.
2. Classify generated text using rule-based filters for PII, prohibited claims, and off-brand language.
3. Configure rejection, rewrite, and escalation paths for failed outputs.
4. Evaluate filter performance by monitoring rejection rates and false positive patterns.
5. Deploy guardrails into a GTM content generation pipeline without degrading personalization quality.