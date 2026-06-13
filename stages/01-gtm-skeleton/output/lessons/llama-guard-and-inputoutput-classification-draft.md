# Llama Guard and Input/Output Classification

## Hook

You're running an AI pipeline that generates outreach copy, qualifies inbound leads, or triages support tickets. One wrong output—PII leaked, harmful content generated, or off-brand messaging sent—creates real liability. Llama Guard is a classifier that sits between your user input and your model output, applying a taxonomy of harm categories to both sides of the exchange.

## Concept

Llama Guard implements a multi-label safety classification schema over a fixed taxonomy of harm categories (violence, hate speech, sexual content, PII, etc.). It is itself a fine-tuned Llama model that takes a conversation role (`User` or `Agent`) and text, then returns `safe` or `unsafe` with specific category violations. The mechanism: prompt-formatted classification where the model's output is constrained to a structured taxonomy rather than free-form generation. This lesson covers the taxonomy schema, the input/output scoring split, and how to interpret the multi-label violation output.

## Model

A mental model for where Llama Guard sits in an inference pipeline: input → Llama Guard (input check) → your task model → Llama Guard (output check) → delivery. The key design decision is whether to run classification on the input, the output, or both, and what to do with `unsafe` verdicts—reject, log, or route to human review.

## Use It

**GTM Redirect:** This is the content moderation layer for AI-generated outreach and inbound lead triage in [CITATION NEEDED — concept: GTM cluster for content safety in automated outreach pipelines]. Build a Python function that classifies a batch of synthetic outreach messages and support ticket responses through Llama Guard's taxonomy, logging which categories trigger `unsafe` and at what confidence. Exercise hooks: *(Easy)* Classify a single input and a single output, print the violation categories. *(Medium)* Build a wrapper that accepts any text, auto-detects role, and returns a structured safety report. *(Hard)* Implement a two-pass pipeline—input check then output check—with a rejection policy and a fallback path.

## Ship It

Package the classifier as a reusable module with a clear API: `classify_input(text)`, `classify_output(text)`, `classify_conversation(messages)`. Include logging for every `unsafe` verdict (category, raw text hash, timestamp), configurable rejection policies per category, and a dry-run mode that logs but doesn't block. Wire it into a mock outreach generation pipeline so the full loop—generate, classify, reject or deliver—is observable end to end.

## Evaluate

- Classify a set of borderline inputs and explain which taxonomy categories triggered and why.
- Compare Llama Guard's `unsafe` verdicts against a hand-labeled test set; calculate precision and recall per category.
- Diagnose a false positive: identify which part of the input triggered a violation and propose a mitigation (prompt rewrite, category exemption, or human review escalation).
- Configure a rejection policy that blocks on specific categories but logs-only on others, and demonstrate the different behaviors with test inputs.

---

**Learning Objectives (for `docs/en.md`):**
1. Classify text inputs and outputs using Llama Guard's harm taxonomy.
2. Explain the difference between input-side and output-side safety classification and when to apply each.
3. Implement a two-pass classification pipeline with configurable rejection policies.
4. Evaluate classifier performance against labeled test data and diagnose false positives.
5. Configure category-level exemption and escalation rules for production deployment.