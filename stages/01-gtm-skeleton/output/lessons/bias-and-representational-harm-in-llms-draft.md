# Bias and Representational Harm in LLMs

## Learning Objectives

1. Distinguish between statistical bias and social bias in language model outputs
2. Categorize representational harm types (stereotyping, under-representation, denigration, erasure) using concrete output examples
3. Run structured bias evaluations against an LLM using WinoBias-style sentence completion
4. Implement output-level filtering that flags potentially harmful completions before they reach end users
5. Compare prompt-level and system-level mitigation strategies for a GTM content generation pipeline

---

## Beat 1: Hook

A hiring tech company deployed an LLM-based tool to generate outreach messages. It consistently described engineers as "he" and nurses as "she." Candidates flagged the pattern. The company pulled the tool within 48 hours. This lesson covers how to detect and interrupt that class of failure before it ships.

---

## Beat 2: Learn It

Definitions and mechanisms first. Distinguish statistical bias (systematic error in predictions) from social bias (representational harm toward groups). Map five harm categories: stereotyping, under-representation, denigration, erasure, and alienation. Trace where bias enters the pipeline: training corpus composition, annotation preferences, RLHF reward modeling, and deployment context.

Key mechanism: LLMs predict the highest-probability next token given training distribution. If training data encodes disproportionate associations (doctor→he, nurse→she), the model reproduces them. This is not a model "having opinions"—it is conditional probability reflecting corpus statistics.

**Exercise hooks:**
- *Easy:* Given 10 sample completions, categorize each by harm type (stereotype, erasure, denigration, none).
- *Medium:* Write a set of 8 sentence-completion prompts designed to surface gender-occupation associations. Run them against an LLM and log results.

---

## Beat 3: See It

Walk through a working Python script that sends paired prompts to an LLM (e.g., "The CEO promised _ would" vs. "The nurse said _ would") and logs pronoun distributions. Show the output. Demonstrate that swapping a single identity term shifts the model's completion probabilities. Then show the same test with mitigation in the system prompt.

Mechanism demonstrated: token-level probability difference across identity-swapped sentence pairs. This is the core diagnostic—not vibes, not anecdotes, but repeatable measurement.

**Exercise hooks:**
- *Easy:* Run the provided script and report the pronoun distribution table.
- *Hard:* Extend the script to test three additional identity axes (race-indicating names, age, disability-related context) and produce a combined bias report.

---

## Beat 4: Use It

GTM redirect: **Zone 2 — Outbound content generation.**

When LLMs generate outbound sequences in Clay or similar tools, representational harm becomes a brand risk. An outreach template that defaults to gendered honorifics, assumes cultural context, or stereotypes job roles will produce biased outputs at scale—across thousands of prospects.

The application: build a bias-check step in your content generation pipeline. Before a generated email or LinkedIn message is queued for sending, run it through a lightweight evaluation: flagging gendered assumptions in role descriptions, checking for exclusive language patterns, and scoring against a rubric you define.

**Exercise hooks:**
- *Medium:* Write a Python function that accepts a generated outreach message and returns a bias-risk score based on pattern matching against a curated list of flagged associations. Test it on 20 generated messages.

---

## Beat 5: Ship It

Production bias-testing harness. A repeatable evaluation suite that:
1. Maintains a prompt library of bias-sentinel inputs (pairs designed to surface harm)
2. Runs those prompts against your deployed model endpoint on a schedule
3. Compares results to a baseline and flags regressions
4. Logs results to a dashboard or file

This is not a one-time audit. Bias can regress with model updates, prompt changes, or fine-tuning. Ship a test that runs continuously.

**Exercise hooks:**
- *Hard:* Build a CLI tool that accepts an API endpoint, runs a 20-prompt bias evaluation suite, outputs a JSON report with per-category scores, and exits with code 1 if any score exceeds a configurable threshold.

---

## Beat 6: Stretch It

Intersectional bias: harm that appears only at the intersection of identity categories (e.g., race + gender) and is invisible in single-axis testing. Cross-lingual bias: harm that manifests differently or more severely in non-English outputs, where most safety tuning has not been applied.

Limitations of prompt-level mitigation: system prompts reduce but do not eliminate bias. The model's weights still encode the associations. For high-stakes GTM applications (hiring, financial services, healthcare), prompt-level mitigation is insufficient. Weight-level intervention (fine-tuning on curated data) or output-level hard filters are necessary.

[CITATION NEEDED — concept: empirical comparison of bias mitigation effectiveness across prompt-level, system-prompt-level, and fine-tuning approaches in production LLM deployments]

**Exercise hooks:**
- *Hard:* Design an intersectional evaluation: test an LLM's completions across a matrix of (gender) × (race-indicating name) × (profession) and report where single-axis testing misses harm that the full matrix reveals.