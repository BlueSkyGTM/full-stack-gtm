# Reviewer Agent: Separate Builder from Marker

## Beat 1: Frame It
The single-actor trap: when one LLM call both generates and self-evaluates, you get inflated quality scores and blind spots. Description: Establish why separating generation from evaluation is a structural pattern, not a style preference — self-review produces systematically biased judgments.

## Beat 2: Explain It
Two-pass architecture with independent contexts. Description: The builder agent produces output with a completion prompt. The marker agent receives that output plus a rubric — never the builder's system prompt — and returns a structured score. The marker's isolation is the mechanism: it cannot see the builder's reasoning, only the artifact.

## Beat 3: Build It
Implement builder and marker as separate function calls with separate system prompts, shared output contract (JSON schema), and a score threshold gate. Exercise hooks: Easy — run builder+marker on three inputs and print scores; Medium — add a revision loop where builder retries on failing scores; Hard — implement marker as an ensemble of two rubrics with disagreement detection.

## Beat 4: Use It
GTM cluster: Outbound copy quality gate. Description: The builder generates personalized email variants; the marker scores each against a four-point rubric (relevance, compliance, tone, claim accuracy). Only variants above threshold enter the send queue. This is the structural pattern behind multi-stage content pipelines in Clay waterfall workflows — generation and validation are separate stages.

## Beat 5: Ship It
Description: Wire the builder-marker pair into a production pipeline with logging, score persistence, and a fallback path for borderline outputs. Exercise hooks: Easy — log every builder-marker round to a JSONL file; Medium — add a human-review escalation path for scores in the gray zone; Hard — implement A/B comparison where two builder strategies compete on marker scores over 50 runs.

## Beat 6: Extend It
Description: The builder-marker split generalizes: planner-reviewer for strategy, researcher-validator for account intelligence, coder-tester for automation scripts. The same structural principle — independent evaluation with isolated context — applies wherever quality gates matter. [CITATION NEEDED — concept: multi-agent review patterns in GTM pipeline architecture]

---

**Learning Objectives (draft):**
1. Implement a two-pass builder-marker pipeline with separate system prompts and a shared JSON output contract.
2. Detect self-review bias by comparing single-actor scores against isolated marker scores on the same outputs.
3. Configure a score threshold gate that routes outputs to approve, retry, or human-review paths.
4. Evaluate marker reliability by running disagreement detection across multiple rubrics on identical builder output.