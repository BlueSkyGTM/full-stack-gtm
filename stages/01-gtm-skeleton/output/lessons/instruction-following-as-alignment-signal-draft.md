# Instruction-Following as Alignment Signal

## Beat 1: Hook

A model that completes text and a model that follows instructions are different systems. This lesson examines why instruction-following behavior is the observable surface of alignment, and how to measure it rigorously.

## Beat 2: Concept

Define instruction-following as an alignment signal: not what the model *knows*, but what it *does* when given a directive. Contrast base models (document completers) with instruction-tuned models (directive followers). Introduce the alignment spectrum from raw next-token prediction through RLHF to instruction-tuned behavior.

**Exercise hook (Easy):** Query a base model and an instruction-tuned model with the same prompt. Print both outputs. Observe the difference in behavior.

## Beat 3: Mechanism

Explain the mechanism: SFT (Supervised Fine-Tuning) on demonstration data, then RLHF or DPO to sharpen instruction-following. Show how reward models score "followed the instruction" vs. "produced fluent text." Explain why instruction-following is a proxy for alignment—what the model does reveals what it optimized for.

**Exercise hook (Medium):** Build a grading function that scores model outputs against explicit instruction constraints (format, length, persona). Run it across multiple outputs. Print scores.

## Beat 4: Use It

GTM redirect: **Zone 2 — Enrichment**. When you write enrichment prompts in Clay or any GTM tool, you are relying on instruction-following alignment. A waterfall that asks "extract the person's title and return JSON" depends on the model following that instruction, not just generating plausible text. Poor instruction-following produces malformed enrichment payloads; strong instruction-following produces clean structured data. This is the Clay waterfall's hidden dependency on alignment quality.

**Exercise hook (Medium):** Write three enrichment prompts of increasing constraint complexity. Test each against a model. Score output against the constraints you specified. Print pass/fail per constraint.

## Beat 5: Ship It

Build an instruction-following evaluator: a script that sends prompts with known-correct outputs, collects model responses, and scores alignment quality across dimensions (format compliance, content accuracy, constraint adherence).

**Exercise hook (Hard):** Implement a batch evaluator that tests 10 prompts, each with 3+ constraints, against a model endpoint. Output a scored matrix and aggregate alignment percentage.

## Beat 6: Review

Summarize: instruction-following is the observable behavior that reveals alignment. Measure it. Test it. Don't assume it. When your GTM enrichment pipeline produces garbage, the root cause is often an instruction-following gap, not a prompt wording issue.

---

**GTM Redirect Rules for this lesson:**
- "Use It" references Zone 2 Enrichment and Clay waterfall structured output extraction
- "Ship It" builds a general evaluator applicable to any GTM pipeline that depends on model instruction-following
- If the GTM connection feels forced in any exercise, the redirect defaults to: "foundational for Zone 1 and Zone 2 — all enrichment depends on instruction-following alignment"