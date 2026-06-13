# Instruction Tuning (SFT)

## Hook
Pretrained models complete text — they don't follow instructions. SFT is the process that converts a next-token predictor into an assistant that responds to directions. Without it, you have a parrot. With it, you have an agent.

## Concept
Mechanism: SFT trains on `(instruction, response)` pairs using supervised cross-entropy loss, but with **loss masking** — the model only learns from response tokens, not the instruction prompt itself. Cover chat templates (the formatting contract between system/user/assistant turns), data composition (what ratio of task types tanks or rescues performance), and the catastrophic forgetting tradeoff — every SFT step that teaches new behavior degrades some pretrain-era knowledge.

## Use It
**GTM Cluster: Zone 2 — Enrichment & Scoring / AI-assisted research**

In GTM pipelines, you need models that follow structured extraction instructions — not freestyle completion. SFT is what makes a model reliably output JSON with fields like `{"company": "...", "intent_signal": "...", "confidence": 0.8}` instead of generating a paragraph about company fit. This is the mechanism behind Clay's AI enrichments and any custom attribute extraction in a GTM waterfall. If you've ever written a prompt in a Clay column and gotten structured data back, SFT (or RLHF on top of SFT) is why that works.

[CITATION NEEDED — concept: specific Clay SFT or RLHF training pipeline details for structured extraction]

## Build It
Write a minimal SFT training loop on synthetic instruction-response data. Demonstrate loss masking by computing loss only on response tokens. Show before/after model behavior on the same instruction. Observable output: training loss decreasing, and the model's response changing from completion-style to instruction-following.

**Exercise hooks:**
- **Easy**: Run the training script on provided data. Report final loss.
- **Medium**: Modify the chat template format. Observe what breaks when the template mismatches between training and inference.
- **Hard**: Build an SFT dataset from GTM enrichment examples (company → intent signal extraction). Train and evaluate extraction accuracy against a held-out set.

## Ship It
Cover the deployment contract: the chat template used during SFT **must** match inference exactly — mismatched templates produce silent degradation, not errors. Discuss quantization tradeoffs (SFT at full precision, deploy at Q4 — what you lose). Show how to version your instruction format so downstream consumers (API callers, Clay integrations, pipeline stages) don't break when you update the template.

## Evaluate It
Benchmark instruction-following against the pretrained baseline. Metrics: exact match for structured extraction, refusal rate on out-of-scope queries, and regression on general knowledge (the forgetting check). Include a specific test: feed the same 20 instructions to both models, score responses on task completion (did it follow the format?) and correctness (is the content right?).

**Exercise hooks:**
- **Easy**: Compare pretrained vs. SFT model outputs on 10 instructions. Categorize failures.
- **Medium**: Build an evaluation harness that scores JSON extraction accuracy across edge cases (missing fields, wrong types).
- **Hard**: Measure catastrophic forgetting by running a general-knowledge benchmark before and after SFT. Plot the accuracy delta per domain.