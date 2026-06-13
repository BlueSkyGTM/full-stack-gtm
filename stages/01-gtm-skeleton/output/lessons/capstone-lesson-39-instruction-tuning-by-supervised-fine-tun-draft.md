# Lesson 39: Instruction Tuning by Supervised Fine-Tuning

## Beat 1: Hook — Why This Lesson Exists

Base models complete text. They don't follow instructions. SFT is the algorithmic step that converts a raw text predictor into an instruction-following agent. Without it, "Write me a sales email" gets you more of the prompt echoed back. With it, you get the email. This lesson shows the mechanism that makes that conversion happen — loss masking over instruction–response pairs — and runs it end-to-end on a small model.

## Beat 2: Concept — How SFT Works

**Instruction dataset structure.** Each example is a triple: system prompt (optional), user instruction, and target response. The model sees the full concatenation during training, but loss is computed only on the response tokens.

**Loss masking mechanism.** Tokenize the full prompt+response sequence. Set labels to `-100` for every token in the instruction portion. The cross-entropy loss ignores `-100` labels. The model learns to generate the response conditioned on the instruction, not to parrot the instruction itself.

**Chat templates.** Different model families use different delimiters — `<|user|>`, `<|assistant|]>`, `### User:`, `### Assistant:`, or ChatML format. The template determines how instruction and response are concatenated. Wrong template = model sees gibberish boundaries.

**SFT vs. continued pretraining.** Continued pretraining trains on all tokens with uniform loss. SFT trains on response tokens only. Same optimizer, same forward pass, different label mask.

## Beat 3: Demo — Running SFT on a Toy Model

One self-contained script that:
1. Loads a small model (`sshleifer/tiny-gpt2` — 8M params, downloads in seconds)
2. Constructs an instruction dataset (10 examples, hard-coded, no external files)
3. Tokenizes with loss masking (labels set to `-100` for instruction tokens)
4. Runs a manual training loop (PyTorch `AdamW`, 3 epochs)
5. Generates a response to a held-out instruction before and after training
6. Prints both outputs to show the behavioral shift

Print statements confirm: dataset size, loss at each epoch, and the before/after generation comparison.

## Beat 4: Use It — GTM Application

**GTM cluster: AI-Driven Personalization (Zone 4)**

Instruction tuning is the mechanism behind custom GTM models that follow specific outreach patterns. A base model given "Write a follow-up email" produces generic text. An instruction-tuned model trained on your team's email-response pairs produces output that follows your formatting rules, tone, and call-to-action structure.

Concrete redirect: This is the same mechanism used when fine-tuning a model on closed-won email threads to generate new outreach that matches historical conversion patterns. The loss mask ensures the model learns from the *responses*, not the prompts.

If no clean GTM mapping exists for a specific variation: "foundational for Zone 4 — any system that requires a model to follow structured output instructions rather than free-form completion."

## Beat 5: Ship It — Production Considerations

**Data quality gates.** SFT amplifies whatever patterns exist in the training data. Garbage in, garbage out — but now it's *confident* garbage. Ship requires: deduplication, label verification, and format consistency checks.

**Catastrophic forgetting.** Training on a narrow instruction set degrades general capabilities. Mitigation: mix a fraction of general instruction data (e.g., Open-Orca subset) with domain-specific examples.

**Inference template consistency.** The chat template used at training must match the template used at inference. Mismatch = degraded performance. Ship checklist: serialize the tokenizer config alongside the model weights.

**Exercise hooks:**
- **Easy:** Modify the demo script to add 5 new instruction-response pairs and confirm the model adapts output.
- **Medium:** Implement a data validation function that rejects examples where the response is empty or the instruction exceeds 512 tokens.
- **Hard:** Add a mixed-data regime — 70% domain instructions, 30% general instructions — and measure perplexity on a held-out general set before and after training to detect catastrophic forgetting.

## Beat 6: Edge Cases — What Breaks

**Format overfitting.** Model learns the template tokens (`### Assistant:`) instead of the task. Symptom: correct format, empty or generic content. Fix: vary the template slightly during training or increase data diversity.

**Instruction leakage into response.** If the loss mask is misconfigured and includes instruction tokens, the model learns to repeat the prompt. Check: print the labels array and confirm `-100` alignment with non-response tokens.

**Short-response collapse.** On small datasets, the model may learn to produce minimally compliant responses ("Sure!") to minimize loss. Fix: filter training data to enforce minimum response length.

**Token boundary errors.** If the instruction/response split happens mid-token (rare with ASCII, common with multilingual content), the loss mask misaligns. Tokenize instruction and response separately, then concatenate the token sequences.

---

**Learning Objectives (for `docs/en.md`):**
1. Implement loss masking by setting instruction token labels to `-100` in a PyTorch training loop.
2. Format an instruction dataset using a chat template compatible with a specific model family.
3. Compare model outputs before and after SFT on a held-out instruction to evaluate behavioral shift.
4. Diagnose catastrophic forgetting by measuring perplexity on general vs. domain-specific held-out sets.
5. Configure a mixed-data training regime to balance domain adaptation with general capability retention.