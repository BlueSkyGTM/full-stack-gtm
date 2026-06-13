# Pre-Training a Mini GPT (124M Parameters)

## GTM Redirect Rules

This lesson maps to **Zone C: Model Training & Fine-Tuning**. Pre-training is the upstream mechanism that makes fine-tuning possible. The redirect: "the same autoregressive training loop that pre-trains GPT is the loop you modify when fine-tuning on your GTM-specific corpus — this is the mechanism beneath every custom model deployment." Where no direct GTM application exists, the redirect is: "foundational for Zone C — you cannot debug fine-tuning failures without understanding what the pre-training loop actually computes."

---

## Beat 1: Hook — Why Pre-Training, Why Now

**Description:** You have a GPT architecture (from the prior lesson). A random-weight model outputs noise. Pre-training is the process that converts random weights into a model that produces coherent text. This beat establishes what changes during training (weights, via gradient descent on next-token prediction loss) and what does not (architecture). States the concrete deliverable: by end of this lesson, you will have trained a 124M-parameter GPT-2 on a real corpus and generated sample outputs that demonstrate progression from noise to structure.

---

## Beat 2: Concept — The Pre-Training Mechanism

**Description:** Covers the four mechanisms that compose the pre-training loop. (1) **Next-token prediction as the training objective**: the model receives a sequence of tokens and must predict the token at each position given only prior context — this is causal language modeling. (2) **Cross-entropy loss**: quantifies the divergence between the model's predicted probability distribution over the vocabulary and the actual next token; this is the scalar signal that drives all weight updates. (3) **Autoregressive teacher forcing**: during training, the input is the ground-truth sequence shifted by one position; during inference, the model feeds its own predictions back autoregressively — this distinction explains why training loss and generation quality are related but not identical. (4) **The training loop structure**: forward pass computes loss, backward pass computes gradients, optimizer step updates weights, then repeat for N iterations over the data.

---

## Beat 3: Code — Build the Pre-Training Loop

**Description:** Implements a complete pre-training loop for a 124M-parameter GPT-2 model architecture. Uses PyTorch and a publicly available text corpus (e.g., OpenWebText subset or a Shakespeare-style dataset for faster iteration). Code includes: data tokenization and batching, model instantiation, training loop with loss printing at regular intervals, gradient accumulation (for simulating larger batch sizes on limited hardware), and a simple text generation function to sample from the model at checkpoints. Every code block runs without modification and prints observable output: loss values decreasing over steps and generated text samples showing progression.

**Exercise Hooks:**
- **Easy:** Modify the learning rate and observe its effect on loss convergence speed. Print loss at every 10 steps instead of every 100.
- **Medium:** Implement gradient accumulation to simulate a batch size of 64 while physically using batch size 8. Verify the loss curve matches training with actual batch size 64.
- **Hard:** Replace the constant learning rate with a cosine annealing schedule with linear warmup. Compare final loss and generation quality against the constant learning rate baseline.

---

## Beat 4: Use It — GTM Application

**Description:** Connects the pre-training mechanism to GTM-specific model development. The autoregressive training loop is the same loop used when continued-pre-training a base model on domain-specific GTM data (e.g., training on a corpus of sales transcripts, support tickets, or ICP descriptions before fine-tuning for classification). The key insight: pre-training on domain text before fine-tuning produces measurably better results than fine-tuning alone, because the model has already adapted its internal representations to the vocabulary and patterns of the target domain. [CITATION NEEDED — concept: domain-adaptive pre-training performance delta in GTM-specific text corpora]. Exercise asks the practitioner to identify what GTM corpus they would use for continued pre-training and estimate the token count required.

**Exercise Hooks:**
- **Easy:** Tokenize a sample of 50 sales emails and compute how many tokens the corpus contains. State whether this is sufficient for continued pre-training or if fine-tuning alone is more appropriate.
- **Medium:** Design a two-phase training plan: phase 1 is continued pre-training on a GTM corpus, phase 2 is fine-tuning for a downstream task (e.g., lead scoring, intent classification). Specify the data requirements for each phase.

---

## Beat 5: Ship It — Checkpoints, Saving, and Validation

**Description:** Covers the operational mechanics of persisting a trained model and validating that training actually worked. Topics: saving model checkpoints at regular intervals using `torch.save` on the `state_dict`, loading checkpoints to resume training, computing validation loss on a held-out split to detect overfitting (training loss decreases but validation loss increases), and a heuristic for selecting which checkpoint to deploy (the one with lowest validation loss, not the last one). Includes code to save, load, and generate text from a saved checkpoint. Discusses the memory footprint of a 124M-parameter model in fp32 (~496MB) vs fp16 (~248MB) and implications for deployment.

**Exercise Hooks:**
- **Easy:** Save a checkpoint at step 500, restart the script, load the checkpoint, and resume training. Print loss before and after resume to confirm continuity.
- **Medium:** Split your dataset into 90% train / 10% validation. Compute and print both training loss and validation loss every 100 steps. Identify the checkpoint with the lowest validation loss.
- **Hard:** Implement early stopping: halt training if validation loss has not improved for 3 consecutive evaluations. Save the best checkpoint automatically.

---

## Beat 6: Debug It — When Training Goes Wrong

**Description:** Covers the most common failure modes in pre-training and how to diagnose each. (1) **Loss is NaN**: learning rate is too high, causing gradient explosion — reduce learning rate by 10x, check for division by zero in loss computation. (2) **Loss plateaus immediately**: learning rate is too low, or the model is already converged (check that weights are actually being updated by printing gradient norms). (3) **Loss decreases but generated text is gibberish**: model may be memorizing shortcuts, context window may be misconfigured, or tokenization may be incorrect — generate samples at multiple checkpoints to catch this early. (4) **Out of memory during training**: reduce batch size, enable gradient checkpointing, or switch to mixed precision (fp16). Each failure mode includes a diagnostic command or code snippet that prints the relevant metric.

**Exercise Hooks:**
- **Easy:** Deliberately set the learning rate to 1.0, run 50 steps, observe NaN loss. Then set learning rate to 1e-5, run 50 steps, observe plateau. Print loss values for both.
- **Medium:** Add gradient norm logging to the training loop. Print the L2 norm of gradients at every step. Identify the threshold above which training becomes unstable for this model.
- **Hard:** Simulate a subtle tokenization bug where input sequences are off-by-one (the predicted token is included in the input). Train for 200 steps and compare loss to a correct implementation. Explain why the loss is artificially low.

---

## Learning Objectives

1. **Implement** the autoregressive pre-training loop (forward pass, cross-entropy loss computation, backward pass, optimizer step) for a 124M-parameter GPT model.
2. **Compare** training behavior under different learning rates and batch sizes, using loss curves and generated text samples as evaluation signals.
3. **Configure** gradient accumulation to simulate larger effective batch sizes on memory-constrained hardware.
4. **Diagnose** the four most common training failure modes (NaN loss, plateau, coherent-loss-but-gibberish-output, OOM) using gradient norms, loss inspection, and checkpoint sampling.
5. **Save and load** model checkpoints, and select the best checkpoint for deployment using validation loss as the criterion.