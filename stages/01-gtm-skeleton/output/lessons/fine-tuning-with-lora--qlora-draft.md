# Fine-Tuning with LoRA & QLoRA

## Learning Objectives

1. Configure a LoRA adapter by specifying rank, alpha, and target modules for a given base model.
2. Compare the parameter counts and memory profiles of full fine-tuning vs. LoRA vs. QLoRA using empirical measurement.
3. Implement a QLoRA fine-tuning run with 4-bit quantization and verify that trainable parameters match expectations.
4. Evaluate fine-tuned model output against base model output on a domain-specific task.

---

## Beat 1: Hook

Full fine-tuning of a 7B-parameter model requires updating all 7 billion weights — which means GPU memory proportional to the full parameter count, plus optimizer states (Adam needs 2 additional tensors per parameter). LoRA freezes the original model and trains only small injected matrices, cutting trainable parameters by 99%+. QLoRA adds 4-bit quantization on top, squeezing the same operation into a fraction of the VRAM. This lesson builds both from mechanism to working code.

---

## Beat 2: Concept

**Mechanism: Low-Rank Adaptation (LoRA).** For a weight matrix $W$ of shape $d \times k$, LoRA injects two small matrices: $A$ (shape $r \times k$) and $B$ (shape $d \times r$), where $r \ll \min(d, k)$. The forward pass becomes $Wx + BAx$. Only $A$ and $B$ are trained. Rank $r$ controls the bottleneck — typical values are 8–64. Alpha scales the LoRA update relative to the base model (alpha/r is the effective scaling factor).

**Mechanism: QLoRA.** Same LoRA adapters, but the frozen base model is stored in 4-bit NormalFloat (NF4) quantization with double quantization (quantizing the quantization constants themselves). Computation still happens in bf16 after dequantizing on-the-fly. This trades a small quality loss for a dramatic memory reduction.

**Tools:** `peft` (Hugging Face library that implements LoRA adapter injection), `bitsandbytes` (implements NF4 quantization for QLoRA), `transformers` (provides the training loop via SFTTrainer).

**Key numbers to internalize:** A LoRA adapter for Llama-2-7B with $r=16$ on all linear layers adds ~20M trainable parameters (vs. 6.7B total). QLoRA at 4-bit reduces the base model from ~14GB in fp16 to ~4GB.

---

## Beat 3: Demo

Working code that:
1. Loads a model in 4-bit quantization using `bitsandbytes`.
2. Attaches a LoRA adapter via `peft` with explicit rank and alpha.
3. Prints the trainable parameter count and total parameter count to confirm the ratio.
4. Runs 10 training steps on a small synthetic dataset.
5. Prints a generation from the base model and the fine-tuned model on the same prompt for side-by-side comparison.

All output is printed to terminal. No notebooks, no browser.

**Exercise hooks:**
- *(Easy)* Modify `r` from 16 to 4 and observe how trainable parameter count changes. Document the ratio.
- *(Medium)* Load the same model without quantization (fp16) and with QLoRA (4-bit). Print peak GPU memory for each using `torch.cuda.max_memory_allocated()`.
- *(Hard)* Fine-tune on a custom dataset of 50 examples in a specific domain (e.g., GTM terminology). Evaluate whether the fine-tuned model correctly uses domain jargon that the base model does not.

---

## Beat 4: Use It

**GTM redirect:** This maps to GTM Cluster 07 — ABM signal orchestration. Fine-tuning with LoRA is the mechanism for training a scoring model on your own deal history. The signals — job changes, social activity, intent spikes — become your training labels. A base model that generates generic scoring can be adapted to score accounts based on what *your* closed-won deals actually looked like.

The key constraint: you need labeled examples. "Closed-won in 30 days after signal X" is a label. "We felt good about this account" is not. LoRA makes the compute tractable; it does not remove the requirement for structured training data.

[CITATION NEEDED — concept: LoRA fine-tuning applied to GTM account scoring models with CRM close data as labels]

---

## Beat 5: Ship It

**Exercise hook:** Take a public dataset of sales call transcripts (or synthetic data modeled on deal outcomes). Fine-tune a QLoRA adapter to classify whether a given account description represents a high-intent or low-intent signal. Save the adapter weights. Write an inference script that loads the base model + adapter and prints classification + confidence for 5 test inputs. The script must run from terminal and print all outputs.

**Deployment constraint to address:** LoRA adapters are ~50–200MB vs. the full model's ~14GB. This means you can ship the adapter independently and merge it at inference time, or serve multiple adapters on one base model. The exercise requires printing adapter size on disk to confirm this property.

---

## Beat 6: Evaluation

**Assessment hooks (not a quiz bank — objectives only):**

1. *Given a model architecture and LoRA config (r=32, alpha=64, target modules), calculate the number of trainable parameters.* (Tests objective 1, 2)

2. *Given output from two fine-tuning runs (r=8 vs r=64) on the same data, compare results on a held-out set and argue which config is preferable, including whether either shows signs of overfitting.* (Tests objective 4)

3. *Trace what happens during a single forward pass through a LoRA-adapted linear layer: which tensors are in 4-bit, which are in bf16, where dequantization occurs, and where gradients flow.* (Tests objective 1, 3)

4. *A practitioner reports that QLoRA fine-tuning uses 6GB VRAM but full fine-tuning OOMs on the same GPU. Explain the mechanism that accounts for this difference, with specific reference to what is stored in memory in each case.* (Tests objective 2)

**Self-check:** If you cannot sketch the $Wx + BAx$ computation graph from memory and label which parts are frozen, quantized, and trainable, revisit Beat 2 before proceeding.