# Show-o and Discrete-Diffusion Unified Models

---

## Learning Objectives

1. Compare discrete diffusion against continuous diffusion and autoregressive decoding on token sequences
2. Implement a minimal mask–unmask discrete diffusion loop with observable intermediate states
3. Diagram the Show-o dual-branch architecture: autoregressive for understanding, masked diffusion for generation
4. Evaluate when a single unified token vocabulary enables or blocks multi-task training
5. Instrument a training step to log the joint loss (understanding term + generation term) separately

---

## Hook

Practitioners currently maintain separate stacks: a VLM for image understanding and a diffusion model for image generation. Show-o collapses both into one transformer with one token vocabulary and one forward pass. The lesson opens with a side-by-side inference trace—same model, two prompts—and asks: what mechanism lets one weight set serve both regimes?

---

## Concept

**Discrete diffusion** replaces the Gaussian noise schedule of continuous diffusion with a Markov chain over token substitutions (mask → correct token, or token → mask). The lesson introduces the transition matrix **Q**, derives the marginal at timestep *t*, and shows why the reverse process is "predict the original token given a corrupted sequence." This is the core mechanism Show-o builds on; once this is clear, the dual-head architecture follows naturally.

---

## Mechanism

Three parts, each explained before any tooling:

1. **Token unification** — images are tokenized via MAGVIT-v2 into discrete codes; text uses BPE. Both share the same codebook, so the transformer sees a flat sequence with no modality tag.
2. **Forward corruption** — for understanding inputs, tokens stay intact (autoregressive left-to-right); for generation inputs, a random subset of image tokens is replaced with `[MASK]` according to a cosine schedule.
3. **Reverse denoising** — the model predicts the original token at each masked position. Cross-entropy over the vocabulary is the loss; no continuous noise, no VAE decoder.

Diagram: single transformer block → two training paths → one inference path (prompt autoregressively, then iteratively unmask image tokens).

---

## Code

Three runnable scripts, all terminal-only:

1. **Minimal discrete diffusion on a toy vocabulary** — define a 16-token vocab, corrupt a sequence with a linear mask schedule, train a 1-layer transformer to reverse it. Prints original → corrupted → recovered at each epoch.
2. **Dual-loss instrumented training step** — mock data (text tokens + image tokens), compute `loss_understanding` (AR cross-entropy) and `loss_generation` (masked token prediction), print both terms and the weighted sum.
3. **Inference trace** — given a text prompt, run 5 denoising steps on a fully masked image grid; print the grid state at each step so the practitioner watches tokens being "filled in."

No comments in code. Every script prints observable output.

---

## Use It

**GTM redirect:** this lesson is *foundational for Zone B (AI Engineering Foundations)*. The mechanism—joint training over heterogeneous token sequences under a single loss—reappears whenever a GTM practitioner fine-tunes a unified model for both understanding (lead scoring, intent classification) and generation (email drafting, personalized copy). No forced Clay waterfall connection; the takeaway is architectural literacy for multi-task model selection.

---

## Ship It

Practitioner ships a minimal configuration that demonstrates:

- A unified data loader batching understanding samples (text-only) and generation samples (text+image) with the correct loss mask per sample.
- A `torch.compile`-compatible training loop logging `loss_ar` and `loss_diffusion` to stderr every 100 steps.
- A CLI entry point: `python train.py --config unified.yaml` that completes 1 epoch on a toy dataset and prints final joint loss.

Exercise hooks:

- **Easy:** Modify the mask schedule from linear to cosine; observe convergence speed change in printed loss.
- **Medium:** Add a third task branch (e.g., text infilling) to the dual-loss script; print all three loss terms.
- **Hard:** Replace the toy vocabulary with a real MAGVIT-v2 codebook (download provided), run end-to-end denoising on a 16×16 image grid, and save the per-step grids as PNGs.

---

*Citation status: Show-o architecture details reference the original paper (Xie et al., 2024). [CITATION NEEDED — concept: MAGVIT-v2 tokenizer integration details in Show-o implementation].*