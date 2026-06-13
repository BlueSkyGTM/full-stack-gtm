# Flamingo and Gated Cross-Attention for Few-Shot VLMs

## Hook

Frozen language models already "know" language. The challenge: injecting visual perception without destabilizing what works. Naive approaches — concatenating image tokens into the LM input, fine-tuning the whole model — either underperform or catastrophically forget. Flamingo's answer: surgically insert new cross-attention layers between frozen ones, and gate them to zero at init so the model starts identical to the original LM.

## Concept

Three mechanisms in sequence. **Perceiver Resampler**: a learned module that compresses variable-size visual features (e.g., from a ViT) into a fixed number of visual tokens using cross-attention — this decouples image resolution from the LM's context window. **Gated Cross-Attention Dense (Gated X-Attention)**: new layers inserted between frozen LM blocks. The language hidden state queries the visual tokens via cross-attention, and the result is multiplied by `tanh(α)` where `α` is a learnable scalar initialized near zero. **Interleaved insertion**: frozen LM block → gated x-attention → frozen LM block → gated x-attention, etc. Only the Perceiver Resampler, gated x-attention layers, and a new per-layer LN are trained. Everything else stays frozen. The zero-init gate means at step 0 the model behaves exactly like the original LM; visual influence ramps up gradually during training.

## Demo

Build a minimal gated cross-attention layer in PyTorch. Initialize `α = 0.0`, run a forward pass, print the gate value and output norm — confirm the output is identical to skipping the layer. Then simulate 10 "training steps" by manually incrementing `α` and printing how the visual signal's contribution grows. Observable output at every step: gate value (`tanh(α)`), L2 norm of cross-attention output, L2 norm of residual after gating. This confirms the mechanism — zero gate = identity, gate opens = visual information flows.

## Guided Exercise

- **Easy**: Change `α` initialization to `2.0` instead of `0.0`. Run the demo. Observe and explain what happens to output stability in the first few steps. Print the gate trajectory.
- **Medium**: Implement the Perceiver Resampler — a cross-attention module that takes `(N_varied, D)` visual features and produces `(M_fixed, D)` tokens where `M` is a hyperparameter. Print input shape, output shape, and confirm M is invariant to N.
- **Hard**: Construct a 4-layer frozen "LM" (linear layers + ReLU), insert gated x-attention after layers 1 and 3, attach a Perceiver Resampler, run a forward pass, and print `requires_grad` status and gradient norms for every parameter. Confirm only the inserted modules receive gradients.

## Use It

The "freeze existing, gate in the new" pattern appears outside VLMs. In GTM enrichment pipelines, you often have a working scoring model and need to fuse a new signal source (intent data, technographics) without retraining everything. The architectural analog: keep the frozen model, add a gated cross-attention to the new signal, initialize the gate to zero, train only the new path. This is the mechanism behind adapter-style enrichment — it maps to **Zone 2 (Enrichment)** in the GTM topic map: structured fusion of new data signals into existing scoring/routing logic without destabilizing production weights.

## Ship It

**Assessment hook**: Given a 6-layer LM with gated x-attention inserted after layers 1, 3, and 5, predict (a) which parameters have non-zero gradients after a backward pass, (b) what the output looks like at initialization vs. after 1000 steps of training, and (c) why removing the gate (directly adding cross-attention output to the residual) causes training instability. Write a short explanation and include a runnable code snippet that demonstrates your prediction. GTM redirect: this adapter-gating mechanism is foundational for Zone 2 enrichment — specifically the pattern of fusing new signal sources into frozen production models without retraining.