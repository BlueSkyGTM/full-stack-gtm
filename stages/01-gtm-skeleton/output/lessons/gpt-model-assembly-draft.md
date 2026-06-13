# GPT Model Assembly

## Hook
A GPT model is a stack of identical transformer blocks sandwiched between an embedding layer and a linear output head. This beat opens with a minimal config object and asks: what does each parameter control, and what breaks if you remove one layer?

## Concept
Covers the five mechanical stages of a forward pass: tokenization → token embedding + positional encoding → N stacked transformer blocks (layernorm → attention → residual → layernorm → FFN → residual) → final layernorm → linear projection to vocabulary logits. Each stage is defined by its input shape, operation, and output shape. The key mechanism is that transformer blocks are weight-isolated but architecturally identical — the model's capacity scales with block count (depth), hidden dimension (width), and attention head count (granularity).

## Demo
Build a 2-layer GPT from scratch in PyTorch. Print tensor shapes at every stage to confirm dimension flow. Run a single token through and print the final logits shape. Then ablate one component (positional encoding, one attention head, the FFN) and print the output difference to show what each piece contributes.

## Use It
Map model assembly parameters to deployment decisions: how `num_layers` and `hidden_dim` affect latency and VRAM, why head count matters for batching, and what changes when you go from GPT-2-small (124M) to GPT-2-xl (1.5B). GTM redirect: this is foundational for Zone 1 (ICP & AI Enrichment) — when selecting a model for classification or enrichment pipelines, you need to predict whether a given model size fits your latency budget and inference hardware.

## Ship It
Write a config-driven factory function that accepts a dict of hyperparameters and returns an initialized GPT model. Print parameter count and per-layer memory estimate. Exercise hooks: (easy) modify the config to double the model size and confirm parameter count scales as expected; (medium) add a method that estimates VRAP requirements given batch size and sequence length; (hard) implement gradient checkpointing on the block stack and measure memory reduction.

## Evaluation
Three questions grounded in the demo code: (1) given a config, calculate the total parameter count by hand and verify against the model output; (2) predict the output logits shape when `vocab_size` or `context_length` changes; (3) explain which component's removal causes the largest output divergence and why. All questions test whether the practitioner can reason about the architecture without running code.