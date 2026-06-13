# Self-Supervised Vision — SimCLR, DINO, MAE

---

## Hook

The ImageNet-labeled dataset is ~1M images. The web has trillions. Self-supervised vision closes that gap by learning visual representations from raw pixels alone — no human labels required. Three architectures dominate: SimCLR (contrastive), DINO (self-distillation), and MAE (masked reconstruction). Each encodes a different assumption about what it means to "understand" an image.

---

## Concept

**Three paradigms, one goal: learn representations without labels.**

**SimCLR** — Contrastive learning pulls augmented views of the same image together in embedding space while pushing different images apart. The mechanism: (1) stochastic augmentation pair, (2) shared encoder, (3) projection head, (4) NT-Xent loss (normalized temperature-scaled cross-entropy). Key insight: the projection head discards task-irrelevant information so the encoder keeps only what generalizes.

**DINO** — Self-distillation with no labels. A student network matches the output distribution of an EMA-updated teacher. Centering prevents collapse; sharpening forces discriminative features. Emergent property: attention maps spontaneously segment objects without any segmentation supervision.

**MAE** — Masked Autoencoders for Vision. Mask 75% of image patches, encode only the visible 25%, reconstruct the full image via a lightweight decoder. Asymmetric encoder-decoder design means inference only runs on visible patches — compute scales with mask ratio.

Comparison axis: collapse avoidance mechanism (negative samples vs. EMA centering vs. reconstruction target), compute cost at pretrain, and transfer quality to downstream tasks.

---

## Demo

1. **SimCLR NT-Xent loss from scratch**: Implement normalized temperature-scaled cross-entropy on synthetic embedding pairs. Print loss values for positive pairs (should decrease) vs. negative pairs (should increase). Confirms the contrastive mechanism works as described.

2. **MAE patch masking visualization**: Given a small image tensor, implement random patch masking at 75% ratio, reconstruct with a linear layer, print MSE between original and reconstruction. Observes the reconstruction target directly.

3. **DINO teacher EMA update**: Implement exponential moving average parameter update from student to teacher with centering operation on logits. Print parameter drift over steps. Confirms slow-drift / fast-drift dynamics.

---

## Use It

**GTM Redirect — Zone 1 Enrichment (Visual Firmographics)**

Self-supervised vision models produce general-purpose visual embeddings. In GTM, those embeddings power logo detection, screenshot classification, and visual firmographic signals (technology stack from landing page screenshots, brand presence from social profiles). The downstream task is always the same: encode an image, compare embeddings to a reference set, score similarity.

Foundational for Zone 1 enrichment pipelines that process company logos, product screenshots, or web presence imagery. [CITATION NEEDED — concept: vision-based firmographic enrichment in GTM workflows]

Exercise hooks:
- *(Easy)* Load a pretrained DINO model, encode two images of the same company logo (different augmentations), compute cosine similarity. Print similarity score.
- *(Medium)* Build a logo-matching function: encode a query logo, compare against a gallery of 10 company logo embeddings, return top-3 matches with scores.
- *(Hard)* Implement a mini enrichment pipeline: given a URL screenshot and a set of technology logo references, detect which logos appear using embedding similarity with a tuned threshold.

---

## Ship It

**Production deployment constraints for self-supervised vision backbones.**

- **Model selection**: SimCLR and DINO produce frozen encoders that transfer well to classification/detection. MAE requires fine-tuning (linear probe underperforms). Choose based on downstream task.
- **Inference cost**: DINO ViT-S processes ~150 images/sec on a single GPU at 224px. Batch prediction, not real-time, for enrichment workloads.
- **Embedding storage**: Vision embeddings are typically 384–768 dimensional float32 vectors. Storage and nearest-neighbor search (FAISS, ScaNN) become the bottleneck at scale, not encoding.
- **Augmentation pipeline at inference**: No augmentation needed — a single center crop suffices for embedding extraction. Augmentation is a training-time mechanism only.

Exercise hooks:
- *(Easy)* Export embeddings from a pretrained model to numpy, save to `.npy`, load and verify shape matches.
- *(Medium)* Build a batch encoding script: iterate over a directory of images, encode all, write embeddings + filenames to a single file. Print total count and average encoding time.
- *(Hard)* Implement approximate nearest-neighbor search over a gallery of 1,000 image embeddings. Given a query, return top-5 in under 10ms. Profile the search step.

---

## Assess

- Explain why SimCLR requires a projection head separate from the encoder, and what happens to representation quality if you remove it.
- Compare collapse-avoidance mechanisms across all three methods: what keeps each from outputting a constant vector?
- Implement NT-Xent loss given a batch of embeddings and a temperature parameter. Print the loss for a known configuration where the answer is calculable by hand.
- Given a downstream task (logo detection on company websites), justify which of the three pretraining methods you would choose and what fine-tuning strategy you would apply.
- Diagnose a failing DINO training run where teacher and student outputs are identical from epoch 1. Identify the most likely configuration error.

---

*Learning objectives (for docs/en.md — not section prose):*
1. Implement the NT-Xent contrastive loss function and verify its behavior on controlled embedding pairs.
2. Compare the collapse-avoidance mechanisms of SimCLR, DINO, and MAE with specific reference to each architecture's design.
3. Configure a masked autoencoder patch-masking pipeline with a specified mask ratio and measure reconstruction error.
4. Evaluate which self-supervised pretraining strategy is appropriate for a given downstream vision task, justified by transfer-learning properties.
5. Extract and compare visual embeddings from a pretrained self-supervised model for a nearest-neighbor retrieval task.