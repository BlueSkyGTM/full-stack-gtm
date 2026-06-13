# Projection Layer for Modality Alignment

## Hook

You have image embeddings (512-d) from ResNet and text embeddings (768-d) from BERT. They describe the same thing—but live in incompatible vector spaces. The projection layer is the narrow bridge that forces them to share coordinates. Without it, "dog" and a photo of a dog have zero mathematical relationship. With it, cosine similarity becomes a meaningful cross-modal search mechanism.

## Concept

**Mechanism**: A learned linear (or shallow MLP) transformation that maps modality-specific embeddings into a shared lower-dimensional space. The projection is trained via contrastive loss—paired examples (e.g., image-caption) are pulled together while unpaired examples are pushed apart. The temperature parameter in the softmax controls how sharply the model discriminates between positive and negative pairs. CLIP uses a symmetric contrastive objective over both modalities simultaneously; ALIGN scales this to noisy web-scraped pairs. The projection dimension is a bottleneck by design—it forces the model to discard modality-specific noise and retain only cross-modal semantics.

**Key variables**: projection dimension (typically 128–512), temperature (learned or fixed, range 0.01–0.1), batch size (determines negative sample count), and whether the projection is linear (`Wx + b`) or a two-layer MLP with GELU (as in CLIP).

**Failure modes**: projection dimension too large = modality-specific noise leaks through; temperature too low = gradient collapse to hard negatives; temperature too high = uniform distribution, no discrimination.

## Demo

Build a minimal dual-encoder projection system. Two random "modalities" (simulating different input spaces), a projection layer for each, contrastive loss with tunable temperature, and a training loop that prints alignment metrics (mean cosine similarity for positive pairs vs negative pairs) at each step. Observable output: before training, positive and negative similarities overlap. After training, they separate.

```
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

modality_a_dim = 64
modality_b_dim = 128
projection_dim = 32
batch_size = 16
temperature = 0.07
epochs = 200

proj_a = nn.Linear(modality_a_dim, projection_dim)
proj_b = nn.Linear(modality_b_dim, projection_dim)

optimizer = torch.optim.Adam(list(proj_a.parameters()) + list(proj_b.parameters()), lr=0.001)

for epoch in range(epochs):
    a_raw = torch.randn(batch_size, modality_a_dim)
    b_raw = torch.randn(batch_size, modality_b_dim)
    for i in range(batch_size):
        b_raw[i] = a_raw[i, :modality_b_dim] * 0.3 + b_raw[i] * 0.7

    a_proj = F.normalize(proj_a(a_raw), dim=-1)
    b_proj = F.normalize(proj_b(b_raw), dim=-1)

    logits = a_proj @ b_proj.T / temperature
    labels = torch.arange(batch_size)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0 or epoch == epochs - 1:
        with torch.no_grad():
            pos_sim = torch.diag(a_proj @ b_proj.T).mean().item()
            neg_sim = (a_proj @ b_proj.T).fill_diagonal_(0).sum() / (batch_size * (batch_size - 1))
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Pos sim: {pos_sim:.3f} | Neg sim: {neg_sim.item():.3f}")

with torch.no_grad():
    a_test = torch.randn(1, modality_a_dim)
    b_test = torch.randn(5, modality_b_dim)
    b_test[0] = a_test[0, :modality_b_dim] * 0.3 + b_test[0] * 0.7
    sim = F.normalize(proj_a(a_test), dim=-1) @ F.normalize(proj_b(b_test), dim=-1).T
    print(f"\nRetrieval test — similarities: {[f'{s:.3f}' for s in sim[0].tolist()]}")
    print(f"Correct match rank: {(sim[0] >= sim[0][0]).sum().item()} (1 = perfect)")
```

## Use It

**GTM Redirect — Zone C Enrichment: Company-Person Embedding Matching**

Projection layers solve a specific GTM problem: you have company-level signals (firmographic vectors) and person-level signals (behavioral/profile vectors) that need to be compared in the same space. The Clay waterfall enrichment pattern depends on record matching across data sources—projection alignment is the mechanism that makes fuzzy cross-entity matching work when exact key matching fails. [CITATION NEEDED — concept: cross-entity embedding matching in enrichment pipelines]

Exercise hooks:
- **Easy**: Take two pre-computed embedding CSVs (company descriptions, technographic profiles), project both to 64-d, compute top-3 nearest neighbors, print match results.
- **Medium**: Implement temperature sweep. Run contrastive alignment at τ = 0.01, 0.07, 0.5. Print positive/negative similarity gap for each. Identify which τ produces the largest separation.
- **Hard**: Build a mini dual-encoder that aligns LinkedIn job title embeddings to company description embeddings. Evaluate with recall@5 on a held-out set. Print the recall score.

## Ship It

Implement a projection alignment pipeline for GTM data: load two embedding sets from parquet files, train projection layers with contrastive loss, serialize the trained projections, and write a matching report (top-k cross-modal matches with similarity scores) to CSV. The pipeline must accept projection dimension and temperature as CLI arguments.

Exercise hooks:
- **Easy**: Load two embedding sets, project them, compute cosine similarity matrix, save top-5 matches per record.
- **Medium**: Add early stopping—halt training when positive similarity plateaus for 10 consecutive epochs. Print the stopping epoch.
- **Hard**: Implement the full ship: CLI tool with `argparse`, configurable architecture (linear vs MLP), logging to file, and a recall@k evaluation step that prints final metrics and writes matches to CSV.

## Assess

1. Given a projection layer mapping 768-d text embeddings to 256-d space, calculate the number of trainable parameters (with and without bias). Print both counts.
2. Modify the demo: set temperature to 2.0 and run 200 epochs. Print the positive/negative similarity gap. Explain the observed collapse.
3. Implement recall@k from scratch: given a similarity matrix and ground-truth indices, compute the fraction of correct matches in the top-k. Print recall@1, recall@5, recall@10.
4. Replace the linear projection with a 2-layer MLP (GELU activation). Run the same training loop. Print the similarity gap difference between linear and MLP projections.

---

**GTM cluster referenced**: Zone C Enrichment — embedding-based record matching across heterogeneous data sources, as used in Clay waterfall patterns for cross-entity alignment.