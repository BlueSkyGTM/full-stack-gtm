## Ship It

Shipping a ViT-based enrichment pipeline into a GTM stack means making three decisions: which pretrained model to use, how to batch inference, and where to store the embeddings.

**Model selection.** For logo and screenshot embedding, SigLIP 2 (Google, 2024) or DINOv2 (Meta, 2023) are the current production defaults. SigLIP 2 uses a contrastive image-text training objective, which means its embeddings align with text descriptions — you can search "blue fintech logo" and retrieve visually matching images. DINOv2 uses self-supervised training on 142 million images, producing embeddings that cluster well for visual similarity even without text labels. For pure visual similarity (logo matching, screenshot clustering), DINOv2 typically outperforms. For tasks that benefit from text-image alignment (describing what a landing page shows), SigLIP 2 is the better choice. [CITATION NEEDED — concept: relative performance of SigLIP 2 vs DINOv2 on GTM logo-matching benchmarks]

**Batching and inference.** ViT inference is GPU-bound. A single ViT-B forward pass on one 224×224 image takes roughly 5ms on an A10G. Batching 64 images reduces per-image cost to under 1ms but requires holding 64 × 3 × 224 × 224 × 4 bytes = 38MB of input tensors in GPU memory — trivial. The bottleneck is usually data loading: scraping logos at scale, resizing them to 224×224, normalizing pixel values, and transferring to GPU. A production pipeline should use a DataLoader with multiple workers and pinned memory to keep the GPU saturated.

**Embedding storage.** ViT-B produces 768-dimensional float32 embeddings — 3KB per image. For a database of 100,000 company logos, that is 300MB of vectors. PostgreSQL with the `pgvector` extension handles this at billion-scale with approximate nearest neighbor (ANN) search via HNSW or IVF indexes. For smaller scale (under 1M vectors), `pgvector` with exact search is sufficient and avoids the complexity of a dedicated vector database. Qdrant or Milvus becomes necessary when you exceed 10M vectors or need sub-10ms query latency at scale. [CITATION NEEDED — concept: pgvector vs Qdrant performance thresholds for GTM enrichment workloads]

The pipeline architecture is straightforward: a worker scrapes logos or screenshots, a ViT model produces embeddings, those embeddings land in a vector store, and a query service performs similarity search against the store. This is foundational infrastructure for Zone 3 signal enrichment — if the visual asset carries no GTM signal (a random stock photo, a blank page), the embedding will still be produced but will not be useful. The signal lives in the relationship between embeddings, not in any single embedding itself.

```python
import torch
import torch.nn as nn
import time

torch.manual_seed(42)

class MiniViTEmbedder(nn.Module):
    def __init__(self, patch_size=16, hidden_dim=256, num_layers=2, num_heads=4):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        num_patches_224 = (224 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches_224 + 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B = x.shape[0]
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)

        seq_len = tokens.shape[1]
        if seq_len != self.pos_embed.shape[1]:
            pos = torch.randn(B, seq_len, self.hidden_dim, device=x.device)
        else:
            pos = self.pos_embed.expand(B, -1, -1)

        tokens = tokens + pos
        encoded = self.encoder(tokens)
        cls_output = self.norm(encoded[:, 0])
        return cls_output

model = MiniViTEmbedder(patch_size=16, hidden_dim=256, num_layers=2)
model.eval()

batch_size = 32
images = torch.randn(batch_size, 3, 224, 224)

with torch.no_grad():
    start = time.time()
    embeddings = model(images)
    elapsed = time.time() - start

print(f"Batch size:        {batch_size}")
print(f"Image size:        224 x 224")
print(f"Patch size:        16 x 16")
print(f"Embedding dim:     {embeddings.shape[1]}")
print(f"Output shape:      {embeddings.shape}")
print(f"Inference time:    {elapsed * 1000:.1f} ms")
print(f"Per-image time:    {elapsed * 1000 / batch_size:.2f} ms")
print(f"Memory per embed:  {embeddings.shape[1] * 4} bytes ({embeddings.shape[1] * 4 / 1024:.1f} KB)")

batch_sizes = [1, 8, 16, 32, 64, 128]
print(f"\n{'Batch':>6} {'Time (ms)':>10} {'Per-img (ms)':>14} {'Throughput':>15}")
print("-" * 48)

for bs in batch_sizes:
    imgs = torch.randn(bs, 3, 224, 224)
    with torch.no_grad():
        times = []
        for _ in range(5):
            start = time.time()
            _ = model(imgs)
            times.append((time.time() - start) * 1000)
    avg_ms = sum(times[1:]) / len(times[1:])
    per_img = avg_ms / bs
    throughput = 1000 / per_img
    print(f"{bs:>6} {avg_ms:>10.1f} {per_img:>14.2f} {throughput:>12.0f}/s")
```