## Ship It

Shipping a vision-language model in production requires choosing the objective family that matches your inference requirements. If you need real-time ranking—matching product images to search queries, filtering user-generated screenshots against policy text—a contrastive model like CLIP gives you sub-millisecond cosine similarity lookups against a pre-computed embedding index. You embed your corpus once, store vectors in a vector database, and query with a single forward pass through the text encoder at request time. The projection layer dimensions (512 for CLIP ViT-B/32) determine your storage cost: 512 floats × 4 bytes × corpus size.

If you need generation—describing product images, writing alt-text at scale, producing image-conditioned marketing copy—you need a generative model. The cost profile is different: each inference call runs a full autoregressive decode pass, which is orders of magnitude more expensive than a dot product. LLaVA-style models also require a vision encoder forward pass per image plus the projection step before generation begins.

```python
import time
import torch
import torch.nn.functional as F

BATCH_SIZES = [1, 10, 100, 1000]
EMBED_DIM = 512

corpus = torch.randn(10000, EMBED_DIM)
corpus_norm = F.normalize(corpus, dim=-1)

print("Contrastive Retrieval Latency (CLIP-style cosine similarity):")
print(f"Corpus size: 10,000 embeddings × {EMBED_DIM} dims")
print(f"Storage: {10000 * EMBED_DIM * 4 / 1024 / 1024:.1f} MB (float32)")
print("-" * 55)

for bs in BATCH_SIZES:
    queries = torch.randn(bs, EMBED_DIM)
    start = time.perf_counter()
    q_norm = F.normalize(queries, dim=-1)
    sims = q_norm @ corpus_norm.T
    top_k = sims.topk(5, dim=-1)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  batch={bs:>5}  →  {elapsed:>7.2f} ms  →  {elapsed/bs:>6.3f} ms/query")

print("\nGenerative Latency Estimate (LLaVA-style, 50 token decode):")
for bs in BATCH_SIZES:
    tokens_per_sec = 150 if bs <= 10 else 80
    gen_time = (50 / tokens_per_sec) * 1000
    print(f"  batch={bs:>5}  →  ~{gen_time:>7.0f} ms  →  ~{gen_time/bs:>6.0f} ms/query")

print("\nContrastive is ~1000x faster for ranking-only workloads.")
print("Generative is required when you need novel text output, not ranking.")
```

Output:

```
Contrastive Retrieval Latency (CLIP-style cosine similarity):
Corpus size: 10,000 embeddings × 512 dims
Storage: 19.5 MB (float32)
-------------------------------------------------------
  batch=     1  →     0.12 ms  →  0.120 ms/query
  batch=    10  →     0.15 ms  →  0.015 ms/query
  batch=   100  →     0.31 ms  →  0.003 ms/query
  batch=  1000  →     2.84 ms  →  0.003 ms/query

Generative Latency Estimate (LLaVA-style, 50 token decode):
  batch=     1  →     333 ms  →    333 ms/query
  batch=    10  →     333 ms  →     33 ms/query
  batch=   100  →     625 ms  →      6 ms/query
  batch=  1000  →     625 ms  →      1 ms/query

Contrastive is ~1000x faster for ranking-only workloads.
Generative is required when you need novel text output, not ranking.
```

For GTM systems specifically, this means: if your "write at scale" pipeline needs to match prospects to relevant case studies before composing outreach, use contrastive retrieval (fast, cheap) for the matching step and reserve generative models for the writing step. This mirrors how production vision-language systems use CLIP for retrieval and LLaVA for description—they do not pay generative costs for ranking operations.

The temperature parameter matters in production too. If your contrastive retriever returns low-similarity scores across the board (everything looks equally unrelated), the model may be operating at too high a temperature. If it returns one result at 0.99 and everything else at 0.01, the temperature may be too low, causing the model to ignore meaningful second-tier matches. Monitor the similarity distribution shape, not just the top-1 score.