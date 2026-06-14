## Ship It

Attention is O(n²) in sequence length. The Q·Kᵀ matrix multiply produces an n×n score matrix, and the attention weights are n×n dense. For a sequence of 2,048 tokens, that is 4.2 million floats just for the weight matrix. At 8,192 tokens, it is 67 million. This is the binding production constraint — not memory for the weights alone, but the compute to fill them and the memory to hold activations for backpropagation.

For GTM pipelines processing thousands of account records, this limits how much context you can feed a single attention call. If you are cross-referencing enrichment signals across 500 accounts each with 20 data points, the sequence length is 10,000 — the attention matrix alone is 100M entries. In practice, you hit a wall where batching becomes impossible and latency spikes. The engineering response is triage: use attention when you genuinely need cross-referencing between every pair of items, and use simpler retrieval when you do not.

Cosine similarity against pre-computed embeddings is O(n·k) where k is the number of retrieved neighbors — linear in corpus size, not quadratic. For most enrichment lookups ("find similar accounts," "match this company to a taxonomy"), cosine similarity over a vector database is sufficient and orders of magnitude faster. Attention earns its cost when the relationships are bidirectional and contextual: the relevance of signal A to account B depends on signal C, which depends on signal D. That mutual dependency graph is what attention computes, and it is what flat retrieval cannot capture.

Here is the benchmark showing the wall:

```python
import time

print("Attention runtime vs sequence length:")
print(f"{'seq_len':>8}  {'time (ms)':>10}  {'ratio vs 32':>12}  {'weight matrix MB':>18}")
print("-" * 55)

base_time = None
for seq_len in [32, 128, 512, 2048]:
    Q = np.random.randn(seq_len, 64)
    K = np.random.randn(seq_len, 64)
    V = np.random.randn(seq_len, 64)

    start = time.time()
    for _ in range(5):
        scaled_dot_product_attention(Q, K, V)
    elapsed = (time.time() - start) / 5

    if base_time is None:
        base_time = elapsed

    matrix_mb = (seq_len * seq_len * 8) / (1024 * 1024)
    ratio = elapsed / base_time

    print(f"{seq_len:>8}  {elapsed*1000:>10.2f}  {ratio:>11.1f}x  {matrix_mb:>17.1f}")
```

You will see the runtime roughly quadruple each time the sequence length doubles. At 2,048 tokens the weight matrix alone is 32 MB — and that is per attention head, per layer, per item in the batch. This is why production transformers cap context length and why retrieval-augmented generation (RAG) exists: retrieve a small relevant chunk with O(n) cosine similarity, then run attention only over that chunk.