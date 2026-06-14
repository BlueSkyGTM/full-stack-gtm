## Ship It

The first decision in production is candidate set size. The cross-encoder processes each (query, candidate) pair through a full transformer forward pass, so latency scales linearly with the number of candidates. On a single CPU, `ms-marco-MiniLM-L-6-v2` processes roughly 10–20 pairs per second. On a GPU (T4 or better), that jumps to 200–500 pairs per second at batch size 32. For a latency budget of 500ms on CPU, you can afford roughly 5–10 candidates. On GPU, you can handle 100–200 candidates in the same window.

```python
import time
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "account scoring for B2B SaaS ICP"
candidates = [f"Company {i} provides B2B software solutions for the {sector} industry."
              for i, sector in enumerate(["HR", "FinTech", "DevOps", "Sales", "Marketing",
              "Healthcare", "Education", "Logistics", "Retail", "RealEstate"] * 20)]

for batch_start in range(0, len(candidates), 50):
    batch = candidates[batch_start:batch_start + 50]
    pairs = [(query, c) for c in batch]
    start = time.time()
    scores = model.predict(pairs)
    elapsed = time.time() - start
    print(f"Candidates: {len(batch):4d}  Time: {elapsed:.3f}s  "
          f"Per-pair: {elapsed/len(batch)*1000:.1f}ms")
```

Output (CPU, will vary by machine):

```
Candidates:   50  Time: 4.821s  Per-pair: 96.4ms
Candidates:   50  Time: 4.712s  Per-pair: 94.2ms
Candidates:   50  Time: 4.855s  Per-pair: 97.1ms
Candidates:   50  Time: 4.698s  Per-pair: 94.0ms
```

At ~95ms per pair on CPU, 200 candidates costs roughly 19 seconds. That is acceptable for a batch job processing an account list overnight. It is not acceptable for real-time personalization in a sales chat. Know your latency budget before choosing N.

Model selection is the second decision. `ms-marco-MiniLM-L-6-v2` has 6 transformer layers and 22M parameters. Larger options like `cross-encoder/stsb-roberta-large` have 24 layers and 355M parameters — roughly 3-4x the latency with modestly better accuracy on most benchmarks. For GTM use cases where the candidate descriptions are short (company about pages, 1-2 sentences), MiniLM is almost always sufficient. For longer documents (full case studies, multi-paragraph product docs), the larger model's additional capacity helps. Benchmark both on your data before committing.

The third decision — and the one that causes the most production bugs — is thresholding. Cross-encoder scores are uncalibrated logits, not probabilities. A score of 5.0 means "relevant" *relative to the other candidates in this batch*, not "relevant with probability e^5 / (1 + e^5)." If you feed a different set of candidates, the same document can produce a different score because the model's internal calibration shifts with the input distribution. Do not set a hardcoded threshold like "score > 3.0 means fit" without validating it on a held-out set of labeled examples. Instead, use rank-based selection: "take the top 10 accounts from the reranked list" is stable across batches. "Take all accounts scoring above 3.0" is not.

Caching strategies help when you rerank the same corpus repeatedly. If your account list changes weekly but your ICP query is stable, cache the cross-encoder scores keyed by (query_hash, account_description_hash). If the query changes per rep or per campaign, caching is less useful because every new query invalidates the cache. In that case, focus on reducing N — a tighter stage-one filter is cheaper than a faster cross-encoder.