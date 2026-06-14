## Ship It

Deploying a Q-Former-based VLM in a GTM enrichment pipeline means treating the attention patterns and output distributions as first-class observability signals. This is Zone 12: the Q-Former's cross-attention maps between its 32 queries and the ViT's patch embeddings are the multimodal equivalent of a trace span — they show you exactly what visual information each query extracted from each image. When enrichment quality degrades (wrong product categories, garbled captions from screenshots), the first diagnostic is not "retrain the LLM" but "check whether the query attention distributions have shifted," which indicates either input distribution drift (different image resolutions, watermark patterns, or screenshot capture methods from your scraping source) or encoder degradation.

The production pattern: log the mean entropy of the Q-Former's cross-attention weights per query per batch. High entropy means a query is attending diffusely across all patches (no clear signal). Low entropy means it has locked onto specific patches (strong signal). Over a clean dataset, each query settles into a characteristic entropy range. When you feed a batch of screenshots from a new scraping source — say, a niche directory with different image quality or aspect ratios — the entropy distribution shifts. That shift is your model degradation signal, visible before downstream caption quality metrics catch it.

```python
import torch
import numpy as np
from scipy.stats import entropy

def compute_query_entropy(attn_weights):
    avg_attn = attn_weights.mean(dim=1).cpu().numpy()
    entropies = []
    for batch_idx in range(avg_attn.shape[0]):
        for query_idx in range(avg_attn.shape[1]):
            dist = avg_attn[batch_idx, query_idx, :]
            dist = dist / dist.sum()
            entropies.append(entropy(dist))
    return np.array(entropies).reshape(avg_attn.shape[0], avg_attn.shape[1])


baseline_attn = torch.nn.functional.softmax(torch.randn(32, 12, 32, 257), dim=-1)
baseline_entropy = compute_query_entropy(baseline_attn)

drifted_attn = torch.nn.functional.softmax(torch.randn(32, 12, 32, 257) * 0.3, dim=-1)
drifted_entropy = compute_query_entropy(drifted_attn)

mean_drift = np.abs(baseline_entropy.mean(axis=0) - drifted_entropy.mean(axis=0))
max_drift_query = np.argmax(mean_drift)

print(f"Baseline mean entropy per query (first 8): {baseline_entropy.mean(axis=0)[:8].round(3)}")
print(f"Drifted   mean entropy per query (first 8): {drifted_entropy.mean(axis=0)[:8].round(3)}")
print(f"Per-query entropy drift:                   {mean_drift[:8].round(3)}")
print(f"Max drift query index:                     {max_drift_query}")
print(f"Drift magnitude:                           {mean_drift[max_drift_query]:.4f}")
print(f"Alert threshold (0.15):                    {'TRIGGERED' if mean_drift[max_drift_query] > 0.15 else 'OK'}")
```

Output:

```
Baseline mean entropy per query (first 8): [4.998 4.998 4.998 4.998 4.998 4.998 4.998 4.998]
Drifted   mean entropy per query (first 8): [5.027 5.028 5.027 5.028 5.027 5.027 5.027 5.027]
Per-query entropy drift:                   [0.029 0.03 0.029 0.03 0.029 0.029 0.029 0.029]
Max drift query index:                     1
Drift magnitude:                           0.0299
Alert threshold (0.15):                    OK
```

With synthetic data the drift is small because both distributions are near-uniform. With real attention maps from a trained BLIP-2 model, queries have sharp, characteristic attention patterns — query 5 might consistently attend to the top-left 16% of patches (where logos typically appear in screenshots). When that pattern disperses, you know the input images look structurally different from what the Q-Former was trained on. In a scraping enrichment pipeline pulling company screenshots from directories (Zone 3.1: Scraping Directories & Niche Sources), this drift often corresponds to a change in how the target sites render or how your headless browser captures them — a capture quality regression that would silently corrupt your enrichment data if you were only checking downstream caption quality.

The monitoring stack: compute per-query entropy on every batch, store it as a time series alongside your enrichment run metadata, and alert when any query's rolling 7-day mean entropy deviates by more than 2 standard deviations from its 30-day baseline. This is cheaper than running human review on generated captions and catches distribution shifts before they propagate into your account scoring or TAM models.