## Ship It

In production GTM systems, you rarely train a projection layer from scratch. You use a pretrained embedding model (OpenAI `text-embedding-3-small`, Cohere `embed-english-v3`, or an open-weight model like `bge-large`) that already maps text into a shared space. The projection is implicit — the model's final layer does it. What you control is the embedding pipeline: what text you embed, how you chunk case studies, what metadata you attach, and how you set retrieval thresholds.

The production checklist for a contrastive-alignment RAG system in a GTM context:

1. Embed your knowledge assets (case studies, product docs, battle cards, objection handlers) into a vector store. Document the chunking strategy — a 2000-word case study embedded as a single vector loses signal; chunked by section (problem, solution, results) and embedded separately preserves it.

2. Embed prospect context using the same model. This is the alignment step — both spaces use the same encoder, so the projection is handled by the shared model. If you use different encoders for prospects (e.g., a firmographic encoder) and case studies (a text encoder), you need an explicit projection layer trained on paired data.

3. Set retrieval thresholds empirically. Start with `top_k=3` and a minimum cosine similarity of 0.7. Log retrieval results against conversion outcomes. If the same case study surfaces for every prospect, your threshold is too low (temperature too high — no discrimination). If prospects frequently get zero results, your threshold is too high (temperature too low — over-discrimination).

4. Monitor for distribution drift. As you add case studies, the vector space gets denser. Retrieval scores that were competitive at 50 assets become noisy at 500. Periodically re-embed with newer models and re-benchmark retrieval quality against known-good prospect→case-study pairings.

The failure mode to watch for is the same one from the contrastive loss: temperature too high means no discrimination. In GTM terms, this shows up as "every prospect gets the same three case studies." The fix is either tightening the similarity threshold (lower temperature) or increasing the embedding dimension (larger projection space) to capture finer-grained distinctions. The opposite failure — "prospects get zero results" — means your threshold is too tight. Loosen it, or re-chunk your knowledge assets into smaller, more specific pieces that match narrower prospect signals.

Here is a production-shaped retrieval example using real embeddings via sentence-transformers. It embeds case study chunks and prospect descriptions with the same model, then retrieves:

```python
import subprocess
import sys

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers", "-q"])
    from sentence_transformers import SentenceTransformer

import torch
import torch.nn.functional as F

model = SentenceTransformer("all-MiniLM-L6-v2")

case_studies = [
    "We reduced API latency by 80% for a fintech startup using edge caching and connection pooling.",
    "A global bank achieved SOC 2 Type II compliance in 90 days using our automated audit trail.",
    "Mid-market retailer unified inventory across 200 stores with real-time sync, cutting stockouts by 40%.",
    "Healthcare provider deployed HIPAA-compliant messaging serving 500k patients with zero downtime.",
    "SaaS company scaled from 10k to 1M users without rearchitecting, using our horizontal sharding layer.",
]

prospects = [
    "Series B SaaS company, 80 engineers, hitting API rate limits during peak hours.",
    "Enterprise healthcare network, 12 hospitals, needs HIPAA-compliant patient communication.",
    "Regional retail chain, 150 locations, struggling with inventory discrepancies across warehouses.",
]

case_embeddings = model.encode(case_studies, convert_to_tensor=True)
prospect_embeddings = model.encode(prospects, convert_to_tensor=True)

case_embeddings = F.normalize(case_embeddings, dim=-1)
prospect_embeddings = F.normalize(prospect_embeddings, dim=-1)

similarity_matrix = prospect_embeddings @ case_embeddings.T

for i, prospect in enumerate(prospects):
    scores = similarity_matrix[i]
    top_k = min(3, len(case_studies))
    top_scores, top_indices = torch.topk(scores, top_k)
    print(f"Prospect: {prospect}")
    for score, idx in zip(top_scores, top_indices):
        print(f"  [{score:.3f}] {case_studies[idx]}")
    print()
```

Run this in a terminal with `sentence-transformers` installed. The output shows semantic retrieval in action — the API-scaling prospect gets the latency case study, the healthcare prospect gets the HIPAA case study. No projection layer needed because both texts use the same encoder. This is the implicit projection that makes Zone 19 RAG systems work in practice. When you need cross-modal alignment (prospect signals that aren't text — behavioral data, technographic signals — mapped against text case studies), you're back to the explicit projection layer you built earlier.