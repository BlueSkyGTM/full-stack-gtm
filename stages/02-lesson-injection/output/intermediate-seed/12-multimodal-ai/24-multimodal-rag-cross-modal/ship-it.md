## Ship It

Production cross-modal RAG needs the same observability discipline as any retrieval pipeline — but the metrics must decompose by modality. Three signals tell you the system is degrading before users complain.

**Modality bias** is the most common production failure. Embedding scales drift: text-to-text cosine similarities cluster around 0.7–0.9 while text-to-image similarities cluster around 0.2–0.4. Without normalization, text results dominate every ranking and images effectively disappear. You detect this by logging the modality distribution of top-k results across queries and alerting when any modality's share drops below a threshold.

**Recall drift per modality** is the model degradation signal. If you have a held-out evaluation set with known-relevant image and text results per query, you compute recall@k separately for each modality. A drop in image recall while text recall holds steady indicates the vision encoder is encountering out-of-distribution images — perhaps a new competitor's deck uses a visual style the model has not seen.

**Chunk boundary failures** appear as recall drops on queries that should return table or diagram content. When a table is split across two image chunks, neither chunk contains enough context to match the query. You detect this by tracking per-source recall: if recall drops for one specific document but holds for others, the problem is chunking, not the model.

```python
import json
from collections import Counter
from datetime import datetime

def log_retrieval_event(query, results, filter_modality="any"):
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "filter": filter_modality,
        "top_k_modalities": [item["modality"] for _, item in results],
        "top_k_scores": [round(score, 4) for score, _ in results],
        "top_score": results[0][0] if results else 0.0,
        "modality_split": dict(Counter(item["modality"] for _, item in results)),
    }

def compute_bias_metrics(events, window=50):
    recent = events[-window:]
    modality_shares = Counter()
    score_by_modality = {}

    for event in recent:
        for mod, score in zip(event["top_k_modalities"], event["top_k_scores"]):
            modality_shares[mod] += 1
            score_by_modality.setdefault(mod, []).append(score)

    total = sum(modality_shares.values())
    print(f"--- Modality Distribution (last {len(recent)} queries) ---")
    for mod in sorted(modality_shares):
        share = modality_shares[mod] / total
        avg_score = np.mean(score_by_modality[mod])
        flag = " *** BIAS ALERT" if share < 0.15 else ""
        print(f"  {