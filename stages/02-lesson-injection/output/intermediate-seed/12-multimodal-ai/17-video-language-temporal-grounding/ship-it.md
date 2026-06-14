## Ship It

Deploying a temporal grounding pipeline in production requires the same observability discipline as any retrieval system — this is where **Zone 12 (Observability, logging, tracing)** connects. The GTM mapping is direct: just as "reply rate drift is your model degradation signal" in a sales sequence, "grounding tIoU drift is your model degradation signal" in a video search system. If average tIoU on production queries drops from 0.62 to 0.48 over two weeks, something in the pipeline has degraded — maybe the frame sampler is hitting a different video codec that produces different frame timing, or the vision encoder checkpoint was silently updated.

```python
import numpy as np
from datetime import datetime, timedelta

np.random.seed(100)

DAYS = 14
QUERIES_PER_DAY = 50

gt_spans = np.array([[10, 20], [30, 45], [5, 12], [50, 70], [15, 25]])

tiou_history = []
daily_results = []

for day in range(DAYS):
    degradation = 0.0 if day < 5 else (day - 4) * 0.025
    day_tious = []
    for q in range(QUERIES_PER_DAY):
        gt = gt_spans[q % len(gt_spans)]
        noise_scale = 2.0 + degradation * 12.0
        pred_start = max(0, gt[0] + np.random.randn() * noise_scale)
        pred_end = gt[1] + np.random.randn() * noise_scale
        if pred_end < pred_start:
            pred_start, pred_end = pred_end, pred_start
        inter = min(pred_end, gt[1]) - max(pred_start, gt[0])
        union = max(pred_end, gt[1]) - min(pred_start, gt[0])
        tiou = max(inter, 0) / max(union, 1e-9)
        day_tious.append(tiou)
    day_mean = np.mean(day_tious)
    day_recall = np.mean([1 if t >= 0.5 else 0 for t in day_tious])
    tiou_history.append(day_mean)
    daily_results.append((day, day_mean, day_recall))

date = datetime(2025, 1, 6)
print("GROUNDING PIPELINE HEALTH — Last 14 days")
print("=" * 62)
print(f"{'Date':<12} {'Avg tIoU':>8} {'R@0.5':>8} "
      f"{'Delta':>8} {'Status':>12}")
print("-" * 62)

baseline = tiou_history[0]
for day, mean_tiou, recall in daily_results:
    d = date + timedelta(days=day)
    delta = mean_tiou - baseline
    if abs(delta) < 0.03:
        status = "OK"
    elif delta < -0.10:
        status = "DEGRADED"
    elif delta < -0.05:
        status = "WARNING"
    else:
        status = "OK"
    print(f"{d.strftime('%Y-%m-%d'):<12} {mean_tiou:>8.3f} "
          f"{recall:>8.2f} {delta:>+8.3f} {status:>12}")

print("-" * 62)
recent_mean = np.mean(tiou_history[-3:])
early_mean = np.mean(tiou_history[:3])
drift = recent_mean - early_mean
print(f"\nWeek 1 avg tIoU: {early_mean:.3f}")
print(f"Week 2 avg tIoU: {recent_mean:.3f}")
print(f"Drift:          {drift:+.3f}")
if drift < -0.05:
    print("ALERT: Grounding accuracy degraded beyond threshold.")
    print("Check: frame sampler timing, encoder checkpoint hash, "
          "input video codec distribution")
elif drift > 0.05:
    print("NOTE: Grounding accuracy improved — investigate "
          "whether input distribution changed")
else:
    print("STATUS: Stable within drift threshold (±0.05)")
```

The alerting logic is straightforward: compare a rolling 3-day mean against the baseline established in the first week of deployment. A drift of more than 0.05 tIoU points triggers investigation. The diagnostic checklist is specific to video pipelines — frame timing errors (off by one frame due to codec differences), encoder checkpoint mismatches (someone swapped ViT-B for ViT-L without updating the projection layer), and input distribution shifts (more screen-share heavy calls than training data). Each of these manifests as tIoU drift before users complain about wrong results.

The production architecture for a GTM team handling call recordings at scale: a queue-based ingestion pipeline extracts temporal tokens asynchronously (one GPU job per recording), stores the token sequence in a vector database keyed by recording ID, and serves grounding queries via a lightweight API that runs the span prediction head on the stored tokens. Query latency is dominated by the span head forward pass, not the vision encoder — the expensive part (frame extraction and embedding) happens at ingestion time, not query time. This is the same pattern as document RAG: pre-compute embeddings at write time, retrieve at read time.