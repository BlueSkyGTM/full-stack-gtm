## Ship It

Deploying a video generation or world model pipeline in production means solving three engineering problems that don't appear in the research papers: latency, cost, and quality control. A video diffusion model generating a 10-second clip at 24fps, 512x512 resolution runs hundreds of denoising steps, each involving spatial and temporal attention over thousands of tokens. On an A100 that's seconds to minutes per clip. On CPU it's impractical. Your deployment target is GPU inference, and your latency budget determines whether you generate synchronously (user waits) or asynchronously (queue and deliver later).

For a GTM enrichment waterfall—the Clay pattern of Find → Enrich → Transform → Export—the video generation step sits in the Transform position. It takes enriched inputs (company name, persona, tech stack, pain point) and produces a video asset that gets exported to the outreach tool. The waterfall pattern matters here because each stage can fail independently. If the enrichment step returns incomplete data (missing tech stack, unverified persona), the conditioning vector is degraded, and the generated video will be generic rather than personalised. The waterfall must handle this: fall back to a template video when enrichment quality is below threshold, or skip the personalised video step entirely and use a recorded Vidyard-style video instead.

```python
import numpy as np
from collections import deque
import time

np.random.seed(42)

class VideoGenerationStage:
    def __init__(self, name, min_quality=0.5, latency_budget=2.0):
        self.name = name
        self.min_quality = min_quality
        self.latency_budget = latency_budget

    def run(self, data):
        start = time.time()
        quality = data.get("enrichment_quality", 0.0)
        if quality < self.min_quality:
            data["video_status"] = "skipped_low_quality"
            data["video_path"] = None
            data["fallback"] = "template_video"
            return data

        time.sleep(0.01)
        data["video_status"] = "generated"
        data["video_path"] = f"/assets/{data['company_id']}_personalized.mp4"
        data["gen_latency_ms"] = (time.time() - start) * 1000
        return data

class EnrichmentWaterfall:
    def __init__(self, stages):
        self.stages = stages
        self.results = []

    def run(self, prospect_batch):
        for prospect in prospect_batch:
            data = prospect.copy()
            for stage in self.stages:
                data = stage.run(data)
                if data.get("video_status") == "skipped_low_quality":
                    break
            self.results.append(data)
        return self.results

class MetricsCollector:
    def __init__(self):
        self.stats = {"generated": 0, "skipped": 0, "latencies": []}

    def collect(self, results):
        for r in results:
            status = r.get("video_status", "unknown")
            if status == "generated":
                self.stats["generated"] += 1
                self.stats["latencies"].append(r.get("gen_latency_ms", 0))
            elif status == "skipped_low_quality":
                self.stats["skipped"] += 1

    def report(self):
        gen = self.stats["generated"]
        skip = self.stats["skipped"]
        total = gen + skip
        lats = self.stats["latencies"]
        print(f"Waterfall Metrics Report")
        print(f"  Total prospects:     {total}")
        print(f"  Videos generated:    {gen} ({gen/max(total,1)*100:.0f}%)")
        print(f"  Videos skipped:      {skip} ({skip/max(total,1)*100:.0f}%)")
        if lats:
            print(f"  Avg generation time: {np.mean(lats):.1f}ms")
            print(f"  P95 generation time: {np.percentile(lats, 95):.1f}ms")

prospects = []
for i in range(100):
    quality = np.random.beta(2, 3)
    prospects.append({
        "company_id": f"comp_{i:03d}",
        "enrichment_quality": quality,
        "persona": "VP Sales" if quality > 0.4 else "Unknown",
    })

pipeline = EnrichmentWaterfall([
    VideoGenerationStage("personalized_video", min_quality=0.4, latency_budget=2.0)
])

results = pipeline.run(prospects)
metrics = MetricsCollector()
metrics.collect(results)
metrics.report()

quality_scores = [p["enrichment_quality"] for p in prospects]
print(f"\nEnrichment quality distribution:")
print(f"  Mean:   {np.mean(quality_scores):.3f}")
print(f"  Median: {np.median(quality_scores):.3f}")
print(f"  Min:    {np.min(quality_scores):.3f}")
print(f"  Max:    {np.max(quality_scores):.3f}")
print(f"  Below threshold (0.4): {sum(1 for q in quality_scores if q < 0.4)} / {len(quality_scores)}")
```

The output shows you the skip rate and latency distribution for a realistic batch. Roughly 40-50% of prospects will have enrichment quality below threshold, which means the waterfall correctly falls back to template video for them. This is the production reality: generative video is not a replacement for deterministic enrichment, it's a transform stage that activates when data quality justifies the compute cost. The Clay waterfall pattern—Find, Enrich, Transform, Export—handles this naturally because each stage is independently failable and the pipeline degrades gracefully.

For model selection when shipping: if your use case is creative marketing video (long-form, high aesthetic quality, no fine-grained control needed), Runway GWM-1 or Sora 2 are the right tier. If you need interactive simulation (action-conditioned, real-time), Genie 3 or a custom DreamerV3-style world model fits. If you need open-source, self-hostable video generation for cost control, HunyuanVideo or Wan-Video are the current options. The decision is not "which model is best"—it's "which architecture matches my conditioning requirements and latency budget."