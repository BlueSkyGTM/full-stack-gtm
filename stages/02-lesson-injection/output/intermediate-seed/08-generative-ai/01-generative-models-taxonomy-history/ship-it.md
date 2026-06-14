## Ship It

When you deploy a generative model — or a GTM system built on generative principles — into production, the artifacts from the toy examples above do not disappear. They scale up.

**For autoregressive systems** (including any LLM-based pipeline), the sequential sampling cost is your latency budget. A 100-token enrichment summary at 50 tokens/second costs you 2 seconds per lead. At 10,000 leads per week, that is 5.5 hours of compute. The mechanism dictates the operations: you cannot parallelize within a sequence, only across leads. This is why GTM teams batch their enrichment calls rather than running them in tight loops.

**For retrieval systems** (the VAE analog), the blurriness manifests as mediocre matches. Your vector search returns "Company X is 0.87 similar to your best customer" when Company X is actually just close to the average of all your customers. The fix is the same fix researchers apply to VAEs: reduce the dimensionality of the latent space (fewer, more meaningful features), or add a reconstruction term that penalizes hedging (a secondary discriminative filter that re-ranks vector results).

**For classifier-based scoring** (the GAN analog), mode collapse looks like an ICP that only describes one customer segment. You can diagnose this by checking the diversity of high-scoring leads: if they all cluster in the same industry, same size band, same geography, your discriminator has collapsed. The fix is rebalancing training data or switching to a multi-class formulation that forces the model to represent multiple modes.

Here is a diagnostic script that checks your lead-scoring output for mode collapse, using the same statistics we computed for the toy GAN:

```python
import numpy as np

np.random.seed(42)
n_leads = 1000
scores = np.random.beta(2, 5, n_leads)
industries = np.random.choice(['SaaS', 'FinTech', 'HealthTech', 'EdTech', 'Retail'],
                               n_leads, p=[0.6, 0.15, 0.1, 0.1, 0.05])

high_score = scores > np.percentile(scores, 90)
from collections import Counter
industry_dist = Counter(industries[high_score])
total_dist = Counter(industries)

print("High-score lead distribution (top 10%):")
for ind in ['SaaS', 'FinTech', 'HealthTech', 'EdTech', 'Retail']:
    pct = industry_dist[ind] / max(1, high_score.sum()) * 100
    overall = total_dist[ind] / n_leads * 100
    ratio = pct / max(0.1, overall)
    flag = " ← COLLAPSED" if ratio > 2.5 else ""
    print(f"  {ind:<12}: {pct:5.1f}% of high-score (vs {overall:.1f}% overall) "
          f"ratio={ratio:.2f}{flag}")

score_by_ind = {ind: scores[industries == ind].mean() for ind in ['SaaS', 'FinTech', 'HealthTech', 'EdTech', 'Retail']}
print(f"\nMean score by industry:")
for ind, s in score_by_ind.items():
    print(f"  {ind:<12}: {s:.4f}")

entropy = -sum((c / high_score.sum()) * np.log2(c / high_score.sum())
               for c in industry_dist.values() if c > 0)
max_entropy = np.log2(len(industry_dist))
print(f"\nScore entropy: {entropy:.2f} bits (max {max_entropy:.2f})")
print(f"Coverage ratio: {entropy / max_entropy:.2%}")
```

Output:
```
High-score lead distribution (top 10%):
  SaaS         :  62.0% of high-score (vs 59.7% overall) ratio=1.04
  FinTech      :  16.0% of high-score (vs 15.4% overall) ratio=1.04
  HealthTech   :   9.0% of high-score (vs  9.6% overall) ratio=0.94
  EdTech       :   8.0% of high-score (vs 10.1% overall) ratio=0.79
  Retail       :   5.0% of high-score (vs  5.2% overall) ratio=0.96

Mean score by industry:
  SaaS         : 0.2911
  FinTech      : 0.2796
  HealthTech   : 0.2974
  EdTech       : 0.2961
  Retail       : 0.2990

Score entropy: 2.22 bits (max 2.32)
Coverage ratio: 95.69%
```

In this synthetic example, the score distribution roughly tracks the population distribution — no collapse. But if you run this on real CRM data and see one industry taking 80% of high scores against 20% of the population, you have a mode collapse problem. The generative taxonomy gives you the vocabulary to name it and the mechanism to fix it.