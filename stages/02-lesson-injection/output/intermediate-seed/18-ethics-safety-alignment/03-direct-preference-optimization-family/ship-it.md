## Ship It

DPO-trained models drift when the preference distribution shifts. A new ICP, a new market vertical, a new quarter with different seasonal response patterns — all of these change what "good outreach" means. The implicit reward your model learned no longer matches the real outcome distribution.

Ship a pipeline that does three things. First, log every generation with its context, so you can later label outcomes once replies (or non-replies) come in. Second, monitor the implicit reward margin on live traffic — if the model's average log-ratio difference between its chosen output and the reference starts collapsing, the model is becoming over-confident on inputs that no longer match training data. Third, trigger a re-fine-tune cycle when the win rate (as measured by reply rate on new generations versus a held-out SFT baseline) drops below a threshold.

```python
import time
from collections import deque

class PreferenceMonitor:
    def __init__(self, ref_win_rate, min_samples=50, retrain_threshold=0.05):
        self.ref_win_rate = ref_win_rate
        self.min_samples = min_samples
        self.retrain_threshold = retrain_threshold
        self.recent_outcomes = deque(maxlen=200)
        self.reward_margins = deque(maxlen=200)
    
    def log_generation(self, prompt, response, reward_margin):
        self.reward_margins.append(reward_margin)
    
    def log_outcome(self, prompt, response, replied, sentiment):
        win = 1 if (replied and sentiment == "positive") else 0
        self.recent_outcomes.append(win)
    
    def check_drift(self):
        if len(self.recent_outcomes) < self.min_samples:
            return None
        
        current_win_rate = sum(self.recent_outcomes) / len(self.recent_outcomes)
        drop = self.ref_win_rate - current_win_rate
        
        margins = list(self.reward_margins)
        avg_margin = sum(margins) / len(margins) if margins else 0
        
        status = {
            "current_win_rate": round(current_win_rate, 4),
            "baseline_win_rate": self.ref_win_rate,
            "drop": round(drop, 4),
            "avg_reward_margin": round(avg_margin, 4),
            "samples": len(self.recent_outcomes),
            "retrain_recommended": drop > self.retrain_threshold
        }
        return status

monitor = PreferenceMonitor(ref_win_rate=0.12, retrain_threshold=0.04)

for i in range(60):
    margin = max(0.5 - i * 0.008, 0.1)
    monitor.log_generation(f"prompt_{i}", f"response_{i}", margin)

outcomes = [(True, "positive")] * 8 + [(False, None)] * 52
for replied, sentiment in outcomes:
    monitor.log_outcome("p", "r", replied, sentiment)

result = monitor.check_drift()
print("Preference drift monitor:")
for k, v in result.items():
    print(f"  {k}: {v}")

if result["retrain_recommended"]:
    print("\nAction: Win rate dropped below threshold.")
    print("  1. Export last 200 generations from CRM as new preference data")
    print("  2. Filter positive replies (chosen) vs. no-reply after 5 days (rejected)")
    print("  3. Run DPOTrainer for 2-3 epochs on the new slice")
    print("  4. Validate on a held-out 20-sample set before deploying")
else:
    print("\nStatus: Within tolerance. No retrain needed.")
```

The monitor tracks two signals: the live win rate (reply rate on recent generations versus the baseline established after the last fine-tune) and the average reward margin (how confident the model is in its preferences). When both drop, the model has drifted — it is producing outputs it thinks are good (high margin) but the market no longer rewards, or it is losing confidence (collapsing margin) on inputs it was never trained on.

For the retrain cycle, use DPO if you accumulated clean A/B pairs, or KTO if you only have binary outcomes. In practice, most GTM teams will use KTO for retraining cycles because they rarely run controlled A/B tests on the same prospect — they send one email and observe whether it worked. The KTO baseline (z_ref) adapts to the current model's average log-ratio, so it naturally handles the fact that the "good" threshold shifts as the model improves.