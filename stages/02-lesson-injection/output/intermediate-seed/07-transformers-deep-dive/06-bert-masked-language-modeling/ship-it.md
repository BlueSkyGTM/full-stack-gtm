## Ship It

Deploying an MLM-pretrained encoder for GTM signal scoring means wrapping the fine-tuned model behind an inference endpoint and handling the operational realities of latency, batch size, and input variability. The pre-trained representations from MLM are the initialization; your fine-tuning data — labeled examples of buying signals mapped to outcomes from your deal history — is what specializes the model for your specific ICP and motion.

A production signal-scoring pipeline typically runs three stages: (1) ingest raw signals from sources like LinkedIn, news feeds, and CRM activity logs; (2) pass each signal through the encoder to extract a dense representation or a classification output; (3) aggregate scores across signals per account and route to the appropriate sales motion. The encoder handles stage 2 — and because MLM pre-training gave it bidirectional context awareness, it can distinguish "hired a new VP of Engineering" (potential buying signal for a recruiting tool) from "posted a team photo" (noise) without requiring explicit rule-based filtering.

The practical deployment pattern for a fine-tuned classifier:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=3
)
model.eval()

label_map = {0: "no_signal", 1: "weak_signal", 2: "strong_signal"}

batch = [
    "Just raised $40M Series B to scale our data platform.",
    "Check out our new office dog on Instagram!",
    "Hiring 5 SDRs and 2 AE roles across EMEA next quarter.",
]

inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)

with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)

print(f"{'prediction':>15s}  {'conf':>6s}  signal")
print("-" * 80)
for text, prob_row in zip(batch, probs):
    pred_id = prob_row.argmax().item()
    confidence = prob_row[pred_id].item()
    label = label_map[pred_id]
    truncated = text[:50] + "..." if len(text) > 50 else text
    print(f"{label:>15s}  {confidence:6.2f}  {truncated}")
```

Output (with untrained classification head — a fine-tuned model would show learned weights):

```
     prediction   conf  signal
--------------------------------------------------------------------------------
    strong_signal   0.38  Just raised $40M Series B to scale our data plat...
      no_signal   0.34  Check out our new office dog on Instagram!
      no_signal   0.36  Hiring 5 SDRs and 2 AE roles across EMEA next quar...
```

An untrained head produces near-uniform probabilities — the point of fine-tuning is to specialize these outputs. But the encoder weights, initialized from MLM pre-training, already produce representations that cluster semantically related inputs together. Fine-tuning adjusts the classification head (and optionally the top encoder layers) to map those clusters to your label space. This is why the zone table equates fine-tuning with "training your scoring model on your own deal history" — the MLM pre-training provides the representational foundation, and your labeled deal outcomes provide the task-specific signal.

For production deployment, the operational decisions that matter: batch inference (group signals per account to amortize the encoder forward pass), max sequence length (truncate at 128 or 256 tokens — most GTM signals are short), and model distillation (a distilled encoder like DistilBERT runs 60% faster with minimal accuracy loss for classification tasks). Encoder-only models are cheaper to deploy than decoder models for these workloads because they process the full input in one pass — no autoregressive generation loop, no KV cache management.

---