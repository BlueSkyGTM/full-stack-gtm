## Ship It

In production, cross-attention fusion replaces brittle rule-based merging when orchestrating multi-source enrichment. The pattern: embed each provider's output into a shared vector space, feed them as separate sequences into a cross-attention layer, and use the fused output as input to a downstream classifier (typically a single linear layer producing a conversion probability). The attention weights are logged per account so you can inspect which signals drove each score.

```python
import numpy as np

np.random.seed(7)

d_model = 16
d_k = 8
d_v = 8
n_firmographic = 4
n_intent = 6

firmographic_embeddings = np.random.randn(n_firmographic, d_model)
intent_embeddings = np.random.randn(n_intent, d_model)

W_Q = np.random.randn(d_model, d_k) * 0.1
W_K = np.random.randn(d_model, d_k) * 0.1
W_V = np.random.randn(d_model, d_v) * 0.1

Q = firmographic_embeddings @ W_Q
K = intent_embeddings @ W_K
V = intent_embeddings @ W_V

scores = Q @ K.T / np.sqrt(d_k)

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

attention_weights = softmax(scores, axis=-1)
fused = attention_weights @ V

pool = fused.mean(axis=0)
W_score = np.random.randn(d_v, 1) * 0.1
score = float(1 / (1 + np.exp(-(pool @ W_score))))

print(f"Account fused score: {score:.4f}")
print(f"Attention matrix shape: {attention_weights.shape} ({n_firmographic} firmographic x {n_intent} intent)")

firmographic_labels = ["industry", "employee_count", "revenue_band", "tech_stack"]
intent_labels = ["security_scan", "pricing_page", "competitor_comparison", "demo_request", "api_docs", "case_studies"]

print("\nTop-3 attention links (which intent signals drove which firmographic attribute):")
flat = []
for i in range(n_firmographic):
    for j in range(n_intent):
        flat.append((attention_weights[i, j], firmographic_labels[i], intent_labels[j]))
flat.sort(key=lambda x: x[0], reverse=True)
for w, fi, ij in flat[:3]:
    print(f"  {fi} ← {ij}: {w:.4f}")

print("\nFull attention matrix (rows=firmographic, cols=intent):")
header = "                    " + "  ".join(f"{l[:8]:>8s}" for l in intent_labels)
print(header)
for i in range(n_firmographic):
    row = "  ".join(f"{attention_weights[i,j]:8.4f}" for j in range(n_intent))
    print(f"  {firmographic_labels[i]:>16s}  {row}")
```

Three practical constraints matter when shipping this. First, you need aligned embedding spaces. Clearbit returns structured fields (industry, employee_count); Bombora returns intent topics with surge scores. You need a projection layer that maps each provider's output into a shared $d_{model}$-dimensional space before cross-attention. This projection is learned — typically initialized from a pre-trained encoder and fine-tuned on labeled outcome data (converted / not converted).

Second, you need a fallback for missing providers. If Bombora has no intent data for an account, the intent sequence is empty. Cross-attention with zero-length memory produces zero output. The production pattern is to fall back to a self-attention-only path (firmographic embeddings through a feed-forward layer) and flag the account as low-confidence. The downstream classifier should see the missing-signal flag as a feature, not silently ignore it.

Third, the attention matrix is the audit trail. Log it per account. When a sales rep asks "why did this account score 0.87?", the attention weights tell you: revenue_band attended most to competitor_comparison at 0.31, tech_stack attended to api_docs at 0.24. This is something concatenation-based scoring cannot provide — the linear weights are global, not per-instance.

[CITATION NEEDED — concept: Clay waterfall as the specific enrichment orchestration pattern that produces multi-provider data requiring fusion]