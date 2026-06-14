## Ship It

Here is the production consideration that catches teams off guard: the loss function you train with is not the metric you deploy against. You train with BCE because it produces clean gradients. You evaluate with precision, recall, and conversion rate because those are what your GTM team cares about. The loss function and the evaluation metric measure different things, and they should. BCE measures probabilistic calibration. Conversion rate measures business outcomes. A model with excellent BCE can still have poor precision if the decision threshold is wrong.

```python
import numpy as np

np.random.seed(42)
n_accounts = 1000
n_positive = 50

true_labels = np.zeros(n_accounts)
true_labels[:n_positive] = 1
features = np.random.randn(n_accounts, 5)
features[:n_positive] += np.array([1.5, -0.8, 0.6, 1.0, -0.4])

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

weights = np.random.randn(5) * 0.1
bias = 0.0
learning_rate = 0.5
epochs = 200

positive_weight = n_accounts / (2 * n_positive)
negative_weight = n_accounts / (2 * (n_accounts - n_positive))
sample_weights = np.where(true_labels == 1, positive_weight, negative_weight)

loss_history_standard = []
loss_history_weighted = []

weights_std = weights.copy()
weights_wgt = weights.copy()
bias_std = bias
bias_wgt = bias

for epoch in range(epochs):
    logits_std = features @ weights_std + bias_std
    probs_std = sigmoid(logits_std)
    eps = 1e-15
    probs_std = np.clip(probs_std, eps, 1 - eps)
    loss_std = -np.mean(true_labels * np.log(probs_std) + (1 - true_labels) * np.log(1 - probs_std))
    loss_history_standard.append(loss_std)
    grad_std = features.T @ (probs_std - true_labels) / n_accounts
    weights_std -= learning_rate * grad_std
    bias_std -= learning_rate * np.mean(probs_std - true_labels)

    logits_wgt = features @ weights_wgt + bias_wgt
    probs_wgt = sigmoid(logits_wgt)
    probs_wgt = np.clip(probs_wgt, eps, 1 - eps)
    loss_wgt = -np.mean(sample_weights * (true_labels * np.log(probs_wgt) + (1 - true_labels) * np.log(1 - probs_wgt)))
    loss_history_weighted.append(loss_wgt)
    grad_wgt = features.T @ (sample_weights * (probs_wgt - true_labels)) / n_accounts
    weights_wgt -= learning_rate * grad_wgt
    bias_wgt -= learning_rate * np.mean(sample_weights * (probs_wgt - true_labels))

print("ICP Model: Standard BCE vs. Weighted BCE (5% positive class)")
print("=" * 60)

threshold = 0.5
preds_std = sigmoid(features @ weights_std + bias_std) > threshold
preds_wgt = sigmoid(features @ weights_wgt + bias_wgt) > threshold

tp_std = np.sum((preds_std == 1) & (true_labels == 1))
fp_std = np.sum((preds_std == 1) & (true_labels == 0))
fn_std = np.sum((preds_std == 0) & (true_labels == 1))

tp_wgt = np.sum((preds_wgt == 1) & (true_labels == 1))
fp_wgt = np.sum((preds_wgt == 1) & (true_labels == 0))
fn_wgt = np.sum((preds_wgt == 0) & (true_labels == 1))

precision_std = tp_std / (tp_std + fp_std) if (tp_std + fp_std) > 0 else 0
recall_std = tp_std / n_positive
precision_wgt = tp_wgt / (tp_wgt + fp_wgt) if (tp_wgt + fp_wgt) > 0 else 0
recall_wgt = tp_wgt / n_positive

print(f"Standard BCE:")
print(f"  True positives found:  {tp_std}/{n_positive}")
print(f"  False positives:       {fp_std}")
print(f"  False negatives:       {fn_std}")
print(f"  Precision:             {precision_std:.3f}")
print(f"  Recall:                {recall_std:.3f}")
print()
print(f"Weighted BCE:")
print(f"  True positives found:  {tp_wgt}/{n_positive}")
print(f"  False positives:       {fp_wgt}")
print(f"  False negatives:       {fn_wgt}")
print(f"  Precision:             {precision_wgt:.3f}")
print(f"  Recall:                {recall_wgt:.3f}")
print()
print(f"Final training loss (standard): {loss_history_standard[-1]:.4f}")
print(f"Final training loss (weighted): {loss_history_weighted[-1]:.4f}")
```

The output tells the story. Standard BCE on imbalanced data will likely classify very few or zero accounts as positive — it minimizes loss by predicting low probabilities for everything. Weighted BCE recovers most of the true positives at the cost of more false positives. Whether that trade-off is correct depends on your GTM economics: if each correctly identified ICP account is worth $50K in pipeline and each wasted outbound sequence costs $20 in SDR time, you should err toward recall. The loss function encodes that trade-off through the class weight, not through the loss formula itself.

In a Clay workflow, these predictions become enrichment columns. The scraper from Zone 03 pulls the raw account data, the model (trained with weighted BCE) assigns a probability score, and a Clay Function — free boolean logic, no API credits — applies the threshold: `if icp_score > 0.7, route to SDR queue; else, park in nurture`. The loss function choice is invisible at deployment, but it determines whether the SDR queue is full of high-value accounts or garbage.