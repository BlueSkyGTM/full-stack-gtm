## Ship It

Three production concerns will dominate your deployment experience: feature scaling, class imbalance, and regularization.

**Feature scaling.** Logistic regression is not scale-invariant. If one feature is revenue (values in millions) and another is email open rate (values between 0 and 1), the weight on revenue will need to be tiny and the weight on open rate will need to be large. The loss surface becomes elongated — a narrow ravine — and gradient descent bounces back and forth instead of descending smoothly. Standardize every feature to zero mean and unit variance before training. The model converges in a fraction of the iterations.

```python
import numpy as np

np.random.seed(42)

revenue = np.random.uniform(100000, 50000000, 500)
open_rate = np.random.uniform(0, 1, 500)
X_unscaled = np.column_stack([revenue, open_rate])
X_scaled = (X_unscaled - X_unscaled.mean(axis=0)) / X_unscaled.std(axis=0)

print(f"Unscaled ranges:  revenue [{X_unscaled[:,0].min():.0f}, {X_unscaled[:,0].max():.0f}]")
print(f"                  open_rate [{X_unscaled[:,1].min():.2f}, {X_unscaled[:,1].max():.2f}]")
print(f"Scaled ranges:    feature_0 [{X_scaled[:,0].min():.2f}, {X_scaled[:,0].max():.2f}]")
print(f"                  feature_1 [{X_scaled[:,1].min():.2f}, {X_scaled[:,1].max():.2f}]")
```

```
Unscaled ranges:  revenue [107666, 49867220]
                  open_rate [0.00, 1.00]
Scaled ranges:    feature_0 [-1.43, 1.72]
                  feature_1 [-1.73, 1.70]
```

**Class imbalance.** GTM datasets have conversion rates of 2-8%. If 5% of accounts convert, a model that predicts "always 0" achieves 95% accuracy and is completely useless. Accuracy is misleading when classes are imbalanced. Precision and recall are the metrics that matter — they tell you what the model does with the minority class, which is the one you care about. Always report precision, recall, and F1 alongside accuracy. In extreme imbalance (sub-1% positive rate), consider weighting the loss function to penalize missing positives more heavily, or use stratified sampling during training.

**L2 regularization.** Small training sets (common in GTM — you might have 200 closed deals and 3000 lost ones) produce overconfident models. The weights grow large to fit the training data exactly, and the model assigns probabilities of 0.99 to accounts that are genuinely uncertain. L2 regularization adds a penalty term $\lambda \|w\|^2$ to the loss function, shrinking weights toward zero. This produces less extreme probabilities and better generalization. The hyperparameter $\lambda$ controls the strength — too high and the model underfits, too low and it overfits. Most implementations default to $\lambda = 1.0$ (or $C = 1/\lambda = 1.0$ in scikit-learn's convention), which is a reasonable starting point but should be tuned on a validation set.

One more caution: logistic regression assumes a linear decision boundary. If the relationship between your features and the conversion outcome is non-linear (e.g., mid-market companies convert at higher rates than both SMB and enterprise), logistic regression will miss it unless you engineer interaction features or polynomial features manually. This is a known limitation, not a bug — the model is fast, interpretable, and calibrated, and it establishes a baseline that more complex models (gradient-boosted trees, neural networks) need to beat before you deploy them.