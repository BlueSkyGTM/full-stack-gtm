## Ship It

Selecting `λ` in development is step one. Monitoring the train-test gap in production is step two. The gap is your drift detector: if the model's training AUC stays high but test AUC degrades over weeks, the input distribution has shifted and the model is no longer generalizing — it needs retraining with updated data and a fresh `λ`.

The following function takes a trained model, a training set, and a held-out set, and prints the overfit diagnostic. Wire it into a weekly monitoring job or a CI check that runs after model retraining.

```python
import numpy as np
from sklearn.metrics import roc_auc_score

def overfit_diagnostic(model, X_train, y_train, X_test, y_test, threshold=0.10):
    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, train_pred)
    test_auc = roc_auc_score(y_test, test_pred)

    gap = train_auc - test_auc

    if hasattr(model, "coef_"):
        coef_norm = np.linalg.norm(model.coef_)
        n_features = len(model.coef_[0])
        n_zero = np.sum(np.abs(model.coef_[0]) < 1e-4)
    else:
        coef_norm = float("nan")
        n_features = 0
        n_zero = 0

    status = "OK" if gap < threshold else "DRIFT DETECTED — retrain"

    print(f"{'Metric':<30} {'Value':>12}")
    print("-" * 44)
    print(f"{'Train AUC':<30} {train_auc:>12.4f}")
    print(f"{'Test AUC':<30} {test_auc:>12.4f}")
    print(f"{'Gap (train - test)':<30} {gap:>12.4f}")
    print(f"{'Drift threshold':<30} {threshold:>12.4f}")
    print(f"{'Status':<30} {status:>12}")
    print(f"{'Coefficient L2 norm':<30} {coef_norm:>12.4f}")
    print(f"{'Total features':<30} {n_features:>12}")
    print(f"{'Zeroed features (L1)':<30} {n_zero:>12}")

    return {"train_auc": train_auc, "test_auc": test_auc, "gap": gap, "status": status}


from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X_synth, y_synth = make_classification(
    n_samples=2000, n_features=50, n_informative=5, n_redundant=10, random_state=99
)

X_tr, X_te = X_synth[:1400], X_synth[1400:]
y_tr, y_te = y_synth[:1400], y_synth[1400:]

overfit_model = LogisticRegression(penalty=None, max_iter=5000, solver="lbfgs")
overfit_model.fit(X_tr, y_tr)

print("=== Unregularized Model ===")
overfit_diagnostic(overfit_model, X_tr, y_tr, X_te, y_te)

print()

regularized_model = LogisticRegression(
    penalty="l2", C=0.01, max_iter=5000, solver="lbfgs"
)
regularized_model.fit(X_tr, y_tr)

print("=== Regularized Model (L2, C=0.01) ===")
overfit_diagnostic(regularized_model, X_tr, y_tr, X_te, y_te)
```

Run this and compare the two outputs. The unregularized model shows a wider train-test gap; the regularized model narrows it. In production, log the gap alongside the model version and the training timestamp. If the gap exceeds your threshold (0.10 AUC is a reasonable starting point), trigger a retraining pipeline that pulls fresh conversion data, re-runs cross-validation for `λ`, and deploys the updated model with its metadata: penalty type, `λ` value, coefficient norm, and the cross-validated score at deployment time.