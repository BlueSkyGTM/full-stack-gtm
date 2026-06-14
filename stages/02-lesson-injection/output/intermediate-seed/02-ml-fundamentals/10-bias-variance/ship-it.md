## Ship It

Before you deploy any scoring model into your GTM stack, run the diagnostic. Build it into your pipeline as a gating check: if the train-test gap exceeds your tolerance or the rows-per-feature ratio falls below 10, the model does not ship. This is not a suggestion — it is the difference between a score your team trusts and one they learn to ignore within a week.

The decision tree is simple. If training accuracy is low (both train and test are bad), your model is underfitting — it has high bias. Add features, increase model complexity, or check whether your labels actually capture the outcome you care about. If training accuracy is high but holdout accuracy is low, your model is overfitting — it has high variance. Regularize with L1 or L2, reduce features via selection or PCA, or collect more training data. More data shrinks variance because the model has more examples to average over, but it never fixes bias because the functional form is unchanged.

In practice, the 10-rows-per-feature heuristic is a floor, not a target. For noisy labels (like "did this account convert?" where conversion depends on factors outside your data), you may need 50 or 100 rows per feature before the model is stable. Cross-validation gives you the empirical answer: if your test MSE varies wildly across folds, variance is high regardless of what the ratio says.

Here is a version of the diagnostic you can drop into a scoring pipeline as a pre-deployment gate:

```python
def scoring_gate(n_rows, n_features, train_score, test_score, max_gap=0.10, min_rows_per_feature=10):
    rows_per_feature = n_rows / max(n_features, 1)
    gap = abs(train_score - test_score)

    checks = []
    checks.append(("Rows per feature >= threshold", rows_per_feature >= min_rows_per_feature, f"{rows_per_feature:.1f} vs {min_rows_per_feature}"))
    checks.append(("Train-test gap within tolerance", gap <= max_gap, f"{gap:.3f} vs {max_gap}"))
    checks.append(("Test score above baseline", test_score > 0.55, f"{test_score:.3f} vs 0.55"))

    print("=" * 60)
    print("SCORING MODEL DEPLOYMENT GATE")
    print("=" * 60)
    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name} ({detail})")

    print("-" * 60)
    if all_pass:
        print("  RESULT: APPROVED for deployment")
    else:
        print("  RESULT: BLOCKED — resolve issues before deploying")
    print("=" * 60)
    return all_pass

scoring_gate(
    n_rows=200,
    n_features=12,
    train_score=0.85,
    test_score=0.79,
    max_gap=0.10
)

print()

scoring_gate(
    n_rows=50,
    n_features=20,
    train_score=0.94,
    test_score=0.61,
    max_gap=0.10
)
```

Output:

```
============================================================
SCORING MODEL DEPLOYMENT GATE
============================================================
  [PASS] Rows per feature >= threshold (16.7 vs 10)
  [PASS] Train-test gap within tolerance (0.060 vs 0.1)
  [PASS] Test score above baseline (0.790 vs 0.55)
------------------------------------------------------------
  RESULT: APPROVED for deployment
============================================================

============================================================
SCORING MODEL DEPLOYMENT GATE
============================================================
  [FAIL] Rows per feature >= threshold (2.5 vs 10)
  [FAIL] Train-test gap within tolerance (0.330 vs 0.1)
  [PASS] Test score above baseline (0.610 vs 0.55)
------------------------------------------------------------
  RESULT: BLOCKED — resolve issues before deploying
============================================================
```

The first model ships. The second does not. That gate is the practical application of the bias-variance tradeoff: it catches the overfitting case before it reaches your SDR team, and it forces you to either collect more data, simplify your feature set, or regularize. Every JSON scoring object your pipeline emits downstream — the kind covered in Zone 2 of the curriculum, where lead scores are data structures with fields and confidence values — inherits its trustworthiness from this check. If the gate fails, the score in that JSON object is noise dressed up as signal.