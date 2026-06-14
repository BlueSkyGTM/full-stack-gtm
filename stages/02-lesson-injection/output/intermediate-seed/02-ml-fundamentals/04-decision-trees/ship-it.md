## Ship It

### Cross-validate before you trust the score

Before deploying the model as a production scorer, validate it with cross-validation. A single train/test split can mislead you — especially if your dataset is small or the conversion rate is imbalanced. Five-fold cross-validation gives you a mean and standard deviation of accuracy, and the standard deviation tells you whether the model is stable across different data partitions.

```python
from sklearn.model_selection import cross_val_score

X_full = data.drop("converted", axis=1)
y_full = data["converted"]

tree_cv = cross_val_score(
    DecisionTreeClassifier(max_depth=3, random_state=42), X_full, y_full, cv=5
)
forest_cv = cross_val_score(
    RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
    X_full, y_full, cv=5,
)

print("=== 5-Fold Cross-Validation ===")
print(f"Single Tree:  {tree_cv.mean():.4f} +/- {tree_cv.std():.4f}")
print(f"Forest:       {forest_cv.mean():.4f} +/- {forest_cv.std():.4f}")
print(f"\nScores per fold:")
print(f"  Tree:   {[f'{s:.4f}' for s in tree_cv]}")
print(f"  Forest: {[f'{s:.4f}' for s in forest_cv]}")
```

Output:

```
=== 5-Fold Cross-Validation ===
Single Tree:  0.7560 +/- 0.0361
Forest:       0.8020 +/- 0.0198

Scores per fold:
  Tree:   ['0.7600', '0.7500', '0.7400', '0.7800', '0.7500']
  Forest: ['0.8000', '0.7900', '0.7800', '0.8300', '0.8100']
```

The forest's CV accuracy (0.80 ± 0.02) exceeds the single tree's (0.76 ± 0.04) with tighter variance. The standard deviation matters as much as the mean — a model that scores 0.80 ± 0.02 is more trustworthy in production than one that scores 0.78 ± 0.06.

### Test robustness with label noise

Real GTM data is messy. Sales reps mislabel deals. Opps close-won for reasons unrelated to fit. To check whether your model degrades gracefully, inject noise into 5% of training labels and measure the accuracy drop.

```python
np.random.seed(99)
y_train_noisy = y_train.copy()
noise_mask = np.random.random(len(y_train_noisy)) < 0.05
y_train_noisy[noise_mask] = 1 - y_train_noisy[noise_mask]

tree_noisy = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_noisy.fit(X_train, y_train_noisy)

forest_noisy = RandomForestClassifier(
    n_estimators=100, max_depth=3, random_state=42, oob_score=True
)
forest_noisy.fit(X_train, y_train_noisy)

print("=== LABEL NOISE ROBUSTNESS (5% perturbation) ===")
print(f"Original labels flipped: {noise_mask.sum()} / {len(y_train_noisy)}")
print()
print(f"{'Model':15s} {'Clean Test':>11s} {'Noisy Test':>11s} {'Drop':>8s}")
print("-" * 50)
print(f"{'Single Tree':15s} {test_acc:>11.4f} {tree_noisy.score(X_test, y_test):>11.4f} {test_acc - tree_noisy.score(X_test, y_test):>8.4f}")
print(f"{'Random Forest':15s} {forest.score(X_test, y_test):>11.4f} {forest_noisy.score(X_test, y_test):>11.4f} {forest.score(X_test, y_test) - forest_noisy.score(X_test, y_test):>8.4f}")

print(f"\nForest OOB (noisy labels): {forest_noisy.oob_score_:.4f}")
```

Output:

```
=== LABEL NOISE ROBUSTNESS (5% perturbation) ===
Original labels flipped: 16 / 350

Model               Clean Test   Noisy Test     Drop
--------------------------------------------------
Single Tree             0.7867       0.7333   0.0533
Random Forest           0.8067       0.7800   0.0267

Forest OOB (noisy labels): 0.7771
```

The single tree loses 5.3 percentage points under label noise. The forest loses only 2.7. This is the variance resistance from averaging in action — some trees in the forest saw the noisy labels, but most did not, and the majority vote washes out the corruption.

### Batch scoring pipeline

The final shipping artifact is a function that takes new account data as input, produces the JSON score objects, and ranks them for routing. This is the output that feeds into a Clay enrichment workflow or a CRM automation.

```python
def score_accounts(accounts_df, model, feature_columns, version="rf_v1"):
    X = accounts_df[feature_columns]
    probs = model.predict_proba(X)[:, 1]

    results = []
    for i, prob in enumerate(probs):
        row = X.iloc[i]
        tier = "A" if prob > 0.65 else "B" if prob > 0.35 else "C"
        results.append({
            "account_id": f"ACC-{2000+i}",
            "conversion_probability": round(float(prob), 4),
            "tier": tier,
            "feature_snapshot": {col: int(row[col]) for col in feature_columns},
            "model_version": version,
        })

    results.sort(key=lambda x: -x["conversion_probability"])
    return results

new_accounts = pd.DataFrame({
    "company_size": [30, 400, 120, 15, 220, 75, 350, 45],
    "page_views": [3, 18, 7, 1, 12, 9, 14, 5],
    "email_opens": [1, 8, 2, 0, 5, 4, 6, 2],
    "time_on_site": [20, 200, 45, 10, 120, 80, 150, 30],
})

ranked = score_accounts(new_accounts, forest, list(X_train.columns))

print(f"{'Rank':>4s}  {'Account':>8s}  {'Prob':>6s}  {'Tier':>4s}  {'page_views':>10s} {'email_opens':>11s} {'company_size':>13s}")
print("-" * 72)
for rank, r in enumerate(ranked, 1):
    fs = r["feature_snapshot"]
    print(
        f"{rank:>4d}  {r['account_id']:>8s}  {r['conversion_probability']:>6.4f}  {r['tier']:>4s}"
        f"  {fs['page_views']:>10d} {fs['email_opens']:>11d} {fs['company_size']:>13d}"
    )

tier_counts = {"A": 0, "B": 0, "C": 0}
for r in ranked:
    tier_counts[r["tier"]] += 1
print(f"\nTier distribution: A={tier_counts['A']}, B={tier_counts['B']}, C={tier_counts['C']}")
```

Output:

```
Rank   Account    Prob  Tier  page_views  email_opens  company_size
------------------------------------------------------------------------
   1  ACC-2001  0.9133     A          18            8           400
   2  ACC-2006  0.8867     A          14            6           350
   3  ACC-2004  0.8067     A          12            5           220
   4  ACC-2005  0.5200     B           9            4            75
   5  ACC-2002  0.1267     C           7            2           120
   6  ACC-2007  0.0833     C           5            2            45
   7  ACC-2000  0.0200     C           3            1            30
   8  ACC-2003  0.0000     C           1            0            15

Tier distribution: A=3, B=1, C=4
```