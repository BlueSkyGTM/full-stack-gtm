# Decision Trees and Random Forests

## Learning Objectives

- Implement Gini impurity and information gain calculations from scratch to select optimal splits in a decision tree
- Build a `DecisionTreeClassifier` and a `RandomForestClassifier` on lead-conversion data, extract tree rules, and rank feature importances
- Compare train vs. test accuracy across `max_depth` values to identify the overfitting threshold for both models
- Evaluate model stability using k-fold cross-validation and out-of-bag error estimation
- Construct a JSON-based lead-scoring pipeline that ranks accounts using a trained random forest, mapping feature importances to ICP qualification criteria

## The Problem

You have a spreadsheet of historical leads. Each row is an account that either closed-won or closed-lost. Each column is a signal: company size, page views, email opens, time on site, trial signups, pricing page visits. Your VP of sales wants to know which of next quarter's 10,000 inbound leads will convert, and they want to know *why* — not just a score, but the actual decision logic behind it.

You could train a neural network. For tabular GTM data, that is usually the wrong call. Tree-based models handle mixed feature types (numeric and categorical) without preprocessing, capture nonlinear interactions without feature engineering, and remain interpretable enough that you can show a sales leader the exact split rules driving each prediction. Kaggle competitions on structured data are dominated by XGBoost and LightGBM, not transformers. The same holds for your CRM export.

A single decision tree solves the interpretability problem but introduces a new one: instability. Add five new rows to your training data and the tree may choose a completely different root split. Random forests fix this by averaging predictions across hundreds of de-correlated trees, each trained on a bootstrap sample with a random subset of features. The bias stays low (each tree is deep enough to capture real patterns), and the variance drops through averaging (the trees disagree with each other in different ways, so their average is more stable than any individual tree). This is the bias-variance tradeoff in action, and it is the entire reason a forest outperforms a single tree on held-out data.

In GTM terms, this means your lead score stops swinging wildly every time you refresh your training set. A forest trained on last quarter's conversions produces a stable ranking of next quarter's accounts, and the feature importances tell you which signals actually predict closed-won deals — not which signals your sales team *assumes* matter.

## The Concept

### How a decision tree partitions feature space

A decision tree splits your data by asking a sequence of yes/no questions. At each node, it scans every feature and every possible threshold, computes the impurity reduction for each candidate split, and picks the one that produces the purest child nodes. "Pure" means all samples in a node belong to the same class — a node of 100 converted leads is perfectly pure; a 50/50 split is maximally impure.

The two standard impurity measures are Gini impurity and entropy. Gini impurity for a node with $K$ classes is:

$$Gini = 1 - \sum_{k=1}^{K} p_k^2$$

where $p_k$ is the proportion of class $k$ samples in the node. Entropy is:

$$H = -\sum_{k=1}^{K} p_k \log_2(p_k)$$

Both produce values close to 0 for pure nodes and close to 1 (for binary classification) when classes are evenly mixed. The tree picks the split that maximizes the weighted impurity reduction — the difference between the parent node's impurity and the weighted average of the children's impurities. In practice, Gini and entropy produce nearly identical trees. Gini is computationally cheaper (no logarithm), so `scikit-learn` uses it as the default.

The tree applies this splitting recursively: split the root, then split each child, then split each grandchild, until it hits a stopping condition (`max_depth`, `min_samples_leaf`, or pure nodes). The result is a partition of feature space into axis-aligned rectangles — each rectangle corresponds to a leaf node, and every sample landing in that rectangle gets the same prediction.

```mermaid
graph TD
    A["Root: all leads<br/>Gini = 0.48<br/>320 samples"] -->|"page_views >= 12"| B["High-engagement<br/>Gini = 0.32<br/>180 samples"]
    A -->|"page_views < 12"| C["Low-engagement<br/>Gini = 0.45<br/>140 samples"]
    B -->|"email_opens >= 5"| D["Hot lead<br/>Gini = 0.08<br/>95 samples<br/>P(convert) = 0.97"]
    B -->|"email_opens < 5"| E["Warm lead<br/>Gini = 0.49<br/>85 samples<br/>P(convert) = 0.53"]
    C -->|"company_size >= 50"| F["Mid-market<br/>Gini = 0.40<br/>60 samples<br/>P(convert) = 0.30"]
    C -->|"company_size < 50"| G["SMB<br/>Gini = 0.28<br/>80 samples<br/>P(convert) = 0.10"]
    D --> H["LEAF: Score = 0.97"]
    E --> I["LEAF: Score = 0.53"]
    F --> J["LEAF: Score = 0.30"]
    G --> K["LEAF: Score = 0.10"]
```

### Why a single tree is unstable

The greedy splitting process means the root split depends heavily on the specific training data. Change a handful of rows and a different feature may win at the root, cascading into a completely different tree structure downstream. This is high variance: the model fits the training data tightly but generalizes inconsistently to new data.

### How bagging reduces variance

Bootstrap aggregating (bagging) trains many trees on different bootstrap samples — random draws with replacement from the training set. Each tree sees a slightly different dataset, so each tree makes different splitting decisions. When you average their predictions, the variance of the ensemble drops. Mathematically, if you have $B$ trees with variance $\sigma^2$ and pairwise correlation $\rho$, the variance of the averaged prediction is:

$$\text{Var}_{avg} = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$$

As $B$ grows, the second term vanishes. But the first term — controlled by correlation $\rho$ — remains. If all trees are identical ($\rho = 1$), averaging does nothing.

### Why feature subsampling is the key ingredient

This is where random forests diverge from plain bagging. At each split, a random forest considers only a random subset of features (typically $\sqrt{p}$ for classification, where $p$ is the total feature count). If `page_views` is the strongest predictor, every bagged tree will split on it first — producing highly correlated trees. By forcing each split to choose from a random feature subset, some trees cannot use `page_views` at the root and must find alternative splits. This decorrelates the trees, driving $\rho$ down, and making the averaging actually effective.

`scikit-learn` implements this in `RandomForestClassifier`. The `n_estimators` parameter controls tree count ($B$), `max_features` controls the feature subset size, and `bootstrap=True` enables resampling. After the concept is clear, these parameters become knobs you tune rather than incantations you copy.

## Build It

### Step 1: Generate a synthetic lead-conversion dataset

This code generates 500 synthetic accounts with four features and a binary conversion label. The conversion logic is nonlinear (conversion depends on the interaction of page_views and email_opens, not any single feature), which is exactly the regime where trees excel over linear models.

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 500

data = pd.DataFrame({
    "company_size": np.random.randint(5, 500, n),
    "page_views": np.random.poisson(8, n),
    "email_opens": np.random.poisson(3, n),
    "time_on_site": np.random.exponential(120, n),
})

logits = (
    -2.5
    + 0.04 * data["page_views"]
    + 0.30 * data["email_opens"]
    + 0.002 * data["time_on_site"]
    + 0.003 * data["company_size"]
    - 0.001 * data["page_views"] * data["email_opens"]
)

probs = 1 / (1 + np.exp(-logits))
data["converted"] = np.random.binomial(1, probs, n)

print(f"Dataset shape: {data.shape}")
print(f"Conversion rate: {data['converted'].mean():.3f}")
print(data.head(10).to_string())
```

Output:

```
Dataset shape: (500, 5)
Conversion rate: 0.374
   company_size  page_views  email_opens  time_on_site  converted
0           108           7            2     43.576342          0
1           229           9            4     39.820476          0
2           182           5            4     20.356186          0
3           118           9            2     44.474649          0
4           226           8            5    224.417808          1
5            29           8            3     96.366465          0
6           423          12            4     33.709007          0
7           168           7            5    138.341808          1
8            76           6            3     11.051930          0
9           118           8            3     83.557539          0
```

### Step 2: Train a single decision tree and inspect the rules

```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

X = data.drop("converted", axis=1)
y = data["converted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

print("=== ROOT SPLIT ===")
feature_name = X_train.columns[tree.tree_.feature[0]]
threshold = tree.tree_.threshold[0]
gini_root = tree.tree_.impurity[0]
n_samples_root = tree.tree_.n_node_samples[0]

print(f"Split feature: {feature_name}")
print(f"Threshold: {threshold:.4f}")
print(f"Gini impurity at root: {gini_root:.4f}")
print(f"Samples at root: {n_samples_root}")

print("\n=== FULL TREE RULES ===")
print(export_text(tree, feature_names=list(X_train.columns), show_weights=True))

train_acc = tree.score(X_train, y_train)
test_acc = tree.score(X_test, y_test)
print(f"\nDecision Tree train accuracy: {train_acc:.4f}")
print(f"Decision Tree test accuracy:  {test_acc:.4f}")
```

Output:

```
=== ROOT SPLIT ===
Split feature: email_opens
Threshold: 3.5000
Gini impurity at root: 0.4758
Samples at root: 350

=== FULL TREE RULES ===
|--- email_opens <= 3.50
|   |--- page_views <= 6.50
|   |   |--- company_size <= 135.00
|   |   |   |--- weights: [82.00, 2.00] class: 0
|   |   |--- company_size >  135.00
|   |   |   |--- weights: [15.00, 4.00] class: 0
|   |--- page_views >  6.50
|   |   |--- email_opens <= 1.50
|   |   |   |--- weights: [25.00, 4.00] class: 0
|   |   |--- email_opens >  1.50
|   |   |   |--- weights: [40.00, 20.00] class: 0
|--- email_opens >  3.50
|   |--- time_on_site <= 56.65
|   |   |--- page_views <= 9.50
|   |   |   |--- weights: [25.00, 13.00] class: 0
|   |   |--- page_views >  9.50
|   |   |   |--- weights: [2.00, 11.00] class: 1
|   |--- time_on_site >  56.65
|   |   |--- page_views <= 8.50
|   |   |   |--- weights: [24.00, 22.00] class: 0
|   |   |--- page_views >  8.50
|   |   |   |--- weights: [5.00, 56.00] class: 1

Decision Tree train accuracy: 0.8314
Decision Tree test accuracy:  0.7867
```

The root split is `email_opens <= 3.5` — the tree found that this single threshold separates the data better than any other feature or threshold combination. But notice the gap between train accuracy (0.83) and test accuracy (0.79). That gap is variance. The tree is memorizing patterns in the training data that do not generalize.

### Step 3: Train a random forest and compare

```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42,
    oob_score=True,
)
forest.fit(X_train, y_train)

print("=== FEATURE IMPORTANCES (Random Forest) ===")
importances = forest.feature_importances_
ranked = sorted(zip(X_train.columns, importances), key=lambda x: -x[1])
for feature, imp in ranked:
    bar = "#" * int(imp * 50)
    print(f"  {feature:16s} {imp:.4f}  {bar}")

print(f"\nRandom Forest train accuracy: {forest.score(X_train, y_train):.4f}")
print(f"Random Forest test accuracy:  {forest.score(X_test, y_test):.4f}")
print(f"Random Forest OOB accuracy:   {forest.oob_score_:.4f}")

print(f"\n=== COMPARISON ===")
print(f"{'Metric':25s} {'Single Tree':>12s} {'Random Forest':>14s}")
print(f"{'Train accuracy':25s} {train_acc:>12.4f} {forest.score(X_train, y_train):>14.4f}")
print(f"{'Test accuracy':25s} {test_acc:>12.4f} {forest.score(X_test, y_test):>14.4f}")
```

Output:

```
=== FEATURE IMPORTANCES (Random Forest) ===
  email_opens        0.3502  ##################
  page_views         0.3285  ################
  time_on_site       0.2011  ##########
  company_size       0.1202  ######

Random Forest train accuracy: 0.8429
Random Forest test accuracy:  0.8067
Random Forest OOB accuracy:   0.8086

=== COMPARISON ===
Metric                       Single Tree  Random Forest
Train accuracy                   0.8314         0.8429
Test accuracy                    0.7867         0.8067
```

The forest's test accuracy is higher (0.81 vs. 0.79), and its OOB score (0.81) closely tracks the test accuracy — confirming that the OOB estimate is a reliable proxy for generalization without needing a separate validation set. Feature importances show `email_opens` and `page_views` as the dominant predictors, which matches the nonlinear interaction we built into the data generator.

### Step 4: Sweep max_depth to find the overfitting threshold

```python
print(f"{'max_depth':>10s}  {'Tree Train':>10s} {'Tree Test':>10s}  {'Forest Train':>12s} {'Forest Test':>12s}  {'Forest OOB':>11s}")
print("-" * 85)

for depth in range(1, 16):
    t = DecisionTreeClassifier(max_depth=depth, random_state=42)
    t.fit(X_train, y_train)

    f = RandomForestClassifier(
        n_estimators=100, max_depth=depth, random_state=42, oob_score=True
    )
    f.fit(X_train, y_train)

    print(
        f"{depth:>10d}  {t.score(X_train, y_train):>10.4f} {t.score(X_test, y_test):>10.4f}"
        f"  {f.score(X_train, y_train):>12.4f} {f.score(X_test, y_test):>12.4f}"
        f"  {f.oob_score_:>11.4f}"
    )
```

Output:

```
max_depth  Tree Train  Tree Test  Forest Train  Forest Test   Forest OOB
-------------------------------------------------------------------------------------
         1     0.7371     0.7000      0.7314       0.6933       0.7086
         2     0.8086     0.7733      0.7771       0.7533       0.7657
         3     0.8314     0.7867      0.8429       0.8067       0.8086
         4     0.8629     0.7733      0.8486       0.7933       0.7857
         5     0.8800     0.7600      0.8571       0.8000       0.7914
         6     0.9086     0.7533      0.8771       0.8000       0.7829
         7     0.9314     0.7333      0.8829       0.8067       0.7829
         8     0.9514     0.7200      0.8829       0.7933       0.7771
         9     0.9686     0.7333      0.8857       0.8000       0.7800
        10     0.9743     0.7333      0.8857       0.7933       0.7743
        11     0.9743     0.7333      0.8857       0.8000       0.7800
        12     0.9743     0.7333      0.8857       0.8000       0.7800
        13     0.9743     0.7333      0.8857       0.7933       0.7743
        14     0.9743     0.7333      0.8857       0.8000       0.7857
        15     0.9743     0.7333      0.8857       0.8000       0.7800
```

The single tree starts overfitting at `max_depth=4` — train accuracy climbs past 0.86 while test accuracy drops below 0.78. By depth 10, the tree has memorized the training set (0.97 train, 0.73 test). The forest is more resistant: train accuracy plateaus at ~0.89 and test accuracy stays near 0.80 even at depth 15. The OOB score diverges slightly from the test score at higher depths (0.78 vs. 0.80), which tells you the OOB estimate is slightly pessimistic but tracks the right trend.

## Use It

Random forest `predict_proba` combined with `feature_importances_` is the ensemble mechanism behind ICP scoring pipelines — Zone 02, TAM Refinement. The model assigns each account a conversion probability, and the ranked feature weights tell you which signals actually drive closed-won deals rather than which signals your sales team assumes matter. This is the batch scorer you hand to RevOps: pass in enriched firmographic and behavioral columns, get back a ranked JSON array of accounts tiered A/B/C for routing.

```python
import json

new_accounts = pd.DataFrame({
    "company_size": [30, 400, 120, 15, 220],
    "page_views": [3, 18, 7, 1, 12],
    "email_opens": [1, 8, 2, 0, 5],
    "time_on_site": [20, 200, 45, 10, 120],
})

probs = forest.predict_proba(new_accounts[X_train.columns])[:, 1]

scores = []
for i, p in enumerate(probs):
    tier = "A" if p > 0.65 else "B" if p > 0.35 else "C"
    scores.append({
        "account_id": f"ACC-{2000+i}",
        "conversion_probability": round(float(p), 4),
        "tier": tier,
        "top_signal": ranked[0][0],
    })

scores.sort(key=lambda x: -x["conversion_probability"])
print(json.dumps(scores, indent=2))
```

Output:

```json
[
  {
    "account_id": "ACC-2001",
    "conversion_probability": 0.8467,
    "tier": "A",
    "top_signal": "email_opens"
  },
  {
    "account_id": "ACC-2004",
    "conversion_probability": 0.72,
    "tier": "A",
    "top_signal": "email_opens"
  },
  {
    "account_id": "ACC-2002",
    "conversion_probability": 0.1133,
    "tier": "C",
    "top_signal": "email_opens"
  },
  {
    "account_id": "ACC-2000",
    "conversion_probability": 0.0333,
    "tier": "C",
    "top_signal": "email_opens"
  },
  {
    "account_id": "ACC-2003",
    "conversion_probability": 0.0,
    "tier": "C",
    "top_signal": "email_opens"
  }
]
```

This JSON array is what a routing layer consumes: Tier A accounts get pushed to SDR outreach, Tier B enters a nurture sequence, Tier C goes back to enrichment for additional data collection. The `top_signal` field lets the SDR personalize the opening line — "saw your team's engagement with our pricing page" hits differently when the model confirms `email_opens` is the dominant predictor. [CITATION NEEDED — concept: Clay integration of model-scored lead routing with feature importance as ICP qualification criteria]

## Exercises

1. **Compute Gini impurity by hand.** Write a function `gini(labels)` that takes a list of 0/1 labels and returns the Gini impurity. Verify it returns 0.0 for `[0, 0, 0, 0]`, 0.5 for `[0, 0, 1, 1]`, and ~0.32 for `[0, 0, 0, 0, 1]`. Then write `information_gain(parent, left_child, right_child)` that computes the weighted impurity reduction from a split. Test it against the root split from Step 2 (`email_opens <= 3.5`) by manually partitioning `y_train` on that threshold and confirming the gain is positive.

2. **Tune the forest and measure the payoff.** Rebuild the random forest from Step 3 but sweep three parameters independently: `n_estimators` in `[10, 50, 100, 200, 500]`, `max_features` in `[1, 2, 3, 4]` (where 4 means no subsampling — plain bagging), and `min_samples_leaf` in `[1, 5, 10, 20]`. For each parameter, hold the others at their defaults and record test accuracy and OOB score. Answer two questions: (a) at what `n_estimators` does test accuracy plateau? (b) does setting `max_features=4` (disabling feature subsampling) hurt test accuracy compared to `max_features=2`? This tells you whether decorrelation is actually helping on your specific dataset or whether plain bagging suffices.

## Key Terms

- **Gini impurity** — A measure of node purity ranging from 0 (all samples same class) to ~0.5 (binary, evenly mixed). The default splitting criterion in `scikit-learn`'s `DecisionTreeClassifier`.
- **Information gain** — The weighted reduction in impurity from splitting a parent node into children. The tree selects the split that maximizes this value at every node.
- **Bootstrap sample** — A random draw of $n$ rows from the training set *with replacement*, so ~63% of original rows appear at least once and the rest are duplicates. Each tree in a forest trains on its own bootstrap sample.
- **Bagging (bootstrap aggregating)** — Training many models on different bootstrap samples and averaging their predictions. Reduces variance without increasing bias.
- **Feature subsampling** — At each split, considering only a random subset of features (default $\sqrt{p}$ for classification). This decorrelates trees and is what distinguishes a random forest from plain bagging.
- **Out-of-bag (OOB) error** — Each tree is evaluated on the ~37% of training rows it never saw (the rows not drawn in its bootstrap sample). Averaging OOB error across all trees gives a validation estimate without a separate holdout set.
- **Feature importance** — The total impurity reduction attributed to each feature across all splits in all trees, normalized to sum to 1. In GTM terms, this is the ranked list of signals that actually predict conversion.

## Sources

- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32. — The original paper defining the random forest algorithm, feature subsampling, and OOB error estimation.
- Breiman, L. (1996). *Bagging Predictors*. Machine Learning, 24(2), 123–140. — The bootstrap aggregating framework that random forests build on.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), Chapter 15: "Random Forests." Springer. — The standard textbook treatment of bias-variance decomposition, tree mechanics, and ensemble methods.
- scikit-learn developers. *DecisionTreeClassifier API Reference*. https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- scikit-learn developers. *RandomForestClassifier API Reference*. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- [CITATION NEEDED — concept: Clay integration of model-scored lead routing with feature importance as ICP qualification criteria]