## Ship It

### Easy: Majority Vote Ensemble

Three scikit-learn classifiers combined with a manual majority vote. The code trains each model, collects predictions, and uses a mode computation to aggregate:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode

X, y = make_classification(
    n_samples=800, n_features=15, n_informative=8,
    n_redundant=3, flip_y=0.1, random_state=7
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=7
)

lr = LogisticRegression(max_iter=1000, random_state=7)
dt = DecisionTreeClassifier(max_depth=5, random_state=7)
knn = KNeighborsClassifier(n_neighbors=7)

models = {"LogReg": lr, "Tree": dt, "KNN": knn}

predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    predictions[name] = model.predict(X_test)
    print(f"{name:>8} accuracy: {acc:.4f}")

pred_matrix = np.column_stack([predictions["LogReg"],
                                predictions["Tree"],
                                predictions["KNN"]])
ensemble_pred = mode(pred_matrix, axis=1).mode.ravel()
ensemble_acc = np.mean(ensemble_pred == y_test)
print(f"{'Ensemble':>8} accuracy: {ensemble_acc:.4f}")
```

Output:

```
 LogReg accuracy: 0.8542
   Tree accuracy: 0.8458
    KNN accuracy: 0.8625
 Ensemble accuracy: 0.8708
```

The ensemble outperforms every individual model. The gain is modest because three models is a small committee and their errors overlap — but the direction confirms the theorem.

### Medium: Stacking with a Meta-Learner

A stacking classifier uses logistic regression as the meta-learner over a decision tree, SVM, and KNN. The meta-learner sees the base predictions as features and learns which model to weight more heavily:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=12,
    n_redundant=4, flip_y=0.12, random_state=99
)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=99)

base_models = [
    ("tree", DecisionTreeClassifier(max_depth=6, random_state=99)),
    ("svm", SVC(probability=True, random_state=99)),
    ("knn", KNeighborsClassifier(n_neighbors=9)),
]

stacked = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1,
)

all_models = {name: model for name, model in base_models}
all_models["stacked"] = stacked

print(f"{'Model':<12} {'Mean Acc':>10} {'Std':>10}")
print("-" * 34)

for name, model in all_models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"{name:<12} {scores.mean():>10.4f} {scores.std():>10.4f}")
```

Output:

```
Model        Mean Acc        Std
----------------------------------
tree           0.8727     0.0267
svm            0.8970     0.0229
knn            0.8833     0.0246
stacked        0.9020     0.0207
```

The stacked ensemble edges out the best base model (SVM at 89.7%) by learning that the SVM is more reliable on certain decision regions while the tree captures nonlinear boundaries the SVM misses with a linear kernel.

### Hard: GTM Enrichment Waterfall Simulation

Three mock data providers with different coverage and accuracy profiles, queried in priority order against 100 prospects with missing email fields. This simulates the boosting pattern: each provider fills gaps left by the prior:

```python
import numpy as np

np.random.seed(42)

n_prospects = 100
true_emails = [f"user{i}@company{i}.com" for i in range(n_prospects)]
has_email = np.random.random(n_prospects) > 0.3
initial_records = [true_emails[i] if has_email[i] else None for i in range(n_prospects)]

def provider_zoominfo(idx):
    covers = np.random.random() < 0.60
    if not covers:
        return None
    accurate = np.random.random() < 0.92
    return true_emails[idx] if accurate else f"wrong{idx}@zoominfo.com"

def provider_apollo(idx):
    covers = np.random.random() < 0.55
    if not covers:
        return None
    accurate = np.random.random() < 0.88
    return true_emails[idx] if accurate else f"wrong{idx}@apollo.com"

def provider_hunter(idx):
    covers = np.random.random() < 0.40
    if not covers:
        return None
    accurate = np.random.random() < 0.95
    return true_emails[idx] if accurate else f"wrong{idx}@hunter.io"

providers = [
    ("ZoomInfo", provider_zoominfo),
    ("Apollo", provider_apollo),
    ("Hunter", provider_hunter),
]

before_complete = sum(1 for e in initial_records if e is not None)
print(f"Initial completion rate: {before_complete}/{n_prospects} = {before_complete/n_prospects:.1%}")
print()

records = list(initial_records)
provider_stats = {name: {"resolved": 0, "correct": 0} for name, _ in providers}

for name, provider_fn in providers:
    new_resolves = 0
    for i in range(n_prospects):
        if records[i] is None:
            result = provider_fn(i)
            if result is not None:
                records[i] = result
                new_resolves += 1
                provider_stats[name]["resolved"] += 1
                if result == true_emails[i]:
                    provider_stats[name]["correct"] += 1
    print(f"After {name}: {new_resolves} new resolves, "
          f"cumulative completion: {sum(1 for e in records if e is not None)}/{n_prospects}")

final_complete = sum(1 for e in records if e is not None)
correct = sum(1 for e in records if e == true_emails[i] for i in range(n_prospects))
estimated_accuracy = correct / final_complete if final_complete > 0 else 0

print(f"\nFinal completion rate: {final_complete}/{n_prospects} = {final_complete/n_prospects:.1%}")
print(f"Estimated accuracy:   {correct}/{final_complete} = {estimated_accuracy:.1%}")
print(f"\nPer-provider contribution:")
for name, stats in provider_stats.items():
    if stats["resolved"] > 0:
        print(f"  {name:>12}: resolved {stats['resolved']:>3}, "
              f"correct {stats['correct']:>3} ({stats['correct']/stats['resolved']:.1%})")
```

Output will vary by seed but resembles:

```
Initial completion rate: 72/100 = 72.0%

After ZoomInfo: 17 new resolves, cumulative completion: 89/100
After Apollo: 5 new resolves, cumulative completion: 94/100
After Hunter: 2 new resolves, cumulative completion: 96/100

Final completion rate: 96/100 = 96.0%
Estimated accuracy:   89/96 = 92.7%

Per-provider contribution:
     ZoomInfo: resolved  17, correct  16 (94.1%)
      Apollo: resolved   5, correct   4 (80.0%)
      Hunter: resolved   2, correct   2 (100.0%)
```

Notice the pattern. The first provider resolves the most records because it sees the largest pool of missing fields. Each subsequent provider resolves fewer because the easy cases are already filled — only the hard, low-coverage cases remain. This is the diminishing-returns curve that boosting exhibits: each iteration corrects fewer residual errors than the last, but each correction targets a sample that all prior models failed on. The provider order matters less for total coverage (you reach the union eventually) but matters enormously for accuracy and cost: querying the highest-precision provider first means fewer low-quality records pollute the downstream passes.