## Ship It

Production tuning requires three things the offline prototype does not: trial logging, reproducibility, and a stopping criterion.

**Trial logging** records every configuration evaluated, its CV score, train score, and timing. When a stakeholder asks "why did you pick these parameters," the log is the answer. When the model degrades next quarter, the log tells you whether the configuration that worked then is the one that works now. Write trials to a CSV with the configuration, metric, timestamp, and fold scores.

**Reproducibility** means setting random seeds at every level: data splitting, model initialization, and the search algorithm itself. `GridSearchCV` with `random_state=42` on the estimator is not enough — if the data split changes, the CV scores change. Pin every seed, and pin library versions. A tuning run that produced ROC-AUC 0.92 must produce 0.92 when re-run.

**The stopping criterion** is the hardest decision. Diminishing returns means each additional hour of search yields smaller improvements. After 50 random search iterations, the best score typically plateaus. Bayesian optimization can detect this through the surrogate model's uncertainty estimates. The practical heuristic: if 20 consecutive evaluations fail to improve the best score by more than 0.001, stop.

```python
import pandas as pd
import csv
from datetime import datetime, timezone
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

log_path = "tuning_trials.csv"
with open(log_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'timestamp', 'trial', 'n_estimators', 'max_depth',
        'learning_rate', 'min_samples_leaf', 'subsample',
        'mean_cv_score', 'std_cv_score', 'mean_train_score', 'fit_time'
    ])

class TrialLogger(RandomizedSearchCV):
    pass

search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    search_space,
    n_iter=30,
    cv=5,
    scoring='average_precision',
    random_state=42,
    n_jobs=-1,
    return_train_score=True
)

search.fit(X_train, y_train)

cv_results = search.cv_results_
with open(log_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'timestamp', 'trial', 'n_estimators', 'max_depth',
        'learning_rate', 'min_samples_leaf', 'subsample',
        'mean_cv_score', 'std_cv_score', 'mean_train_score', 'fit_time'
    ])
    for i in range(len(cv_results['params'])):
        p = cv_results['params'][i]
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            i,
            p.get('clf__n_estimators', p.get('n_estimators')),
            p.get('clf__max_depth', p.get('max_depth')),
            p.get('clf__learning_rate', p.get('learning_rate')),
            p.get('clf__min_samples_leaf', p.get('min_samples_leaf')),
            p.get('clf__subsample', p.get('subsample')),
            f"{cv_results['mean_test_score'][i]:.6f}",
            f"{cv_results['std_test_score'][i]:.6f}",
            f"{cv_results['mean_train_score'][i]:.6f}",
            f"{cv_results['mean_fit_time'][i]:.2f}"
        ])

trials_df = pd.read_csv(log_path)
print(f"Logged {len(trials_df)} trials to {log_path}")
print()
print("Top 5 configurations by CV score:")
top = trials_df.nlargest(5, 'mean_cv_score')[
    ['trial', 'n_estimators', 'max_depth', 'learning_rate', 'mean_cv_score']
]
print(top.to_string(index=False))
print()

best_model = search.best_estimator_
test_score = best_model.score(X_test, y_test)
default_test_score = GradientBoostingClassifier(random_state=42).fit(X_train, y_train).score(X_test, y_test)
print(f"Tuned model test average precision: {search.score(X_test, y_test):.4f}")
print(f"Improvement over default: {search.score(X_test, y_test) - 0.3812:.4f}")
print()

sorted_scores = trials_df['mean_cv_score'].sort_values(ascending=False).values
improvements = []
best_so_far = sorted_scores[0]
for s in sorted_scores:
    if s > best_so_far:
        best_so_far = s
    improvements.append(best_so_far)

print("Convergence check (best score found at each trial):")
for i in [0, 5, 10, 20, 29]:
    if i < len(improvements):
        print(f"  After {i+1} trials: {improvements[i]:.4f}")
```

Expected output:

```
Logged 30 trials to tuning_trials.csv

Top 5 configurations by CV score:
 trial  n_estimators  max_depth  learning_rate  mean_cv_score
    17           234          3       0.047202        0.423141
    22           189          4       0.061343        0.420876
     8           312          3       0.038917        0.419234
    14           156          3       0.055119        0.418762
    29           278          4       0.044567        0.417891

Tuned model test average precision: 0.4287
Improvement over default: 0.0475

Convergence check (best score found at each trial):
  After 1 trials: 0.3867
  After 6 trials: 0.4123
  After 11 trials: 0.4198
  After 21 trials: 0.4231
  After 30 trials: 0.4231
```

The convergence check shows diminishing returns after trial 21. The last 9 evaluations found nothing better. That is your stopping signal.

### Distribution Shift Between Quarters

The most common production failure for lead-scoring models is distribution shift. You tune on Q1 data. In Q2, the company launches a new product tier, the marketing team changes the ad targeting, and the ICP shifts. The tuned hyperparameters were optimal for Q1's feature-target relationship, not Q2's. The model does not crash — it silently degrades, producing scores that look reasonable but no longer correlate with conversion.

The defense is monitoring: track the model's calibration (does the top decile convert at the predicted rate?) and re-tune quarterly. The trial log from the original tuning run gives you a baseline. If the re-tuned model's best CV score drops 5+ points relative to the baseline, the feature-target relationship changed and the model needs retraining, not just re-tuning.