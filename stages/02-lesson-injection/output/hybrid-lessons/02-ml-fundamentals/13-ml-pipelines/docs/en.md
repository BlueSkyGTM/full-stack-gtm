# ML Pipelines

## Learning Objectives

- Build a scikit-learn `Pipeline` that chains imputation, scaling, encoding, and model fitting into a single serializable object
- Detect data leakage scenarios where fitted preprocessing parameters from test data contaminate training, and explain how `Pipeline` prevents this by isolating fit state to training folds
- Construct a `ColumnTransformer` that routes different transformers to numeric and categorical feature subsets within a single pipeline
- Serialize a fitted pipeline with `joblib` and verify that the reloaded object produces identical predictions on new data
- Compare in-process pipelines (scikit-learn) against orchestrator-level pipelines (Airflow, ZenML) and justify when each is appropriate for GTM scoring systems

## The Problem

You have a notebook that loads data, fills missing values with the median, scales features, one-hot encodes a categorical column, trains a logistic regression, and prints accuracy. It scores 0.87. You ship the model weights to production. A week later, the predictions coming back from the inference endpoint look wrong — not catastrophically wrong, but drifted enough that the sales team has stopped trusting the scores. What happened?

The median you computed included the test set. The scaler's mean and variance were calculated on the full dataset, not just the training split. The one-hot encoder produced a different set of columns in training (where it saw all categories) versus in the notebook you copy-pasted into the serving script (where the column order was different because pandas reindexed it). The model weights are correct, but the data flowing into `model.predict(X)` at inference time has been transformed by a different set of fitted parameters than the data the model was trained on. The model is fine. The preprocessing is not, and nobody noticed because the score still looked reasonable.

```python
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = np.random.randint(0, 100, size=(200, 3)).astype(float)
X[np.random.rand(*X.shape) < 0.1] = np.nan
y = (X[:, 0] > 50).astype(int)
y[np.random.rand(len(y)) < 0.1] = 1 - y[np.random.rand(len(y)) < 0.1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

model = LogisticRegression(random_state=42).fit(X_train_scaled, y_train)
clean_score = model.score(X_test_scaled, y_test)

imputer_leak = SimpleImputer(strategy='median')
X_all_imp = imputer_leak.fit_transform(np.vstack([X_train, X_test]))
X_all_scaled = StandardScaler().fit_transform(X_all_imp)
X_train_leak = X_all_scaled[:len(X_train)]
X_test_leak = X_all_scaled[len(X_train):]
model_leak = LogisticRegression(random_state=42).fit(X_train_leak, y_train)
leak_score = model_leak.score(X_test_leak, y_test)

print(f"Clean pipeline score:  {clean_score:.4f}")
print(f"Leaked pipeline score: {leak_score:.4f}")
print(f"Difference:            {abs(clean_score - leak_score):.4f}")
```

```
Clean pipeline score:  0.7500
Leaked pipeline score: 0.7667
Difference:            0.0167
```

The leaked score is higher — and that is the worst outcome, because it reinforces the bad practice. The model has seen information about the test distribution through the imputer's median and the scaler's mean and variance. When new data arrives in production that doesn't match those leaked statistics, performance will degrade silently. This is not a hypothetical scenario. It is the single most common reason ML scoring systems produce different results in production than they did in development, and it happens because preprocessing logic is scattered across notebooks, scripts, and serving code rather than packaged into one object with one fit path.

## The Concept

A pipeline is a directed acyclic graph of executable steps where each step consumes the output of the previous step. In scikit-learn's implementation, a `Pipeline` is an ordered list of `(name, transformer)` tuples ending with an estimator. The entire object exposes a single `.fit()` method and a single `.predict()` method, and internally it applies every transformation in sequence before passing data to the next step. The critical mechanism is the distinction between what happens during `fit` and what happens during `predict`: during `fit`, every intermediate transformer calls `fit_transform` (learning its parameters from the data it receives, then transforming), and the final estimator calls `fit` on the fully transformed output. During `predict`, every intermediate transformer calls only `transform` — no fitting occurs. The parameters learned during `fit` are frozen in the transformer's internal state and reused exactly.

```mermaid
flowchart TD
    subgraph "During fit()"
        A1[Raw X_train] -->|fit_transform| B1[Imputer: learns median]
        B1 -->|fit_transform| C1[Scaler: learns mean, std]
        C1 -->|fit_transform| D1[Encoder: learns categories]
        D1 -->|fit| E1[Model: learns weights]
    end
    subgraph "During predict()"
        A2[Raw X_new] -->|transform only| B2[Imputer: applies frozen median]
        B2 -->|transform only| C2[Scaler: applies frozen mean, std]
        C2 -->|transform only| D2[Encoder: applies frozen categories]
        D2 -->|predict| E2[Model: applies frozen weights]
    end
```

This mechanism is what prevents data leakage. Because `fit_transform` is called only on the data passed to `pipeline.fit(X_train, y_train)`, every transformer learns its parameters exclusively from the training split. When you later call `pipeline.predict(X_test)`, those same parameters — the imputer's median, the scaler's mean and standard deviation, the encoder's vocabulary — are applied via `transform` without refitting. The test data is processed by parameters it never influenced. If you instead computed the median on the full dataset before splitting, you would leak the test distribution's information into the training process, and the model would appear to perform better than it will on genuinely unseen data.

The `ColumnTransformer` extends this mechanism to handle the common case where different columns require different preprocessing. Numeric columns might need imputation followed by scaling; categorical columns might need imputation followed by one-hot encoding. A `ColumnTransformer` takes a list of `(name, transformer, columns)` tuples, applies each transformer to its specified columns, and concatenates the results. Each sub-transformer follows the same fit/transform discipline — its parameters are learned from the training data during `pipeline.fit()` and frozen for inference. The `make_pipeline` convenience function constructs a `Pipeline` from an unordered list of transformers without requiring names, auto-generating names from the class names; it is functionally identical to `Pipeline` but produces less readable keys when you need to access individual steps via `pipeline.named_steps`.

There is an important distinction between in-process pipelines and orchestrator-level pipelines. A scikit-learn `Pipeline` runs within a single Python process: it chains transformations and model fitting in memory, and it can be serialized with `joblib` to a single file. This is the right tool when your problem is "apply the same preprocessing at train and inference time within one service." Tools like Airflow, ZenML, or Kubeflow Pipelines operate at a different level — they orchestrate separate jobs (data extraction, feature computation, model training, evaluation, deployment) that may run on different machines, in different containers, on different schedules. You use an orchestrator when your pipeline needs to span infrastructure boundaries, run on a schedule, retry failed steps independently, or coordinate between teams. For the common case of "preprocess features and score a model in one request handler," a scikit-learn pipeline serialized with `joblib` is the correct choice and the orchestrator is overkill.

## Build It

The pipeline below chains a `ColumnTransformer` (which itself contains two sub-pipelines for numeric and categorical features) into a `LogisticRegression` estimator. Every preprocessing step — imputation, scaling, one-hot encoding — is inside the pipeline. There is no preprocessing code outside it.

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 300
data = {
    'employees': np.random.choice([10, 50, 200, 500, 1000, np.nan], size=n, p=[0.2, 0.25, 0.2, 0.15, 0.1, 0.1]),
    'revenue': np.random.lognormal(mean=13, sigma=1.2, size=n).round(-3),
    'industry': np.random.choice(['SaaS', 'Finance', 'Healthcare', 'Manufacturing', np.nan], size=n),
    'region': np.random.choice(['NA', 'EMEA', 'APAC'], size=n)
}
df = pd.DataFrame(data)
df['revenue'] = df['revenue'].replace([np.inf, -np.inf], np.nan)
conversion_prob = (
    0.15
    + 0.3 * (df['employees'] > 200).astype(float)
    + 0.2 * (df['revenue'] > 500000).astype(float)
    + 0.15 * (df['industry'] == 'SaaS').astype(float)
).fillna(0.15)
y = (np.random.rand(n) < conversion_prob).astype(int)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)

numeric_features = ['employees', 'revenue']
categorical_features = ['industry', 'region']

numeric_transformer = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

categorical_transformer = Pipeline([
    ('impute', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

clf_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

clf_pipeline.fit(X_train, y_train)

train_score = clf_pipeline.score(X_train, y_train)
test_score = clf_pipeline.score(X_test, y_test)

print(f"Train accuracy: {train_score:.4f}")
print(f"Test accuracy:  {test_score:.4f}")
print(f"Gap:            {train_score - test_score:.4f}")

sample = X_test.head(3)
predictions = clf_pipeline.predict(sample)
probabilities = clf_pipeline.predict_proba(sample)[:, 1]

for i in range(len(sample)):
    print(f"\nSample {i}: employees={sample.iloc[i]['employees']}, "
          f"revenue={sample.iloc[i]['revenue']:.0f}, "
          f"industry={sample.iloc[i]['industry']}")
    print(f"  Predicted class: {predictions[i]}, probability: {probabilities[i]:.4f}")
```

```
Train accuracy: 0.8286
Test accuracy:  0.8000
Gap:            0.0286

Sample 0: employees=50.0, revenue=558000.0, industry=Healthcare
  Predicted class: 0, probability: 0.2743

Sample 1: employees=200.0, revenue=3872000.0, industry=SaaS
  Predicted class: 1, probability: 0.8572

Sample 2: employees=1000.0, revenue=555000.0, industry=SaaS
  Predicted class: 1, probability: 0.8163
```

Notice what is not in this code: there is no separate `imputer.fit(X_train)` call, no `scaler.fit(X_train)` call, no manual column alignment between training and test. The pipeline handles all of it. The `ColumnTransformer` routes `employees` and `revenue` through the numeric sub-pipeline, and `industry` and `region` through the categorical sub-pipeline, then concatenates the outputs into a single feature matrix that `LogisticRegression` consumes. When you call `clf_pipeline.predict(X_test)`, the same median values, scaling parameters, and one-hot encoding vocabulary learned during `fit` are applied — no refitting occurs.

The `make_pipeline` convenience produces the same object with auto-generated step names. The difference is readability of `named_steps` access and hyperparameter tuning keys. When you tune hyperparameters with `GridSearchCV`, you reference parameters as `step_name__param_name` (e.g., `classifier__C`). With `Pipeline`, you control the names; with `make_pipeline`, they are derived from class names (e.g., `logisticregression__C`), which can be brittle if you swap estimators.

```python
quick_pipe = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler(),
    LogisticRegression(random_state=42)
)

X_quick = df[['employees', 'revenue']].copy()
X_quick_train, X_quick_test, y_quick_train, y_quick_test = train_test_split(
    X_quick, y, test_size=0.3, random_state=42
)

quick_pipe.fit(X_quick_train,