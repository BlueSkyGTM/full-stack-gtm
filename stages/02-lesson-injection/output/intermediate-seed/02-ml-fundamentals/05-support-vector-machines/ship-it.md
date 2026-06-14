## Ship It

Before you deploy an SVM to production, you need to understand its inference cost. SVM prediction time scales with the number of support vectors, not the training set size. Each prediction requires computing the kernel between the input and every support vector. A model trained on 100,000 rows might have 5,000 support vectors, and every single prediction runs 5,000 kernel evaluations. This is fundamentally different from logistic regression, where prediction is a single dot product regardless of training set size.

```python
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

np.random.seed(42)

for n in [1000, 5000, 20000]:
    X, y = make_classification(n_samples=n, n_features=20, n_informative=10,
                                n_redundant=5, random_state=42)
    y_svm = np.where(y == 0, -1, 1)

    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X, y_svm)

    start = time.time()
    preds = svm.predict(X)
    elapsed = time.time() - start

    print(f"n={n:6d} | support_vectors={len(svm.support_):5d} "
          f"({len(svm.support_)/n*100:.1f}% of train) | "
          f"predict_time={elapsed:.3f}s | "
          f"accuracy={accuracy_score(y_svm, preds):.4f}")
```

Run this and watch how the number of support vectors and prediction time grow. At 20,000 rows, if 15% of your training points become support vectors, you are running 3,000 kernel evaluations per prediction. For a real-time lead-scoring API that needs to score a lead in under 50ms, that budget disappears fast. The practical threshold: kernel SVMs on datasets above 50,000 rows become slow enough that you should either (a) switch to a linear SVM using `LinearSVC` which does not store support vectors for prediction—it collapses to a weight vector like logistic regression—or (b) switch to a different algorithm entirely.

For GTM workflows specifically, the SVM's value proposition is the margin as a confidence signal, not raw throughput. If you are scoring 500 inbound leads per day, SVM inference cost is irrelevant—even 5,000 support vectors times 500 predictions is 2.5 million kernel evaluations, which completes in under a second on modern hardware. If you are scoring 500,000 leads in a batch enrichment job against your full TAM, use `LinearSVC` or switch to a tree-based model. The margin confidence bucketing still works with `LinearSVC`—you get the decision function distance, just with a linear boundary instead of a kernel-induced nonlinear one.

One more production detail: SVMs do not natively output probabilities. The `decision_function` gives you a signed distance, not a probability. scikit-learn's `SVC` has a `probability=True` flag that internally runs Platt scaling (fitting a logistic regression on top of the SVM's decision function outputs via cross-validation), but this doubles your training time and the resulting probabilities are a post-hoc calibration, not a native output. For lead scoring, the raw decision function distance bucketed into high/medium/review is often more useful than a calibrated probability, because the distance has a geometric meaning (how far from the margin) while the Platt-scaled probability is a monotonic transform of that distance with no additional information content.