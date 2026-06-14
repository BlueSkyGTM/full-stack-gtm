## Ship It

The from-scratch implementation was for understanding. In production, use `scikit-learn`'s `MultinomialNB` — it is faster (C-backed), tested, and plugs into the standard pipeline API. For imbalanced GTM data (you have 5× more newsletters than sales-intent emails), consider `ComplementNB`, which was designed specifically for imbalanced classes by correcting the decision boundary toward the minority class.

The production pipeline is: raw text → vectorizer (count or TF-IDF) → Naive Bayes → predicted class + probability. The vectorizer converts text into the numeric matrix Naive Bayes needs. `CountVectorizer` produces raw word counts (matches Multinomial NB's assumptions). `TfidfVectorizer` produces TF-IDF weights — these are not true counts, but Multinomial NB still works well on them in practice because the relative magnitudes preserve the ranking information.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

emails = [
    ("need enterprise pricing for 200 seats", "sales_intent"),
    ("demo request for our revops team", "sales_intent"),
    ("pricing comparison vs competitor", "sales_intent"),
    ("want to run a proof of concept", "sales_intent"),
    ("can we schedule a technical demo", "sales_intent"),
    ("how much does the platform cost", "sales_intent"),
    ("quarterly product newsletter inside", "newsletter"),
    ("new blog post on revops trends", "newsletter"),
    ("webinar replay from last week", "newsletter"),
    ("monthly digest of saas news", "newsletter"),
    ("year in review blog roundup", "newsletter"),
    ("product update v2 released", "newsletter"),
    ("CONGRATULATIONS you won a prize", "spam"),
    ("limited offer click here now", "spam"),
    ("free gift card claim today", "spam"),
    ("exclusive deal just for you", "spam"),
    ("you have been selected winner", "spam"),
    ("urgent act now last chance", "spam"),
]

texts = [t for t, _ in emails]
labels = [l for _, l in emails]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(lowercase=True, stop_words="english", min_df=1)),
    ("classifier", MultinomialNB(alpha=0.5)),
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)

print("=== TEST SET PREDICTIONS ===")
for i, (text, true_label, pred_label, probs) in enumerate(zip(X_test, y_test, y_pred, y_proba)):
    classes = pipeline.classes_
    prob_str = ", ".join(f"{c}={p:.2f}" for c, p in zip(classes, probs))
    match = "✓" if true_label == pred_label else "✗"
    print(f"  {match} '{text[:45]:45s}' | true={true_label:14s} pred={pred_label:14s} | {prob_str}")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, zero_division=0))

print("=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
print(f"  Classes: {list(pipeline.classes_)}")
print(f"  Matrix:\n{cm}")

joblib.dump(pipeline, "lead_intent_nb.joblib")
print("\n=== MODEL SAVED ===")
print("  File: lead_intent_nb.joblib")

loaded = joblib.load("lead_intent_nb.joblib")
new_email = "what are your enterprise prices"
pred = loaded.predict([new_email])[0]
probs = loaded.predict_proba([new_email])[0]
print(f"\n=== INFERENCE ON NEW EMAIL ===")
print(f"  Input: '{new_email}'")
print(f"  Predicted: {pred}")
print(f"  Probabilities: {dict(zip(loaded.classes_, [round(p,4) for p in probs]))}")
```

Two production concerns specific to GTM data:

**Threshold tuning.** The default decision rule picks the class with the highest probability. On imbalanced data, this is often wrong. If 70% of your inbound is newsletters, the prior alone pushes everything toward "newsletter" and sales-intent emails get misrouted. The fix: instead of `predict()`, use `predict_proba()` and set custom thresholds per class. For sales intent, require confidence ≥ 0.6 before auto-routing to an AE; anything below goes to an LLM for deeper analysis. The threshold is a business decision — false negatives (missed leads) cost more than false positives (wasted AE time), so bias toward sensitivity.

```python
print("\n=== THRESHOLD TUNING DEMO ===")
test_cases = [
    "pricing for enterprise",
    "check out our newsletter",
    "free gift card",
    "maybe interested eventually",
]
THRESHOLD = 0.6
for text in test_cases:
    probs = loaded.predict_proba([text])[0]
    classes = loaded.classes_
    best_idx = np.argmax(probs)
    best_class = classes[best_idx]
    best_prob = probs[best_idx]

    if best_prob >= THRESHOLD:
        action = f"auto-route to {best_class}"
    else:
        action = f"escalate to LLM (best={best_class} at {best_prob:.2f})"

    print(f"  '{text:35s}' -> {action}")
```

**Confusion matrix as diagnostic.** The confusion matrix tells you *which* classes are confused with each other, which is more actionable than accuracy. If "sales_intent" is consistently confused with "newsletter," your training data probably lacks enough sales-intent examples with newsletter-like vocabulary. The fix is data, not model tuning. This is why GTM engineers who own classification pipelines also own the feedback loop: misclassified emails get labeled and fed back into training data weekly.