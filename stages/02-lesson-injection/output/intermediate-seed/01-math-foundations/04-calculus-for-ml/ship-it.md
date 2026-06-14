## Ship It

Take the trained regression and wrap it as a scoring function that could be called from a Clay webhook or an enrichment script. The model takes account features, applies the learned weights, and returns a score. In a real GTM pipeline, this score feeds into routing logic — does this account get routed to enterprise sales, to the SMB sequence, or to the nurture pool?

The outbound foundation — where every GTM engineering engagement begins — requires a list that reflects your ICP, enriched with signals that predict conversion [CITATION NEEDED — concept: outbound foundation requires ICP-aligned list with predictive signals]. A trained scoring model is one way to produce those signals from historical data.

```python
random.seed(42)
true_w, true_b = 3.0, 7.0
X_train = [random.uniform(0, 10) for _ in range(100)]
y_train = [true_w * xi + true_b + random.gauss(0, 1.5) for xi in X_train]

w, b = 0.0, 0.0
lr = 0.01
for _ in range(300):
    n = len(X_train)
    dw = sum(2 * (w * X_train[i] + b - y_train[i]) * X_train[i] for i in range(n)) / n
    db = sum(2 * (w * X_train[i] + b - y_train[i]) for i in range(n)) / n
    w = w - lr * dw
    b = b - lr * db

def score_account(feature_value, weight, bias):
    raw = weight * feature_value + bias
    return raw

test_accounts = [
    ("DataNest", 2.1),
    ("Acme Corp", 4.5),
    ("CloudPeak", 8.0),
    ("TechFlow", 9.5),
]

print("Account scoring model (feature = engagement score 0