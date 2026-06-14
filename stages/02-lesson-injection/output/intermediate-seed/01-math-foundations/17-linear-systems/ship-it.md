## Ship It

In production, your signal matrix will have missing values, collinear columns, and noise. Company revenue and employee count are correlated — sometimes at r > 0.9 — which means the two columns carry nearly identical information. The solver sees this as near-singularity: small perturbations in the input data produce wild swings in the computed weights. One week your model says revenue matters most; next week, after one account changes, it says employee count matters most. Both are wrong because the matrix cannot distinguish the two signals.

The condition number quantifies this. `np.linalg.cond(A)` returns the ratio of the largest to smallest singular value. A condition number of 1 means all columns are perfectly orthogonal (ideal). A condition number of 10^10 means the matrix is effectively singular for floating-point purposes — the solver will return numbers, but they are noise. Here is a production-grade scorer that validates the matrix before solving and degrades gracefully:

```python
import numpy as np

def production_scorer(raw_signals, outcomes, new_accounts, cond_threshold=1e4):
    A = np.array(raw_signals, dtype=float)
    b = np.array(outcomes, dtype=float)
    new = np.array(new_accounts, dtype=float)
    
    if np.any(np.isnan(A)) or np.any(np.isnan(b)):
        print("FAIL: Missing values detected. Impute before scoring.")
        return None
    
    cond = np.linalg.cond(A)
    rank = np.linalg.matrix_rank(A)
    n_cols = A.shape[1]
    
    print(f"Shape: {A.shape[0]} accounts x {n_cols} signals")
    print(f"Rank: {rank} / {n_cols}")
    print(f"Condition number: {cond:.2e}")
    
    if rank < n_cols:
        print(f"WARN: {n_cols - rank} column(s) are linearly dependent.")
        print("      Dropping collinear signals or using PCA before solving.")
        u, s, vt = np.linalg.svd(A, full_matrices=False)
        A_reduced = u[:, :rank] @ np.diag(s[:rank])
        weights, _, _, _ = np.linalg.lstsq(A_reduced, b, rcond=None)
        print(f"      Solved in reduced {rank}-dim space.")
    elif cond > cond_threshold:
        print(f"WARN: Condition number {cond:.2e} exceeds threshold {cond_threshold:.0e}.")
        print("      Applying ridge regularization (lambda=1.0).")
        lam = 1.0
        A_reg = A.T @ A + lam * np.eye(n_cols)
        b_reg = A.T @ b
        weights = np.linalg.solve(A_reg, b_reg)
        print("      Solved with ridge regularization.")
    else:
        weights, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        print("      Solved with ordinary least squares.")
    
    scores = new @ weights
    print(f"\nWeights: {np.round(weights, 4)}")
    print(f"New scores: {np.round(scores, 3)}")
    
    return {"weights": weights, "condition": cond, "scores": scores}

raw_signals = [
    [50, 0.4, 85],
    [200, 0.7, 30],
    [15, 0.2, 90],
    [300, 0.9, 10],
    [80, 0.5, 60],
    [120, 0.6, 45],
    [250, 0.8, 20],
    [30, 0.3, 75],
]

outcomes = [0, 1, 0, 1, 0, 1, 1, 0]

new_accounts = [[100, 0.5, 50], [180, 0.7, 35], [20, 0.1, 95]]

result = production_scorer(raw_signals, outcomes, new_accounts)

collinear_signals = [
    [50, 49, 0.4, 85],
    [200, 201, 0.7, 30],
    [15, 14, 0.2, 90],
    [300, 298, 0.9, 10],
    [80, 81, 0.5, 60],
]

collinear_outcomes = [0, 1, 0, 1, 0]
collinear_new = [[100, 101, 0.5, 50]]

print("\n--- Collinear case ---")
result2 = production_scorer(collinear_signals, collinear_outcomes, collinear_new)
```

Run this and compare the two cases. The first produces stable weights because the three signals are reasonably independent. The second triggers the collinearity warning because columns one and two are nearly identical — the rank drops, and the solver either reduces the space via SVD or applies ridge regularization to stabilize the weights.

Ridge regularization works by adding a penalty term λI to the normal equations: instead of solving AᵀAx = Aᵀb, you solve (AᵀA + λI)x = Aᵀb. This shrinks the weights toward zero, which trades a small amount of bias for a large reduction in variance. In practice, this means your scoring model becomes more stable across different snapshots of data — the weights do not swing wildly when one account enters or leaves the dataset. The cost is that the model slightly underfits, which is almost always preferable to the alternative of a model that fits perfectly on training data but produces nonsense on new accounts.

The natural extension of this lesson is eigenvalues and eigenvectors, which reveal which dimensions of your signal matrix carry independent information. When you have dozens of firmographic signals and need to collapse them into a smaller set, PCA (principal component analysis) uses eigenvectors of the covariance matrix to find the directions of maximum variance. SVD generalizes this to non-square matrices and is what powers recommender systems and data-driven attribution models. [CITATION NEEDED — concept: PCA for feature reduction in CRM enrichment pipelines]