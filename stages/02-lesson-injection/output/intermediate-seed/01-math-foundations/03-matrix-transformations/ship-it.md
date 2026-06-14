## Ship It

Build a working account-feature transformer. The pipeline accepts a matrix of raw firmographic features, normalizes each column to zero mean and unit variance, projects the result into a 2D scoring space using a fixed projection matrix, and prints the scored vectors alongside rank information that confirms the projection preserved enough structure to differentiate accounts.

```python
import numpy as np

accounts = np.array([
    [  50,   500_000,  2, 1],
    [ 200,  2_000_000,  5, 2],
    [ 800,  8_000_000,  9, 3],
    [1500, 15_000_000, 12, 4],
    [5000, 60_000_000, 20, 7],
], dtype=float)

feature_names = ["employees", "revenue", "tech_count", "funding_stage"]

print("=== Raw Account Features ===")
print(f"Shape: {accounts.shape}")
print(f"{'Account':<10}", end="")
for name in feature_names:
    print(f"{name:<15}", end="")
print()
for i, row in enumerate(accounts):
    print(f"  #{i+1:<8}", end="")
    for val in row:
        print(f"{val:<15.1f}", end="")
print()

means = accounts.mean(axis=0)
stds = accounts.std(axis=0)
normalized = (accounts - means) / stds

print("\n=== Normalized Features ===")
print(f"Column means: {normalized.mean(axis=0).round(6)}")
print(f"Column stds:  {normalized.std(axis=0).round(6)}")

np.random.seed(42)
random_projection = np.random.randn(4, 2)
projection, _ = np.linalg.qr(random_projection)

scores = normalized @ projection

print("\n=== 2D Projection Matrix ===")
print(projection)
print(f"Projection shape: {projection.shape}")
print(f"Columns are orthonormal: {np.allclose(projection.T @ projection, np.eye(2))}")

print("\n=== Account Scores (2D) ===")
for i, score in enumerate(scores):
    magnitude = np.linalg.norm(score)
    print(f"  Account #{i+1}: ({score[0]:+.4f}, {score[1]:+.4f})  |v|={magnitude:.4f}")

distances_from_origin = np.linalg.norm(scores, axis=1)
ranked = np.argsort(distances_from_origin)[::-1]

print("\n=== Accounts Ranked by Score Magnitude ===")
for rank, idx in enumerate(ranked):
    print(f"  {rank+1}. Account #{idx+1} — magnitude {distances_from_origin[idx]:.4f}")

print(f"\nOriginal rank (matrix): {np.linalg.matrix_rank(accounts)}")
print(f"Normalized rank:        {np.linalg.matrix_rank(normalized)}")
print(f"Projected rank:         {np.linalg.matrix_rank(scores)}")
print("Rank dropped → information lost in projection (expected for dim reduction).")
```

The rank comparison at the end demonstrates a property you need to internalize: projecting from 4 dimensions to 2 inherently loses information. The rank drops because the projection matrix collapses the 4-dimensional feature space onto a 2-dimensional plane. Whether that loss is acceptable depends on whether the discarded dimensions carried signal or noise — and that is what eigenvalue analysis (covered in the next lesson) lets you quantify.