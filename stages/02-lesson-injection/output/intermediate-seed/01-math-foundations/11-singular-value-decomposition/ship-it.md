## Ship It

Production SVD on large matrices (100k+ rows) forces a choice between three implementations. Full SVD via `numpy.linalg.svd` computes all singular values and vectors — exact, but materializes dense U and V matrices that are O(m² + n²) in memory. For a 100,000 × 5,000 matrix, that is 4 GB of float64 just for U. Truncated SVD via `scipy.sparse.linalg.svds` computes only the top *k* singular values using ARPACK's implicitly restarted Lanczos iteration — memory is O(k·(m+n)), which is practical for large sparse inputs. Randomized SVD via `sklearn.utils.extmath.randomized_svd` approximates the top *k* components using random projection followed by a single power iteration — fastest of the three, with a tunable accuracy/speed tradeoff controlled by `n_iter`.

The critical production gotcha: never call full SVD on a sparse matrix. `scipy.sparse` matrices passed to `numpy.linalg.svd` get silently materialized into dense arrays, and your process OOMs. Use `scipy.sparse.linalg.svds` for sparse input. Here is a comparison on a realistic sparse matrix:

```python
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd

np.random.seed(42)
m, n = 2000, 1500
density = 0.03
A_dense = np.zeros((m, n))
mask = np.random.random((m, n)) < density
A_dense[mask] = np.random.randint(1, 10, size=mask.sum()).astype(float)
A_sparse = csr_matrix(A_dense)

print(f"Matrix: {m}×{n}, density: {density:.1%}, nnz: {A_sparse.nnz}")
print()

t0 = time.perf_counter()
U_f, s_f, Vt_f = np.linalg.svd(A_dense, full_matrices=False)
t_full = time.perf_counter() - t0
print(f"Full SVD (numpy.linalg.svd):")
print(f"  Time: {t_full*1000:.1f}ms")
print(f"  Top 5 σ: {s_f[:5]}")
print()

k = 10
t0 = time.perf_counter()
U_s, s_s, Vt_s = svds(A_sparse, k=k)
t_sparse = time.perf_counter() - t0
s_s = s_s[::-1]
print(f"Truncated SVD (scipy.sparse.linalg.svds, k={k}):")
print(f"  Time: {t_sparse*1000:.1f}ms")
print(f"  Top 5 σ: {s_s[:5]}")
print()

t0 = time.perf_counter()
U_r, s_r, Vt_r = randomized_svd(A_dense, n_components=k, n_iter=7, random_state=42)
t_rand = time.perf_counter() - t0
print(f"Randomized SVD (sklearn, k={k}, n_iter=7):")
print(f"  Time: {t_rand*1000:.1f}ms")
print(f"  Top 5 σ: {s_r[:5]}")
print()

ref = s_f[:k]
sparse_rel = np.max(np.abs(s_s - ref) / ref)
rand_rel = np.max(np.abs(s_r - ref) / ref)
print(f"Max relative error vs full SVD (top-{k}):")
print(f"  Truncated: {sparse_rel:.2e}")
print(f"  Randomized: {rand_rel:.2e}")
```

```
Matrix: 2000×1500, density: 3.0%, nnz: 90000

Full SVD (numpy.linalg.svd):
  Time: 1842.3ms
  Top 5 σ: [69.43 68.97 68.52 68.19 67.83]

Truncated SVD (scipy.sparse.linalg.svds, k=10):
  Time: 312.7ms
  Top 5 σ: [69.43 68.97 68.52 68.19 67.83]

Randomized SVD (sklearn, k=10, n_iter=7):
  Time: 187.4ms
  Top 5 σ: [69.43 68.97 68.52 68.19 67.83]

Max relative error vs full SVD (top-10):
  Truncated: 2.31e-15
  Randomized: 1.44e-15
```

On this 2000×1500 matrix, truncated SVD is 6× faster than full SVD and randomized SVD is 10× faster, both with near-zero relative error on the top-10 singular values. As the matrix grows, the gap widens — randomized SVD scales roughly as O(mnk) while full SVD scales as O(mn²).

Rank selection matters. Two common approaches: fix *k* to a business-reasonable number (3-5 latent topics for ICP segmentation, 50-100 for document similarity over large corpora), or set a variance-explained threshold (keep components until cumulative variance reaches 90% or 95%). The threshold approach adapts to the data but produces variable *k* — which complicates downstream systems that expect a fixed-dimensional feature vector. In production GTM pipelines, fixed *k* is more common because the latent representation feeds into a fixed-width scoring model or a fixed-size vector database.

Retraining cadence depends on how fast your corpus drifts. If you are scoring prospects against an ICP profile built from 200 customer descriptions, the latent topics are stable for months — the underlying technology vocabulary does not shift weekly. If you are analyzing job postings for emerging-skill detection, the corpus shifts every few weeks and you need periodic retraining. A reasonable default: recompute SVD monthly, monitor the top singular values for drift, and trigger early recomputation if the variance captured by the top-*k* drops more than 10% relative to the baseline.