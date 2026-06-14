## Ship It

In production GTM pipelines, you rarely compare against a single ICP vector. You have multiple ICP profiles—say, "enterprise SaaS" and "mid-market fintech"—each with its own centroid embedding, and you want to score every company against every profile to route accounts to the right campaign. This batched similarity computation is the core operation in embedding-based lookup tables used for account matching and enrichment waterfalls in Zone II. [CITATION NEEDED — concept: multi-ICP embedding similarity in GTM account routing].

The function below accepts company embeddings `(N, D)` and ICP centroids `(K, D)`, reshapes for broadcasting, and produces an `(N, K)` similarity matrix where `result[i, j]` is the cosine similarity between company `i` and ICP profile `j`. It handles the 1D edge case—when someone passes a single company or single ICP—by reshaping to 2D before computing:

```python
import numpy as np

def batched_cosine_similarity(companies, icp_profiles):
    if companies.ndim == 1:
        companies = companies.reshape(1, -1)
    if icp_profiles.ndim == 1:
        icp_profiles = icp_profiles.reshape(1, -1)

    company_norms = np.linalg.norm(companies, axis=1, keepdims=True)
    icp_norms = np.linalg.norm(icp_profiles, axis=1, keepdims=True)

    dot_products = companies @ icp_profiles.T

    similarity = dot_products / (company_norms @ icp_norms.T)

    return similarity

np.random.seed(42)
test_companies = np.random.randn(4, 64)
test_icp1 = np.random.randn(64)
test_icp2 = np.random.randn(64)
test_icps = np.stack([test_icp1, test_icp2])

sim = batched_cosine_similarity(test_companies, test_icps)
print(f"Companies: {test_companies.shape}")
print(f"ICP profiles: {test_icps.shape}")
print(f"Similarity matrix: {sim.shape}")
print(np.round(sim, 4))

best_icp = np.argmax(sim, axis=1)
print(f"\nBest ICP per company: {best_icp}")
print(f"  Company 0 → ICP {best_icp[0]}")
print(f"  Company 1 → ICP {best_icp[1]}")
print(f"  Company 2 → ICP {best_icp[2]}")
print(f"  Company 3 → ICP {best_icp[3]}")

single = batched_cosine_similarity(test_companies[0], test_icps)
print(f"\nSingle company input → output shape: {single.shape}")
```

Output:

```
Companies: (4, 64)
ICP profiles: (2, 64)
Similarity matrix: (4, 2)
[[ 0.0428 -0.0703]
 [ 0.0133  0.0101]
 [-0.0544 -0.0035]
 [-0.044   0.0785]]

Best ICP per company: [0 0 1 1]
  Company 0 → ICP 0
  Company 1 → ICP 0
  Company 2 → ICP 1
  Company 3 → ICP 1

Single company input → output shape: (1, 2)
```

The shape trace: `companies @ icp_profiles.T` is `(4, 64) @ (64, 2)` → `(4, 2)`. The norm normalization uses `company_norms @ icp_norms.T` which is `(4, 1) @ (1, 2)` → `(4, 2)` via matmul broadcasting—the outer product of the two norm vectors. This is dense in one line but every shape is dictated by the rules you've already learned.