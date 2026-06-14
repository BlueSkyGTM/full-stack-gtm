## Ship It

The production pattern is a two-stage pipeline: Truncated SVD for dimensionality reduction, then KMeans for clustering. This is how you turn a 47-column enrichment export into actionable account segments. The reduced space gives KMeans orthogonal axes to work with, which produces tighter, more separated clusters.

```python
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n_accounts = 500
n_features = 47

latent_dimensions = np.random.normal(0, 1, (n_accounts, 5))

feature_groups = {
    "scale": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "tech_adoption": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    "growth": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    "intent": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    "engagement": [40, 41, 42, 43, 44, 45, 46],
}

enrichment = np.zeros((n_accounts, n_features))
for group_name, indices in feature_groups.items():
    latent_idx = list(feature_groups.keys()).index(group_name)
    for idx in indices:
        enrichment[:, idx] = (
            latent_dimensions[:, latent_idx] * np.random.uniform(0.8, 1.5)
            + np.random.normal(0, 0.2, n_accounts)
        )

scaler = StandardScaler()
X_scaled = scaler.fit_transform(enrichment)

print("=== Stage 1: Clustering on raw 47D enrichment data ===")
kmeans_raw = KMeans(n_clusters=5, n_init=10, random_state=42)
labels_raw = kmeans_raw.fit_predict(X_scaled)
score_raw = silhouette_score(X_scaled, labels_raw)
print(f"Silhouette score: {score_raw:.4f}")
print(f"(Closer to 1 = better separated clusters)")

print("\n=== Stage 2: SVD → KMeans pipeline ===")
results = []
for k in [3, 5, 8, 10, 15]:
    svd = TruncatedSVD(n_components=k, random_state=42)
    X_reduced = svd.fit_transform(X_scaled)
    
    kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_reduced)
    score = silhouette_score(X_reduced, labels)
    variance = svd.explained_variance_ratio_.sum()
    results.append((k, score, variance))
    print(f"  k={k:2d} components | silhouette: {score:.4f} | variance retained: {variance:.2%}")

best_k, best_score, best_var = max(results, key=lambda r: r[1])
print(f"\nBest: k={best_k} with silhouette {best_score:.4f}")
print(f"Improvement over raw: {(best_score - score_raw) / score_raw * 100:+.1f}%")

print("\n=== Stage 3: Inspect cluster composition ===")
svd_best = TruncatedSVD(n_components=best_k, random_state=42)
X_best = svd_best.fit_transform(X_scaled)
kmeans_best = KMeans(n_clusters=5, n_init=10, random_state=42)
labels_best = kmeans_best.fit_predict(X_best)

for cluster_id in range(5):
    mask = labels_best == cluster_id
    size = mask.sum()
    centroid = X_best[mask].mean(axis=0)
    top_dims = np.argsort(np.abs(centroid))[::-1][:3]
    print(f"  Cluster {cluster_id}: {size:3d} accounts | top components: {top_dims.tolist()}")
```

The pipeline should show improved silhouette scores at low k values (5–8 components) compared to raw 47-dimensional clustering. The improvement comes from removing collinear noise that was inflating distance calculations without adding discriminative information.

Now produce a 2D visualization for stakeholders. This is where UMAP enters — not as a preprocessing step, but as a communication tool:

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

np.random.seed(42)

tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
X_2d = tsne.fit_transform(X_best)

fig, ax = plt.subplots(figsize=(10, 7))
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
for cluster_id in range(5):
    mask = labels_best == cluster_id
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colors[cluster_id], 
               label=f'Segment {cluster_id} ({mask.sum()} accounts)', alpha=0.6, s=30)
ax.set_title('Account Segments (t-SNE projection of SVD-reduced enrichment data)')
ax.legend()
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig('account_segments.png', dpi=150)
print("Saved visualization to account_segments.png")
print("Use this in your slide deck — each color is a cluster from the pipeline above.")
```

This is the artifact you ship: a Python script that takes a raw enrichment export, reduces it to independent components, clusters accounts into segments, and produces a 2D plot for the weekly GTM review. The same script runs in the Python environment where you process Clay webhooks and Apollo API exports — your Zone 01 workspace for TAM mapping and signal scoring.