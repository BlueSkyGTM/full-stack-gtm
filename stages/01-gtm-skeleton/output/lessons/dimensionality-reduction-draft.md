# Dimensionality Reduction

## Hook

You enriched 2,000 accounts with 47 firmographic signals. Your clustering algorithm produced garbage. The problem isn't the algorithm — it's that most of those 47 signals are correlated noise encoding the same underlying dimension three times. Dimensionality reduction collapses redundant features into their actual information content.

---

## Concept

**Mechanism first.** PCA finds orthogonal axes of maximum variance by eigendecomposing the covariance matrix. Each principal component is a linear combination of original features, ranked by how much variance it explains. You keep enough components to retain ~95% variance and discard the rest — which is usually the noise floor.

**SVD** generalizes this: any matrix *A* factors into *U Σ Vᵀ*. Truncated SVD keeps only the top *k* singular values. PCA is SVD on centered data. Both give you a compressed representation with minimal reconstruction error.

**Non-linear methods (t-SNE, UMAP)** operate differently — they preserve local neighborhood structure for visualization, not for downstream modeling. t-SNE destroys global geometry; UMAP preserves some of it. Neither produces a reusable transform you can apply to new data without refitting.

**What to use when:**
- PCA / Truncated SVD → preprocessing before clustering or classification on high-dimension enrichment data
- UMAP → visualizing account segments in 2D to explain them to stakeholders
- Feature selection (not extraction) → when interpretability matters more than compression

---

## Use It

**GTM Redirect: ICP Scoring Pipeline (Zone 01) & Account Segmentation (Zone 02)**

Your enrichment waterfall produces a wide feature matrix: company size, tech stack indicators (binary), funding rounds, growth signals, intent keywords. Many of these are collinear — employee count and revenue correlate; multiple intent keywords encode the same buying stage.

PCA on this matrix collapses correlated features into principal components that represent actual independent buying signals. You then cluster or score on the reduced space instead of the raw feature matrix.

This is the same pattern used in Clay when you aggregate multiple enrichment columns into a composite score — the manual version of what PCA does automatically. [CITATION NEEDED — concept: Clay formula column used for composite ICP scoring from multiple enrichment signals]

**Exercise hooks:**
- *Easy:* Run PCA on a synthetic firmographic matrix (employee_count, revenue, funding, num_technologies) and print explained variance ratio. Confirm that 2 components capture 90%+ of variance because the features are correlated.
- *Medium:* Load a dataset of 200 accounts with 30 binary tech-stack indicators. Apply Truncated SVD (no centering needed for binary). Print reconstruction error at k=5, 10, 15.
- *Hard:* Build a two-stage pipeline: (1) TruncatedSVD on enrichment features, (2) KMeans on reduced space. Compare cluster silhouette scores with and without dimensionality reduction.

---

## Ship It

**Production considerations:**

PCA produces a dense transform matrix. You fit it on training data, then apply the same transform to new accounts at inference time. Store the fitted `PCA` object (pickle or joblib). Do not refit on every batch — the component definitions drift, and your clusters become uninterpretable.

Reconstruction error is your health check. If a new batch of accounts has high reconstruction error using the stored PCA, your new data has drifted from the distribution the components were trained on. Flag it.

Scaling is non-negotiable. PCA seeks maximum variance — if revenue is in millions and headcount is in tens, revenue dominates every component. Standardize before PCA. This is `StandardScaler → PCA`, always.

**When not to use it:**
- If you have <15 features, PCA adds complexity without benefit.
- If stakeholders need to see "this account scored high because of revenue and tech stack X," PCA components are uninterpretable. Use feature selection instead.
- If features are categorical with many levels, PCA on one-hot encoded matrices produces components that don't map to anything meaningful. Use correspondence analysis or embeddings.

**Exercise hooks:**
- *Easy:* Fit PCA on 500 synthetic accounts. Pickle the model. Load it in a new process and transform 50 new accounts. Print the first 3 component values for account 0.
- *Medium:* Build a reconstruction error monitor. For each new account, compute MSE between original features and PCA-reconstructed features. Flag any account above the 95th percentile of training reconstruction error.
- *Hard:* Implement a drift detector: store the mean explained variance per component at fit time. On new batches, refit PCA and compare component loadings using cosine similarity. Print a warning if any component's loading vector changes by more than 0.3.

---

## Debug It

**Problem: First component captures 99% of variance.**
Diagnosis: One feature has vastly larger scale than the others. You forgot `StandardScaler`.
Fix: Standardize before PCA.

**Problem: Components don't map to anything interpretable.**
Diagnosis: This is expected with PCA — components are linear combinations of all features. If you need interpretability, use feature selection or inspect the loading weights to assign semantic meaning post-hoc.

**Problem: t-SNE shows "clusters" that don't exist in the original space.**
Diagnosis: t-SNE creates visual clusters from noise at high perplexity values. Perplexity parameter controls effective neighborhood size. Default (30) is too high for small datasets.
Fix: Set perplexity to `sqrt(n_samples)`. Verify clusters exist in original space using silhouette score before trusting the visualization.

**Problem: PCA on sparse binary matrix crashes or takes forever.**
Diagnosis: Dense PCA computes full covariance matrix — O(n × d²) memory for d features. On a wide sparse matrix, this explodes.
Fix: Use `TruncatedSVD` instead. It operates on the sparse matrix directly without centering.

**Exercise hooks:**
- *Easy:* Reproduce the "first component captures 99% variance" bug on synthetic data with unscaled features. Fix it with `StandardScaler`. Print variance ratios before and after.
- *Medium:* Run t-SNE on uniformly random data with default perplexity. Observe visual clusters. Reduce perplexity. Observe uniform scatter. Print both plots.
- *Hard:* Benchmark `PCA` vs `TruncatedSVD` on a 10,000 × 1,000 sparse binary matrix (1% density). Print runtime and peak memory for both. Explain why PCA fails.

---

## Own It

**Learning Objectives:**

1. Implement PCA on a multi-feature account dataset and determine the number of components needed to retain 95% variance.
2. Compare PCA, Truncated SVD, and UMAP on the same dataset — print explained variance (PCA/SVD) and trustworthiness scores (UMAP) to justify choice of method.
3. Construct a production pipeline that fits PCA on training accounts, persists the fitted model, applies the transform to new accounts, and flags drift via reconstruction error.
4. Diagnose and fix the three most common PCA failure modes: unscaled features, sparse matrices, and over-interpreted components.
5. Evaluate whether dimensionality reduction is warranted for a given dataset by comparing downstream clustering quality (silhouette score) with and without reduction.

**Assessment hooks:**
- Given a 500-account dataset with 20 correlated enrichment features, determine the optimal number of PCA components and justify the choice with variance explained.
- A stakeholder asks why "Component 1" has a score of 2.3 for their account. Write a 3-sentence explanation of what that number means in terms of the original features.
- Your batch inference pipeline's reconstruction error has tripled over the past week. Propose a diagnostic procedure and identify the most likely cause.