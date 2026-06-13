# Singular Value Decomposition

## Beat 1: Concept

SVD factorizes any matrix **A** (m×n) into three matrices: **U** (m×m), **Σ** (m×n diagonal of singular values), and **V^T** (n×n). Every real matrix has an SVD — no exceptions, no square-matrix requirement. This decomposition reveals the orthogonal basis for both the row space and column space simultaneously.

## Beat 2: Mechanism

The singular values in **Σ** are ordered largest to smallest and encode how much variance each component captures. Truncating to the top *k* singular values yields the best rank-*k* approximation to **A** in the Frobenius norm sense — this is the Eckart-Young theorem. The left singular vectors (**U**) span the column space; the right singular vectors (**V**) span the row space. Computing SVD via eigenvalue decomposition of **A^T A** or **AA^T** works but is numerically unstable; modern implementations use bidiagonalization followed by QR iteration.

## Beat 3: Implementation

Build a working example that constructs a term-document matrix from a small corpus of GTM-relevant text (job postings or company descriptions), computes the full SVD using `numpy.linalg.svd`, prints the singular values to show variance concentration, and reconstructs the original matrix to verify correctness. Then truncate to rank-2 and measure reconstruction error with Frobenius norm.

**Exercise hooks:**
- **Easy:** Print singular values and confirm they descend. Identify how many components capture 90% of variance.
- **Medium:** Truncate at rank *k*, reconstruct, and print the Frobenius norm of the residual.
- **Hard:** Build a term-document matrix from 10 real company "about" pages, compute SVD, and extract the top 3 latent topics by inspecting the highest-magnitude entries in the first 3 columns of **V**.

## Beat 4: Use It

**GTM Redirect:** Zone 1 — ICP & Targeting / Zone 3 — Scoring & Prioritization. Latent Semantic Analysis (LSA) uses SVD on term-document matrices to discover latent topics in firmographic text data. This is the mechanism behind topic clustering for intent signal detection: when prospects publish job posts or blog content, SVD extracts the latent technology themes without keyword matching. Build a similarity scorer that projects new company descriptions into the latent space (via **V_k**) and computes cosine similarity against a reference ICP profile.

[CITATION NEEDED — concept: LSA applied to firmographic enrichment in GTM workflows]

## Beat 5: Ship It

Production SVD on large matrices (100k+ rows) requires choosing between full SVD (`numpy.linalg.svd`), truncated SVD (`scipy.sparse.linalg.svds` for sparse matrices), or randomized SVD (`sklearn.utils.extmath.randomized_svd`). The tradeoff is accuracy vs. compute time. Sparse input requires `scipy.sparse.linalg.svds` — full SVD on a sparse matrix materializes a dense matrix and will OOM. Document the rank *k* selection method (variance explained threshold vs. fixed *k*) and the retraining cadence when new documents arrive.

**Exercise hooks:**
- **Easy:** Benchmark `numpy.linalg.svd` vs. `randomized_svd` on a 5000×200 dense matrix — print wall clock time for both.
- **Medium:** Convert a sparse term-document matrix to CSR format, compute truncated SVD with `svds`, and print the top 5 singular values alongside the full-SVD singular values for comparison.
- **Hard:** Wrap SVD projection in a function that takes new documents, projects into latent space using a cached **V_k**, and returns cosine similarity scores against a target vector. Test with held-out documents.

## Beat 6: Evaluate

**Exercise hooks:**
- **Easy:** Given a 4×3 matrix with singular values [5.0, 2.0, 0.1], what rank-*k* approximation captures >90% of the Frobenius norm? Show the arithmetic.
- **Medium:** Implement the Eckart-Young proof-by-example: compute rank-1 through rank-full approximations of a 6×4 matrix, print the Frobenius norm of each residual, and confirm rank-1 is the best rank-1 approximation.
- **Hard:** Given a term-document matrix with 8 terms and 5 documents, compute SVD, extract rank-2 latent topics, assign human-readable labels to each topic by examining the top 3 terms per component, and defend the choice of rank 2 over rank 3 with variance-explained numbers.

---

**Learning Objectives:**
1. Decompose any real matrix into **U**, **Σ**, and **V^T** and verify correctness via reconstruction.
2. Truncate SVD to rank *k* and quantify the approximation error using Frobenius norm.
3. Implement latent semantic analysis on a term-document matrix to extract latent topics from text.
4. Compare full SVD, truncated SVD, and randomized SVD on sparse vs. dense inputs with timing benchmarks.
5. Project new documents into a pre-computed latent space and compute similarity scores against a target profile.