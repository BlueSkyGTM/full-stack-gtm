# Vectors, Matrices & Operations

---

## Hook It

Every embedding model, similarity scorer, and classifier you'll build in GTM runs on linear algebra. You can't debug why two companies scored 0.92 cosine similarity if you can't read the vector math underneath. This is the literacy layer.

---

## Learn It

Core mechanisms only — no theory for theory's sake:

- **Vectors**: ordered lists of numbers representing magnitude + direction in N-dimensional space. In GTM, a vector is a company or contact compressed into numeric form by an embedding model.
- **Matrices**: rectangular arrays of numbers. A matrix lets you transform many vectors at once (batch operations) or represent pairwise relationships (distance matrices).
- **Element-wise operations**: add, subtract, multiply corresponding elements. Used for combining feature vectors.
- **Dot product**: single number measuring how much two vectors point in the same direction. Core of cosine similarity.
- **Matrix multiplication (matmul)**: transforms vectors through weight matrices — the actual computation inside every neural network layer.
- **Norms**: L1 (Manhattan), L2 (Euclidean). Used to normalize vectors before similarity comparison.
- **Broadcasting**: how operations expand across dimensions without explicit loops.

---

## See It

Working code examples with observable output:

- Create vectors and compute dot product manually vs. with numpy — print both to show equivalence
- Build a 3×3 matrix and multiply it by a vector — print input and output to show transformation
- Compute L2 norm of a vector, normalize it, print before/after magnitudes
- Build a pairwise distance matrix from 5 company embedding vectors — print the matrix
- Demonstrate broadcasting: add a scalar vector to every row of a matrix — print shapes and results

---

## Do It

Three exercise hooks:

- **Easy**: Given two vectors, compute cosine similarity manually (dot product ÷ product of norms). Print result and verify against `numpy` or `scipy`.
- **Medium**: Build a 10×4 matrix of "company embeddings" (random), compute the full pairwise cosine similarity matrix, print the top-3 most similar pairs with scores.
- **Hard**: Implement a single-layer neural network forward pass: matrix multiply input (batch of 8 vectors, dim 4) by a weight matrix (4×3), add bias, apply ReLU. Print input shape, weight shape, output shape, and first output row.

---

## Use It

**GTM Cluster: ICP Matching & Lead Scoring (Zone 1 → Zone 2)**

This is the math behind embedding-based similarity scoring. When an enrichment pipeline encodes company descriptions into vectors and compares them against your ICP vector, the comparison is a dot product divided by norms (cosine similarity). When a Clay waterfall scores and ranks accounts, the ranking operation is a matrix of account vectors compared to a query vector — this is matmul.

- Map: company description → embedding vector → cosine similarity to ICP vector → ranked lead list
- The distance matrix exercise above is structurally identical to deduplicating accounts by description similarity

---

## Ship It

**Deployment pattern**: batch vector operations in a lead enrichment pipeline.

- Store company embedding vectors in a numpy array or parquet file (not a database — yet)
- Nightly job: load new company embeddings, concatenate to existing matrix, recompute pairwise similarity or similarity-to-ICP, output top-N new matches to CSV
- This is the foundation for the embedding storage and retrieval lesson (which introduces FAISS/chroma and vector databases)

**GTM redirect**: This batch job is the mechanism behind "find me companies similar to my best customers" — the core loop of Zone 1 account identification.

---

## Learning Objectives

1. **Compute** dot products, norms, and cosine similarity between vectors using numpy
2. **Explain** why cosine similarity, not raw dot product, is used for embedding comparison
3. **Implement** pairwise distance/similarity matrix computation across a set of vectors
4. **Execute** matrix multiplication and describe what it represents in a neural network forward pass
5. **Map** vector similarity operations to the ICP matching mechanism in GTM enrichment pipelines