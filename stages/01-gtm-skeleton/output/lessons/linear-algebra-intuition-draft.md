# Linear Algebra Intuition

## Learning Objectives

1. Implement dot product and cosine similarity operations on raw vectors, printing intermediate values to confirm correctness.
2. Explain matrix-vector multiplication as a geometric transformation by tracing a single vector through a rotation matrix.
3. Compare Euclidean distance versus cosine similarity on a concrete pair of embeddings, articulating when each metric fails.
4. Diagnose rank deficiency in a matrix by computing its singular values and identifying near-zero entries.
5. Wire a cosine similarity function into a lead-scoring comparison between a target ICP vector and a set of account vectors.

---

## Beat 1: Hook

You've run embeddings through APIs and fed the results into clustering. The outputs work, but the inputs feel like incantations. Linear algebra is the layer beneath that — once you can *see* what a dot product actually computes, similarity search stops being magic and starts being engineering. This is the prerequisite for everything in embedding-based GTM pipelines.

---

## Beat 2: Concept

### Vectors as Arrows, Vectors as Records

A vector is an ordered list of numbers. Two interpretations: geometric (an arrow in space) and tabular (a row of features). Both are correct; the useful one depends on context. An embedding of a company description is tabular data treated as a point in high-dimensional space.

### Dot Product as Projection

The dot product of two vectors measures how much one points in the direction of the other. Formula: sum of element-wise products. Geometric reading: `a · b = |a||b|cos(θ)`. When either vector has unit length, the dot product *is* the cosine of the angle between them.

### Cosine Similarity

Normalize both vectors to length 1, then dot product. Result is in [-1, 1]. 1 = identical direction. 0 = orthogonal. -1 = opposite. This is the standard similarity metric in embedding-based retrieval — not because it's "best," but because it discards magnitude and isolates directional alignment.

### Matrix-Vector Multiplication as Transformation

A matrix multiplying a vector produces a new vector. Geometrically: the matrix stretches, rotates, or shears the space, and the output is where the input vector lands. This is the mechanism behind every linear layer in a neural network.

### Rank and Singularity

A matrix's rank is the number of independent directions it preserves. If rank < number of columns, some input directions get flattened to zero. Singular value decomposition (SVD) reveals this: near-zero singular values mark the flat directions. This matters when you're inverting matrices or reducing dimensionality.

---

## Beat 3: Demonstration

Code blocks with observable output:

1. **Dot product by hand vs. library.** Two 3-dimensional vectors, compute dot product three ways (loop, list comprehension, numpy). Print all three to show they match.

2. **Cosine similarity implementation.** Build it from raw numpy operations (normalize, dot). Test on three pairs: nearly identical, orthogonal, opposite. Print similarity scores.

3. **Matrix-vector transform.** Define a 2D rotation matrix, apply it to a unit vector at 0°, print the result, confirm it matches the expected rotated coordinates.

4. **SVD rank check.** Construct a matrix with a linearly dependent column, run SVD, print singular values, show one is near-zero.

Exercise hooks:
- **Easy:** Modify the angle in the rotation matrix and predict the output coordinates before running.
- **Medium:** Build a cosine similarity function that handles zero-vectors without crashing (return 0 similarity).
- **Hard:** Implement a function that takes a matrix and returns its effective rank by counting singular values above a threshold (default: 1e-10).

---

## Beat 4: Use It

### GTM Redirect: Zone 3 — Enrichment / ICP Matching via Embedding Similarity

[CITATION NEEDED — concept: Clay waterfall enrichment step using embedding similarity]

The mechanism: encode your ICP as a vector (either from structured features or a text embedding of the ICP description), encode each account the same way, compute cosine similarity, rank. This is how embedding-based lead scoring works — not keyword match, but directional alignment in vector space.

Concrete application: given a target ICP embedding and a list of 10 account embeddings, compute pairwise cosine similarity, rank descending, print the top 3 matches with scores.

Exercise hooks:
- **Easy:** Hardcode two company description embeddings (use dummy vectors) and compute similarity.
- **Medium:** Fetch real embeddings via an API call for two descriptions, then compare. Print the raw vectors and the similarity score.
- **Hard:** Given a matrix of 20 account embeddings (rows) and a single ICP vector, compute all similarities in one matrix multiply, rank, and return top 5 with scores.

---

## Beat 5: Ship It

Production-grade considerations:

1. **Normalization caching.** If you're comparing one query vector against N stored vectors, pre-normalize the stored set once. Store normalized embeddings. Cosine similarity becomes a single dot product per comparison.

2. **Batch similarity via matrix multiplication.** Stack account embeddings as rows in matrix A (shape: N × d). Put ICP vector as column in matrix B (shape: d × 1). Result of A @ B: N similarity scores in one operation. This is the mechanism behind approximate nearest neighbor search at scale.

3. **Numerical stability.** Division by near-zero norms produces NaN. Guard with an epsilon clamp. This is the kind of thing that doesn't show up in demos and crashes pipelines at 2am.

Exercise hooks:
- **Easy:** Write a cosine similarity function with epsilon-guarded normalization. Test on a zero-vector input.
- **Medium:** Implement batch similarity using matrix multiply on a 50 × 8 matrix of fake account embeddings. Print top 5 indices and scores.
- **Hard:** Benchmark: compare loop-based pairwise cosine similarity vs. single matrix multiply on 1000 vectors. Print timing for both approaches.

---

## Beat 6: Evaluate

### Conceptual Checks

1. Given two vectors with cosine similarity of 0.95, explain what happens to the similarity if you double the magnitude of one vector. Why?
2. A 3×3 matrix has singular values [4.2, 0.001, 3.1]. What is the effective rank? What happens to inputs along the direction corresponding to 0.001?
3. Why does cosine similarity outperform Euclidean distance for comparing text embeddings of different lengths? Give a concrete example.

### Mechanism Tracing

4. Trace the dot product of `[3, 4, 0]` and `[0, 0, 5]` by hand. What does the result tell you about the geometric relationship?
5. A matrix has two identical columns. Predict what SVD will show before computing. Then compute and confirm.

### GTM Application

6. You have 10,000 account embeddings and need to rank them against an ICP vector every time a new campaign launches. Describe the exact sequence of operations (which are precomputed, which are per-campaign). Justify the ordering.

---

## GTM Redirect Rules for This Lesson

- **Primary cluster:** Zone 3 — Enrichment (embedding-based ICP matching and lead scoring)
- **Secondary cluster:** Zone 4 — Qualification (similarity-based prioritization)
- **Specific mechanism:** Cosine similarity ranking of account embeddings against an ICP profile vector
- **If a student asks "when will I use this":** Every embedding operation in GTM tooling (Clay waterfall enrichment, semantic search in Apollo, vector-based deduplication) runs on these primitives. You don't need to derive proofs, but you need to read a similarity score and know what it means.