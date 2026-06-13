# Tensor Operations

## Hook
You've got a batch of company embeddings from an API—shape (128, 768)—and you need to compute cosine similarity against an ICP profile vector. That's a tensor operation. If you don't know how shapes compose, you'll spend an hour fighting dimension mismatches instead of shipping.

## Concept
Tensors are multidimensional arrays with explicit shape semantics. A scalar is 0D, a vector is 1D, a matrix is 2D, and anything beyond is just "more dimensions." The operations that matter: reshaping (same data, different shape), broadcasting (implicit expansion of mismatched shapes), reduction (collapsing dimensions via sum/mean/max), and matmul (the backbone of every neural network). NumPy implements these as `ndarray` methods and standalone functions. PyTorch mirrors the same API with `torch.Tensor`.

## Mechanism
Three mechanisms to cover: (1) **Shape rules**—how dimensions are ordered (batch, sequence, features), how reshape reads left-to-right, and how `-1` infers the remaining dimension. (2) **Broadcasting**—the algorithm that compares shapes right-to-left, expands dimensions of size 1, and rejects incompatible shapes. (3) **Reduction along axes**—how `axis=0` collapses rows, `axis=1` collapses columns, and `keepdims=True` preserves the dimension for downstream operations. Each mechanism gets a runnable code block with printed output confirming the shape and values.

## Use It
**GTM Redirect:** Tensor operations are foundational for Zone II (Signal) — specifically, computing similarity scores between company embeddings and ICP vectors in lead scoring pipelines. [CITATION NEEDED — concept: embedding-based similarity scoring in GTM workflows]

*Exercise hook (Easy):* Given a tensor of shape (5, 3) representing 5 companies × 3 features, compute the L2 norm of each row and print the shape of the result.

*Exercise hook (Medium):* Given a batch of 10 company embeddings of dimension 128 and a target ICP embedding of dimension 128, compute cosine similarity for all 10 companies in a single vectorized operation. Print the index of the highest-scoring company.

## Ship It
**GTM Redirect:** Same Zone II connection — this batched similarity computation is the core operation in embedding-based lookup tables used for account matching and enrichment waterfalls.

*Exercise hook (Hard):* Build a function that accepts a 2D tensor of company embeddings (N × D), a 2D tensor of ICP centroid embeddings (K × D), and returns an (N × K) similarity matrix. Handle the edge case where inputs are 1D (single company or single ICP) by reshaping to 2D before computing. Print the full similarity matrix for a test case with 4 companies and 2 ICP profiles.

## Evaluate
*Exercise hook:* Three debug scenarios where tensor operations fail with shape mismatches. Student must identify the broadcasting error, the reshape error, and the matmul dimension conflict—then fix each one with a single-line correction.

---

**Learning Objectives (5):**
1. Reshape tensors using explicit dimensions and `-1` inference, and predict the output shape before running code.
2. Apply NumPy broadcasting rules to determine whether two shapes are compatible and predict the result shape.
3. Implement batched matrix multiplication to compute similarity scores across multiple vector pairs simultaneously.
4. Reduce tensors along specified axes using `sum`, `mean`, and `max`, with and without `keepdims`.
5. Diagnose and fix shape mismatch errors in multi-step tensor pipelines.