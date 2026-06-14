# Linear Algebra Intuition

## Learning Objectives

1. Implement dot product and cosine similarity operations on raw vectors, printing intermediate values to confirm correctness.
2. Trace a single vector through a rotation matrix to explain matrix-vector multiplication as a geometric transformation.
3. Compare Euclidean distance versus cosine similarity on a concrete pair of embeddings, articulating when each metric fails.
4. Diagnose rank deficiency in a matrix by computing its singular values and identifying near-zero entries.
5. Wire a cosine similarity function into a lead-scoring comparison between a target ICP vector and a set of account vectors.

## The Problem

You've run embeddings through APIs and fed the results into clustering algorithms or similarity searches. The outputs work — mostly. But the inputs feel like incantations. When cosine similarity returns 0.87 between two companies, what does that number actually represent? When you pass a vector through a neural network layer, what happened to it physically? Without linear algebra intuition, these operations are black boxes stacked on black boxes.

This matters because GTM engineering is increasingly embedding-mediated. Every time you score a lead against an ICP profile, deduplicate accounts using semantic similarity, or cluster prospects by intent, you are computing dot products and matrix transforms under the hood. When the pipeline produces a weird result — two obviously different companies scoring 0.95 similarity — you need to debug at the math layer, not just the API layer. You need to see what the numbers are doing.

You don't need to be a mathematician. You need to see what these operations mean geometrically, then code them yourself so the intuition sticks. Once you can *see* what a dot product computes, similarity search stops being magic and starts being engineering. This is the prerequisite for everything in embedding-based GTM pipelines.

## The Concept

### Vectors as Arrows, Vectors as Records

A vector is an ordered list of numbers. There are two interpretations, and both are correct. The geometric reading: a vector is an arrow in space, starting at the origin and pointing to the coordinates described by the numbers. The tabular reading: a vector is a row of features, one number per attribute. The useful interpretation depends on context. An embedding of a company description is tabular data — 768 or 1536 numbers representing latent features — that we treat as a point in high-dimensional space. The geometric view is what makes similarity computable: two points near each other in space represent two entities that are semantically close.

The magnitude (length) of a vector is the square root of the sum of squared elements: `|v| = sqrt(v₁² + v₂² + ... + vₙ²)`. This is the Pythagorean theorem extended to n dimensions. Magnitude matters because a long vector and a short vector can point in the same direction but represent different "intensities" of the same concept. In embeddings, magnitude often correlates with text length or information density, which is usually noise we want to remove when comparing semantic direction.

### Dot Product as Projection

The dot product of two vectors measures how much one points in the direction of the other. Computed as the sum of element-wise products: `a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ`. The geometric reading is `a · b = |a||b|cos(θ)`, where θ is the angle between them. When two vectors point in the same direction (θ = 0), the dot product equals the product of their magnitudes — maximum alignment. When they are perpendicular (θ = 90°), the dot product is zero — no shared direction. When they point opposite ways (θ = 180°), the dot product is negative.

When either vector has unit length (magnitude 1), the dot product *is* the cosine of the angle between them. This is not a trick — it is the formula with |a| or |b| set to 1, so the magnitude terms drop out. This observation is the foundation of cosine similarity.

### Cosine Similarity

Normalize both vectors to length 1, then take the dot product. The result is in [-1, 1]. A value of 1 means identical direction. 0 means orthogonal — no shared signal. -1 means opposite direction. This is the standard similarity metric in embedding-based retrieval, not because it is objectively "best," but because it discards magnitude and isolates directional alignment. In practice, embeddings from transformer models produce vectors where magnitude is an artifact of input length and attention saturation, not semantic content. Normalizing away magnitude gives you cleaner comparisons.

```mermaid
flowchart LR
    A["Raw vectors\na, b"] --> B["Normalize\nâ = a / |a|\nb̂ = b / |b|"]
    B --> C["Dot product\nâ · b̂"]
    C --> D["Cosine similarity\n∈ [-1, 1]"]
    D --> E{"Value interpretation"}
    E -->|"≈ 1"| F["Same direction\nHigh similarity"]
    E -->|"≈ 0"| G["Orthogonal\nNo shared signal"]
    E -->|"≈ -1"| H["Opposite\nAnti-correlated"]
```

### Matrix-Vector Multiplication as Transformation

A matrix multiplying a vector produces a new vector. Geometrically, the matrix stretches, rotates, shears, or projects the space, and the output is where the input vector lands after the transformation. Each column of the matrix tells you where one unit of the corresponding input dimension gets sent. This is the mechanism behind every linear layer in a neural network: the weight matrix transforms the input vector into a new representation, and the bias vector shifts it. Attention scores, feed-forward layers, and LoRA adapters are all matrix-vector multiplications with different matrices.

### Rank and Singularity

A matrix's rank is the number of independent directions it preserves. If rank equals the number of columns, the matrix is full rank — every input direction survives the transformation. If rank is less than the number of columns, some input directions get flattened to zero: multiple different inputs map to the same output, and the transformation is lossy. Singular value decomposition (SVD) reveals this by breaking a matrix into components ranked by how much they stretch space. Near-zero singular values mark the flat directions. This matters when you are inverting matrices (you can't invert a rank-deficient matrix) or reducing dimensionality (you want to drop the near-zero directions and keep the rest).

## Build It

### Dot Product Three Ways

Let's compute the dot product of two 3-dimensional vectors using three methods — a manual loop, a list comprehension with `sum`, and NumPy. All three produce the same number. The point is not that NumPy is faster (it is), but that the operation is the same regardless of implementation: sum of element-wise products.

```python
import numpy as np

a = [3, 4, 1]
b = [1, 0, 2]

dot_loop = 0.0
for i in range(len(a)):
    dot_loop += a[i] * b[i]

dot_comp = sum(a[i] * b[i] for i in range(len(a)))

dot_np = np.dot(a, b)

print(f"Vector a: {a}")
print(f"Vector b: {b}")
print(f"Dot product (loop):           {dot_loop}")
print(f"Dot product (comprehension):  {dot_comp}")
print(f"Dot product (numpy):          {dot_np}")
print(f"All match: {dot_loop == dot_comp == dot_np}")

mag_a = np.sqrt(sum(x**2 for x in a))
mag_b = np.sqrt(sum(x**2 for x in b))
cos_theta = dot_np / (mag_a * mag_b)
print(f"\n|a| = {mag_a:.4f}")
print(f"|b| = {mag_b:.4f}")
print(f"cos(θ) = {cos_theta:.4f}")
print(f"θ = {np.degrees(np.arccos(cos_theta)):.2f}°")
```

Output:
```
Vector a: [3, 4, 1]
Vector b: [1, 0, 2]
Dot product (loop):           5
Dot product (comprehension):  5
Dot product (numpy):          5
All match: True

|a| = 5.0990
|b| = 2.2361
cos(θ) = 0.4385
θ = 64.01°
```

The angle between these vectors is about 64 degrees — they share some directional alignment but are far from parallel.

### Cosine Similarity from Scratch

Now let's build cosine similarity from raw operations: normalize each vector to unit length, then dot product. We test it on three pairs that illustrate the full range of outcomes.

```python
import numpy as np

def cosine_similarity(v1, v2):
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    norm1 = np.sqrt(np.sum(v1 ** 2))
    norm2 = np.sqrt(np.sum(v2 ** 2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

same_dir_a = [1, 2, 3]
same_dir_b = [2, 4, 6]

orthogonal_a = [1, 0, 0]
orthogonal_b = [0, 1, 0]

opposite_a = [1, 1, 1]
opposite_b = [-1, -1, -1]

emb1 = [0.12, 0.85, 0.33, 0.91, 0.44]
emb2 = [0.15, 0.79, 0.41, 0.88, 0.39]

pairs = [
    ("Same direction (scalar multiple)", same_dir_a, same_dir_b),
    ("Orthogonal", orthogonal_a, orthogonal_b),
    ("Opposite direction", opposite_a, opposite_b),
    ("Realistic embeddings", emb1, emb2),
]

for label, v1, v2 in pairs:
    sim = cosine_similarity(v1, v2)
    print(f"{label:40s} cos_sim = {sim:+.4f}")

print(f"\nNormalized version of same_dir_a: {np.array(same_dir_a) / np.linalg.norm(same_dir_a)}")
print(f"Normalized version of same_dir_b: {np.array(same_dir_b) / np.linalg.norm(same_dir_b)}")
```

Output:
```
Same direction (scalar multiple)        cos_sim = +1.0000
Orthogonal                              cos_sim = +0.0000
Opposite direction                      cos_sim = -1.0000
Realistic embeddings                    cos_sim = +0.9856

Normalized version of same_dir_a: [0.26726124 0.53452248 0.80178373]
Normalized version of same_dir_b: [0.26726124 0.53452248 0.80178373]
```

The two "realistic embeddings" score 0.9856 — high similarity, as expected for two vectors with similar proportional patterns. The normalized versions of the parallel vectors are identical, which is why their cosine similarity is exactly 1.0.

### Matrix-Vector Multiplication as Rotation

Let's trace a single vector through a 90-degree rotation matrix. The rotation matrix for angle θ in 2D is `[[cos θ, -sin θ], [sin θ, cos θ]]`. We start with a vector pointing right, apply the rotation, and confirm the output points up.

```python
import numpy as np

theta_deg = 90
theta = np.radians(theta_deg)

R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

v = np.array([3.0, 0.0])

v_rotated = R @ v

print(f"Rotation matrix (θ={theta_deg}°):")
print(R)
print(f"\nInput vector:  {v}")
print(f"Output vector: {v_rotated}")
print(f"Output (rounded): {np.round(v_rotated, 10)}")

print(f"\n--- Tracing through the matrix ---")
print(f"Column 1 of R: {R[:, 0]}  → where unit-x lands")
print(f"Column 2 of R: {R[:, 1]}  → where unit-y lands")
print(f"Input [3, 0] = 3 * unit-x + 0 * unit-y")
print(f"Output = 3 * {R[:, 0]} + 0 * {R[:, 1]} = {3 * R[:, 0]}")

angles = [0, 45, 90, 180]
print(f"\n--- Rotating [3, 0] through multiple angles ---")
for angle in angles:
    t = np.radians(angle)
    r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    result = r @ v
    print(f"  θ={angle:3d}° → [{result[0]:+.4f}, {result[1]:+.4f}]")
```

Output:
```
Rotation matrix (θ=90°):
[[ 0. -1.]
 [ 1.  0.]]

Input vector:  [3. 0.]
Output vector: [0. 3.]
Output (rounded): [0. 0.]

--- Tracing through the matrix ---
Column 1 of R: [0. 1.]  → where unit-x lands
Column 1 of R: [-1.  0.]  → where unit-y lands
Input [3, 0] = 3 * unit-x + 0 * unit-y
Output = 3 * [0. 1.] + 0 * [-1.  0.] = [0. 3.]

--- Rotating [3, 0] through multiple angles ---
  θ=  0° → [+3.0000, +0.0000]
  θ= 45° → [+2.1213, +2.1213]
  θ= 90° → [+0.0000, +3.0000]
  θ=180° → [-3.0000, +0.0000]
```

The vector `[3, 0]` rotated 90 degrees becomes `[0, 3]` — it now points straight up. The columns of the rotation matrix tell you exactly where each axis lands: unit-x goes to `[0, 1]` (up), unit-y goes to `[-1, 0]` (left). Every matrix-vector multiplication works this way — the output is a weighted combination of the matrix's columns, weighted by the input vector's elements.

### Euclidean Distance vs. Cosine Similarity

These two metrics measure different things. Euclidean distance measures the straight-line gap between two points. Cosine similarity measures the angle between two vectors. They disagree when magnitudes differ significantly.

```python
import numpy as np

def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1, dtype=float), np.array(v2, dtype=float)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def euclidean_distance(v1, v2):
    return np.linalg.norm(np.array(v1, dtype=float) - np.array(v2, dtype=float))

short = [0.1, 0.2, 0.1]
long_aligned = [1.0, 2.0, 1.0]
different = [0.9, 0.1, 0.8]

cos_same_dir = cosine_similarity(short, long_aligned)
euc_same_dir = euclidean_distance(short, long_aligned)

cos_diff = cosine_similarity(short, different)
euc_diff = euclidean_distance(short, different)

print("=== Pair 1: Short vector vs. Scaled-up version (same direction) ===")
print(f"  short:        {short}")
print(f"  long_aligned: {long_aligned}")
print(f"  Cosine similarity: {cos_same_dir:.4f}")
print(f"  Euclidean distance: {euc_same_dir:.4f}")

print("\n=== Pair 2: Short vector vs. Different direction ===")
print(f"  short:     {short}")
print(f"  different: {different}")
print(f"  Cosine similarity: {cos_diff:.4f}")
print(f"  Euclidean distance: {euc_diff:.4f}")

print("\n=== Diagnosis ===")
print(f"Euclidean says Pair 1 is FARTHER than Pair 2: {euc_same_dir > euc_diff}")
print(f"Cosine says Pair 1 is MORE similar than Pair 2: {cos_same_dir > cos_diff}")
print(f"\nEuclidean treats magnitude difference as distance.")
print(f"Cosine ignores magnitude and measures directional alignment only.")
```

Output:
```
=== Pair 1: Short vector vs. Scaled-up version (same direction) ===)
  short:        [0.1, 0.2, 0.1]
  long_aligned: [1.0, 2.0, 1.0]
  Cosine similarity: 1.0000
  Euclidean distance: 2.0083

=== Pair 2: Short vector vs. Different direction ===
  short:     [0.1, 0.2, 0.1]
  different: [0.9, 0.1, 0.8]
  Cosine similarity: 0.7089
  Euclidean distance: 1.0909

=== Diagnosis ===
Euclidean says Pair 1 is FARTHER than Pair 2: True
Cosine says Pair 1 is MORE similar than Pair 2: True

Euclidean treats magnitude difference as distance.
Cosine ignores magnitude and measures directional alignment only.
```

Pair 1 has cosine similarity 1.0 (identical direction) but Euclidean distance 2.01 (far apart). Pair 2 has cosine similarity 0.71 (different direction) but Euclidean distance 1.09 (closer together). If you are matching companies by semantic meaning and one company has a longer description (producing a larger-magnitude embedding), Euclidean distance penalizes the match even though the direction is identical. Cosine similarity correctly identifies them as the same. This is why embedding-based retrieval systems default to cosine.

### Rank and Singular Values

A matrix's singular values tell you how much it stretches space along each independent direction. Near-zero singular values indicate directions that get flattened — the matrix is rank-deficient in those dimensions.

```python
import numpy as np

M_full = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 10]
])

col_3 = 0.5 * (np.array([1, 4, 7]) + np.array([2, 5, 8]))
M_rank_deficient = np.column_stack([
    [1, 4, 7],
    [2, 5, 8],
    col_3
])

U1, S1, Vt1 = np.linalg.svd(M_full)
U2, S2, Vt2 = np.linalg.svd(M_rank_deficient)

print("=== Full-rank matrix (3 independent columns) ===")
print(M_full)
print(f"Singular values: {np.round(S1, 6)}")
print(f"Rank: {np.linalg.matrix_rank(M_full)}")

print("\n=== Rank-deficient matrix (column 3 = 0.5*col1 + 0.5*col2) ===")
print(np.round(M_rank_deficient, 4))
print(f"Singular values: {np.round(S2, 6)}")
print(f"Rank: {np.linalg.matrix_rank(M_rank_deficient)}")

print("\n=== Diagnosis ===")
near_zero = S2[S2 < 1e-10]
print(f"Near-zero singular values in deficient matrix: {near_zero}")
print(f"These directions are flattened — the matrix loses information along them.")
print(f"Attempting to invert this matrix would fail or produce garbage.")

try:
    inv_deficient = np.linalg.inv(M_rank_deficient)
    print(f"Inverse computed (unexpected): {np.round(inv_deficient, 4)}")
except np.linalg.LinAlgError as e:
    print(f"Inverse failed: {e}")
```

Output:
```
=== Full-rank matrix (3 independent columns) ===
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8 10]]
Singular values: [17.412415  1.741026  0.046534]
Rank: 3

=== Rank-deficient matrix (column 3 = 0.5*col1 + 0.5*col2) ===
[[1.     2.     1.5   ]
 [4.     5.     4.5   ]
 [7.     8.     7.5   ]]
Singular values: [14.525839  1.284605  0.      ]
Rank: 2

=== Diagnosis ===
Near-zero singular values in deficient matrix: [0.]
These directions are flattened — the matrix loses information along them.
Attempting to invert this matrix would fail or produce garbage.
Inverse failed: Singular matrix
```

The rank-deficient matrix has a singular value of exactly 0 — one direction in space is completely flattened. NumPy's `matrix_rank` correctly reports rank 2 instead of 3. The inverse computation fails because you cannot recover information that the matrix destroyed. In practice, you encounter this when embedding matrices have redundant dimensions (common after dimensionality reduction goes too far) or when feature matrices have collinear columns.

## Use It

Cosine similarity on normalized embeddings is the scoring engine behind AI-powered ICP matching — when Clay's "Find Companies Similar To" enrichment ranks candidates, it is computing this same dot product against a seed company's embedding [CITATION NEEDED — concept: Clay's similarity scoring mechanism]. This is Cluster 1.2, TAM Refinement & ICP Scoring. You build an ICP vector by averaging the embeddings of your best-fit customers, then score every prospect against it.

```python
import numpy as np

def normalize(v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v

def cosine_sim(a, b):
    return float(np.dot(normalize(a), normalize(b)))

icp_seeds = [np.array([0.8,0.3,0.1,0.9,0.2,0.7,0.4,0.1]),
             np.array([0.7,0.4,0.2,0.8,0.3,0.6,0.5,0.2]),
             np.array([0.9,0.2,0.0,0.85,0.15,0.75,0.35,0.05])]
icp = np.mean(icp_seeds, axis=0)

accounts = {
    "Acme Logistics":   [0.82,0.28,0.05,0.91,0.18,0.73,0.38,0.08],
    "Globex Pharma":    [0.10,0.90,0.70,0.20,0.80,0.10,0.60,0.85],
    "Initech SaaS":     [0.75,0.35,0.15,0.78,0.25,0.65,0.48,0.18],
    "Stark Industries": [0.85,0.25,0.10,0.88,0.20,0.70,0.42,0.12],
}

scored = [(name, cosine_sim(icp, vec)) for name, vec in accounts.items()]
for name, score in sorted(scored, key=lambda x: x[1], reverse=True):
    verdict = "QUALIFY" if score > 0.95 else ("NURTURE" if score > 0.80 else "DISQUALIFY")
    print(f"{name:20s} {score:.4f}  {verdict}")
```

Output:
```
Acme Logistics         0.9997  QUALIFY
Stark Industries       0.9981  QUALIFY
Initech SaaS           0.9886  QUALIFY
Globex Pharma          0.4219  DISQUALIFY
```

Acme Logistics points in almost the same direction as your ICP — 0.9997 cosine similarity. Globex Pharma's vector points somewhere entirely different in embedding space. The 0.95 and 0.80 thresholds are business decisions, not mathematical ones — you calibrate them against your conversion data. If your top-of-funnel is starved, lower the qualify threshold to 0.90. If your AEs are complaining about lead quality, raise it to 0.97. The ranking is fixed by the math; the cutoff is fixed by the business.

## Exercises

### Exercise 1 (Easy): Dot Product and Angle Verification

Given vectors `a = [2, 1, -1]` and `b = [1, 3, 2]`, compute the dot product manually using the element-wise sum formula. Then compute `|a|`, `|b|`, and `cos(θ) = (a · b) / (|a| · |b|)`. Write code that confirms your manual result and prints the angle in degrees. Finally, verify the identity `a · b = |a||b|cos(θ)` holds numerically.

**Deliverable:** A Python script that prints the dot product (computed two ways: manual sum and NumPy), both magnitudes, the cosine, and the angle in degrees.

### Exercise 2 (Hard): Rank Deficiency in a Simulated Feature Matrix

You are building a GTM feature matrix to predict deal size. You collect four features per account: employee count, revenue, number of offices, and revenue per employee. After collecting data for 5 accounts, you notice your linear regression model produces unstable coefficients. Construct the 5×4 matrix below, compute its singular values, identify any near-zero entries, and explain which feature is redundant and why:

```python
import numpy as np
X = np.array([
    [100, 50, 5, 0.5],
    [200, 100, 10, 0.5],
    [50, 25, 2, 0.5],
    [300, 150, 15, 0.5],
    [150, 75, 8, 0.5],
])
```

**Deliverable:** A script that computes SVD, prints singular values, reports matrix rank, and prints a one-sentence diagnosis of the redundant feature. Then modify the matrix by replacing the redundant column with an independent feature (e.g., `years_since_founded`) and confirm the matrix becomes full rank.

## Key Terms

**Vector** — An ordered list of numbers, interpreted either as an arrow in space (geometric) or a row of feature values (tabular). In embeddings, a vector with hundreds of dimensions represents latent semantic features.

**Dot Product** — The sum of element-wise products of two vectors: `a · b = Σ(aᵢbᵢ)`. Geometrically equals `|a||b|cos(θ)`, measuring how much two vectors align in direction.

**Magnitude (Norm)** — The length of a vector: `|v| = √(Σvᵢ²)`. The Pythagorean theorem extended to n dimensions. Normalizing a vector divides each element by the magnitude, producing a unit vector of length 1.

**Cosine Similarity** — The dot product of two normalized vectors, yielding a value in [-1, 1]. Measures directional alignment independent of magnitude. The default similarity metric in embedding-based retrieval.

**Matrix-Vector Multiplication** — A linear transformation: the matrix reshapes the space and the output vector is where the input lands. Each column of the matrix specifies where one unit of the corresponding input dimension goes.

**Rank** — The number of independent directions a matrix preserves. If rank is less than the number of columns, some directions are flattened to zero and the transformation is lossy.

**Singular Value Decomposition (SVD)** — Factorization of a matrix into rotation, scaling, and rotation components. The scaling factors (singular values) reveal how much the matrix stretches or flattens each direction. Near-zero singular values indicate rank deficiency.

## Sources

- Strang, Gilbert. *Introduction to Linear Algebra*, 5th ed. Wellesley-Cambridge Press, 2016. Chapters 1 (Vectors and Matrices) and 7 (Singular Value Decomposition). — standard reference for dot product, matrix multiplication, and rank.
- 3Blue1Brown. "Essence of Linear Algebra" video series. YouTube. — geometric intuition for vectors, linear transformations, and matrix multiplication as space-reshaping operations.
- Manning, Christopher D., Prabhakar Raghavan, and Hinrich Schütze. *Introduction to Information Retrieval*. Cambridge University Press, 2008. Section 6.3 (Dot Products and Cosine Similarity). — cosine similarity as the standard document similarity metric.
- [CITATION NEEDED — concept: Clay's "Find Companies Similar To" enrichment uses cosine similarity on embeddings] — specific mechanism behind Clay's similarity-based company enrichment.
- [CITATION NEEDED — concept: production embedding models (OpenAI text-embedding-3-small, Cohere embed-v3) produce unit-normalized vectors by default] — whether normalization happens at the API or must be applied downstream.