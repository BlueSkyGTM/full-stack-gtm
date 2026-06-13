# Matrix Transformations

## Hook

You've been working with vectors—now see what happens when a matrix reshapes an entire space. Rotation, scaling, shearing, and projection all reduce to one operation: multiply.

## Explain It

A matrix is a function from ℝⁿ to ℝᵐ. Describe the mechanism—columns as basis vectors, how multiplication re-maps every point simultaneously, and why composition of transformations is just matrix multiplication. Cover the four canonical 2D transforms (scale, rotate, shear, reflect) with their matrix forms, then generalize to higher dimensions.

## Show It

Working Python code (NumPy only, no plotting dependencies) that: defines each canonical 2D transform matrix, applies it to a set of vectors, prints before/after coordinates, chains two transforms via `@` operator, and prints the composed matrix alongside the sequential result to confirm they match. All output is terminal-visible text.

## Use It

Matrix transforms underpin every embedding operation in GTM tooling. When Clay [CITATION NEEDED — concept: Clay's embedding/vector lookup feature] computes similarity between a prospect vector and an ICP centroid, that dot product is a 1×n matrix multiplication. Dimensionality reduction on account feature matrices (PCA) is a matrix transform that projects high-dimensional firmographic data into a scoring plane. The redirect is Zone 03 (Targeting & Enrichment): ICP scoring and account segmentation rely on these linear maps.

## Ship It

Take the transformation patterns and implement a working account-feature transformer: accept a matrix of raw firmographic features (employee count, revenue, tech stack count), apply a normalization + projection transform, and output scored vectors. Exercise hook (medium difficulty): "Build a transform pipeline that normalizes and projects a 5×4 account feature matrix into a 2D scoring space—print the result."

## Prove It

Learning objectives (action verbs only):
1. Implement the four canonical 2D transformation matrices and apply them to arbitrary vectors.
2. Compose multiple transformations via matrix multiplication and verify equivalence to sequential application.
3. Diagnose what a given matrix does to a vector by reading its column vectors.
4. Build a transform pipeline that projects high-dimensional data into a lower-dimensional space.

Exercise hooks:
- Easy: "Given a matrix, print its column vectors and predict whether it scales, rotates, or shears."
- Medium: "Compose a 45° rotation with a 2× scale and verify the result matches sequential application on 5 test vectors."
- Hard: "Construct a projection matrix that collapses a 3D firmographic vector onto a 2D scoring plane—print the rank before and after to confirm information loss."