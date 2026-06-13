# Linear Systems

## Hook

Every lead-scoring model, every attribution weight calculation, and every "rank these accounts" feature in your GTM stack is solving a linear system under the hood. If you've ever wondered why your CRM's priority score behaves the way it does — or why it sometimes breaks — this is the mechanism.

## Concept

A linear system is a set of equations where each unknown appears only to the first power, multiplied by a constant. The entire system can be represented as a matrix-vector equation **Ax = b**. Three outcomes: one solution, no solution, or infinite solutions. Determining which outcome you have — and extracting the solution when it exists — is what linear algebra solvers do. NumPy's `linalg.solve` implements LU decomposition for this. When no unique solution exists, `linalg.lstsq` finds the best approximation by minimizing squared error.

## Use It

**GTM Redirect:** This is the mechanism behind Zone 01 — lead scoring and multi-touch attribution. When you assign weights to firmographic signals (company size × 0.4 + intent score × 0.6 = priority), you are evaluating a linear function. When you solve for the weights themselves from historical conversion data, you are solving a linear system. [CITATION NEEDED — concept: linear systems in lead scoring weight optimization]

## Build It

Build a minimal lead-scoring weight solver. Given a matrix of signal values (rows = past accounts, columns = signals) and a vector of outcomes (converted or not), solve for the weight vector. Print the weights and predicted scores. Exercise hooks: Easy — solve a 3×3 system with known solution; Medium — solve an overdetermined system with `lstsq`; Hard — handle a singular matrix and detect it programmatically.

## Ship It

In production, your signal matrix will have missing values, collinear columns (company revenue and employee count are correlated), and noise. Document the failure modes: singular matrices, ill-conditioned systems, and what `np.linalg.cond` tells you about trustworthiness. Ship a function that takes raw account data, validates the matrix condition number, and returns scored accounts or a warning.

## Extend It

Eigenvalues and eigenvectors are the natural extension — they reveal which dimensions of your signal matrix actually carry independent information. This is the mechanism behind PCA (principal component analysis), which GTM tools use for feature reduction when you have dozens of firmographic signals and need to collapse them. [CITATION NEEDED — concept: PCA in CRM feature reduction] SVD (singular value decomposition) generalizes this to non-square matrices and is what powers recommender systems.