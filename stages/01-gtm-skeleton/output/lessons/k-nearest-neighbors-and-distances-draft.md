# K-Nearest Neighbors and Distances

## Hook It
A new account lands in your CRM with 40 firmographic features. You have 10,000 labeled accounts—some churned, some expanded. KNN asks: which existing accounts are this account's nearest neighbors, and what happened to them? The entire algorithm is distance computation plus a vote. That's it.

## Define It
Cover the KNN mechanism in full: store all training points, compute distances from a query point to every stored point, sort by distance, take the top K, aggregate (majority vote for classification, mean for regression). Define four distance metrics—Euclidean, Manhattan, Cosine, Minkowski—with their formulas and when each breaks down. Explain how K controls bias-variance: K=1 overfits, K=N underfits, and odd K avoids tie votes in binary classification. Address the scaling requirement: features on different scales distort distance, so standardization is mandatory before any neighbor search.

## Show It
Implement KNN from scratch using only NumPy. Compute all four distance metrics between a query vector and a matrix of stored vectors. Run a classification pass with K=5 on a small synthetic dataset. Print the neighbor indices, their distances, their labels, and the final vote. Then demonstrate what happens when one feature is unscaled—the nearest neighbors flip entirely.

## Use It
[CITATION NEEDED — concept: lookalike account modeling in GTM, specifically using firmographic feature vectors to find similar accounts for targeting or scoring]

Build a lookalike model: given a set of closed-won accounts encoded as feature vectors (employee count, revenue, industry numeric, tech stack count), find the K nearest neighbors for a new prospect. Print the neighbor IDs, distances, and their outcomes. This is the mechanism behind "similar companies" features in enrichment platforms. Redirect: Zone 2, account scoring and prioritization—KNN provides a distance-based alternative to logistic regression for predicting account fit.

## Ship It
Write a CLI tool that accepts a JSON file of labeled accounts and a single query account, runs KNN with configurable K and distance metric, and prints the predicted outcome plus the neighbor table to stdout. Easy: fixed K=5, Euclidean only. Medium: add `--metric` flag supporting euclidean, manhattan, cosine. Hard: add `--k` sweep mode that prints accuracy across K=1 through K=20 using leave-one-out cross-validation on the provided dataset.

## Stretch It
Curse of dimensionality: as feature count grows, distances compress and neighbors become meaningless. Demonstrate with random vectors in 2D vs. 200D—ratio of nearest to farthest distance approaches 1.0. Approximate Nearest Neighbors (ANN) as the production answer: ball trees, KD-trees, and HNSW all trade exact results for sub-millisecond lookup at scale. Name FAISS and Annoy as implementations of approximate search. Weighted KNN: inverse-distance weighting so closer neighbors matter more than distant ones.