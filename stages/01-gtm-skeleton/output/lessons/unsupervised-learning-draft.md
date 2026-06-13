# Unsupervised Learning

## Hook

You have a spreadsheet of 4,000 accounts and zero labels. No "good fit / bad fit" column. No win/loss history. Unsupervised learning is the algorithm family designed for exactly this scenario—finding structure when no one handed you the answer key.

## Concept

The core mechanism: the algorithm defines an objective function (minimize intra-cluster distance, maximize variance retained, reconstruct input from bottleneck) and optimizes it without labeled examples. Three families matter for practitioners—clustering (K-Means, DBSCAN, HDBSCAN), dimensionality reduction (PCA, UMAP, t-SNE), and density estimation for anomaly detection. Each trades off interpretability, scalability, and assumptions about data shape. K-Means assumes spherical clusters of similar size; DBSCAN does not, but requires density hyperparameters. UMAP preserves local and global structure better than t-SNE but has its own hyperparameter sensitivities.

## Demonstration

Working Python script that clusters a synthetic accounts dataset (revenue, employee count, tech stack count) using K-Means and HDBSCAN, prints cluster assignments, and compares silhouette scores. Observable output: cluster labels, score comparison, and a plain-text confusion matrix showing where the two algorithms disagree.

## Guided Exercise

- **Easy**: Run the provided script on a different random seed. Print how cluster assignments shift.
- **Medium**: Replace K-Means with AgglomerativeClustering. Compare silhouette scores.
- **Hard**: Feed the clustered output into a supervised model as features. Measure whether cluster membership improves prediction of a synthetic target variable.

## Use It

In GTM, unsupervised learning surfaces in ICP discovery and account segmentation when you lack closed-win labels. [CITATION NEEDED — concept: unsupervised clustering for ICP discovery in GTM workflows] The mechanism: cluster accounts by firmographic and behavioral features, then inspect cluster composition to identify which segments correlate with downstream outcomes. This is the pattern behind territory planning and account prioritization when historical win data is sparse or unreliable. Tools like Clay do not implement this natively—this is a preprocessing step you run before feeding segmented lists into enrichment or outbound workflows.

## Ship It

Build a clustering pipeline that takes a CSV of accounts, engineers features (log-transformed revenue, encoded industry, tech stack binary vectors), runs HDBSCAN, and outputs a new CSV with a `segment_id` column. That segmented list becomes the input for personalized outbound sequences. [CITATION NEEDED — concept: clustered account lists feeding outbound sequence personalization] Evaluation: re-run quarterly, check whether segment boundaries shift—drift in cluster structure signals your market is changing.