## Ship It

This script reads a CSV of accounts, engineers features (log-transforming skewed revenue and employee counts, one-hot encoding industry, keeping tech stack count as-is), runs HDBSCAN, and writes a new CSV with a `segment_id` column appended. That segmented list becomes the input for personalized outbound sequences.

```python
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
import tempfile, os

np.random.seed(42)
n = 250

csv_path = os.path.join(tempfile.gettempdir(), "accounts_input.csv")
pd.DataFrame({
    'company': [f'Company_{i:04d}' for i in range(n)],
    'revenue': np.exp(np.random.randn(n) * 1.3 + 7),
    'employees': np.exp(np.random.randn(n) * 1.1 + 4),
    'tech_stack_count': np.random.poisson(lam=8, size=n),
    'industry': np.random.choice(['SaaS', 'FinTech', 'Healthcare', 'Manufacturing'], size=n, p=[0.4, 0.25, 0.2, 0.15])
}).to_csv(csv_path, index=False)

df = pd.read_csv(csv_path)

df['log_revenue'] = np.log1p(df['revenue'])
df['log_employees'] = np.log1p(df['employees'])
df = pd.get_dummies(df, columns=['industry'], prefix='ind')

feature_cols = ['log_revenue', 'log_employees', 'tech_stack_count'] + \
               [c for c in df.columns if c.startswith('ind_')]

X = df[feature_cols].values
X_scaled = StandardScaler().fit_transform(X)

clusterer = HDBSCAN(min_cluster_size=12)
df['segment_id'] = clusterer.fit_predict(X_scaled)

output_path = os.path.join(tempfile.gettempdir(), "accounts_segmented.csv")
out_cols = ['company', 'revenue', 'employees', 'tech_stack_count', 'segment_id']
df[out_cols].to_csv(output_path, index=False)

print(f"Input:  {csv_path}")
print(f"Output: {output_path}")
print(f"Accounts processed: {len(df)}")
print(f"Segments found: {sorted(df['segment_id'].unique())}")
print(f"Noise accounts (segment_id = -1): {(df['segment_id'] == -1).sum()}")
print()

summary = df[df['segment_id'] != -1].groupby('segment_id').agg(
    count=('company', 'count'),
    median_revenue=('revenue', 'median'),
    median_employees=('employees', 'median'),
    avg_tech_stack=('tech_stack_count', 'mean')
).round(2)
print("Segment profiles (excluding noise):")
print(summary.to_string())
```

The segment profiles in the output are your ICP hypotheses. A segment with median revenue of $2M, 150 employees, and 12 technologies in the stack is a different ICP than one with $50M revenue, 1,200 employees, and 30 technologies. You validate these hypotheses by joining segment IDs back to your CRM and checking which segments actually converted over the past 6-12 months.

[CITATION NEEDED — concept: clustered account lists feeding outbound sequence personalization]

Drift detection is the maintenance pattern. Markets shift — new competitors emerge, funding environments change, technology stacks evolve. Re-run the clustering pipeline quarterly on fresh account data and compare the new segment boundaries to the previous run. If a segment that was previously coherent splits into two, or if a new dense cluster appears where none existed before, that is a signal your market is changing. The `segment_id` values will not be stable across runs (HDBSCAN does not produce deterministic labels across different data), so compare cluster profiles (centroids, sizes, feature distributions) rather than label IDs.

The output CSV with `segment_id` is a preprocessing artifact. You import it into Clay as an enriched column, use it to branch waterfall logic (different data enrichment lookups per segment), and route accounts into segment-specific outbound sequences. The clustering pipeline itself runs outside Clay — in a notebook, a scheduled job, or a dbt model — because it requires scikit-learn and enough compute to handle matrix operations on thousands of rows.