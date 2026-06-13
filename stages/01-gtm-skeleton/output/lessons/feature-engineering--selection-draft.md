# Feature Engineering & Selection

## Beat 1: Hook — Raw Data Is Never Enough

The model you deployed last week is underperforming. The data is "clean" — no nulls, proper types — but the signal-to-noise ratio is killing you. This beat opens with a concrete scenario: a B2B lead scoring model that has 47 firmographic columns but only 3 actually predict conversion. The problem isn't the algorithm. The problem is the features.

## Beat 2: Concept — Transform, Create, Select

Three distinct mechanisms, in order:

**Feature Transformation**: Rescaling existing features so algorithms don't misinterpret magnitude as importance. Covers standardization (zero mean, unit variance), min-max normalization, and log transforms for skewed distributions like company revenue or employee count.

**Feature Creation**: Deriving new columns that encode domain knowledge. Interaction terms (employee_count × funding_stage), binning (revenue buckets), and temporal deltas (days_since_funding_round). The mechanism: compress raw signals into representations the model can exploit with fewer parameters.

**Feature Selection**: Removing features that add noise or collinearity. Three approaches — filter methods (variance threshold, correlation-based), wrapper methods (recursive feature elimination), and embedded methods (L1 regularization, tree-based importance). The mechanism: each approach trades compute cost for selection accuracy differently.

---

**GTM Redirect Context**: Feature engineering maps directly to enrichment pipeline design. When you pull 40+ firmographic signals from a data provider into a prospect record, most of those signals are noise. Feature selection tells you which 5 signals actually predict ICP fit. This is the statistical foundation for deciding which enrichment columns matter in a Clay waterfall — specifically Zone 2 (ICP Scoring) and Zone 3 (Signal Detection). [CITATION NEEDED — concept: enrichment-to-feature pipeline in GTM topic map]

## Beat 3: Code — Build, Transform, Rank

Working code examples in Python:

**3a. Feature Engineering Pipeline**
- Load a synthetic B2B dataset (company_id, revenue, employee_count, industry, funding_rounds, months_since_last_funding)
- Demonstrate: log transform on revenue, standardization on employee_count, one-hot encoding on industry, interaction term (employees_per_funding_round), temporal binning on months_since_last_funding
- Print before/after shapes and sample rows to confirm transformation

**3b. Feature Selection Pipeline**
- Using the engineered dataset from 3a
- Variance threshold to drop near-zero-variance columns
- Correlation matrix to identify and remove one feature from each pair with |r| > 0.85
- Permutation importance using a small sklearn model (RandomForestClassifier) to rank remaining features
- Print final feature list with importance scores

All output is terminal-printed. No visualizations that require a browser.

## Beat 4: Use It — Which Enrichment Signals Actually Matter

Connect feature selection to GTM enrichment decisions:

- A Clay waterfall pulls firmographic data from multiple providers. Each provider returns 10-30 columns. Your prospect table now has 80+ columns. Feature selection — specifically variance threshold and permutation importance — tells you which 8 columns to feed into your ICP scoring model. The rest are noise you're paying to enrich and store.

- Mechanism → tool: Run feature importance once on historical closed-won/lost data. The output is a ranked list. Use that ranked list to configure which enrichment columns your waterfall actually writes to the prospect record. This reduces API costs and speeds up enrichment.

- Redirect: This is Zone 2 (ICP Scoring) — the ranked feature list becomes the input schema for your scoring model.

## Beat 5: Ship It — Exercises

**Easy**: Given a dataset with `revenue` and `employee_count`, create two new features — `log_revenue` and `revenue_per_employee` — and print summary statistics for all four columns.

**Medium**: Load a dataset with 15 features. Apply variance threshold (threshold=0.01) and correlation-based removal (|r| > 0.8). Print the dropped features and the reason each was dropped.

**Hard**: Build a complete feature engineering and selection pipeline for a synthetic B2B lead scoring dataset. Start with 20 raw columns. Engineer 5 derived features. Use recursive feature elimination to select the top 8 features. Print the final feature set with rankings. Compare model accuracy (any sklearn classifier) using all 25 features vs. the selected 8.

## Beat 6: Review — What You Carry Forward

- Raw data almost never has the right representation. Transformation fixes scale; creation encodes domain knowledge; selection removes noise.
- Feature selection is not optional when you're enriching from multiple providers. Without it, you're paying for and processing signals that hurt model performance.
- The ranked feature list from selection is a deliverable — it becomes the schema configuration for enrichment pipelines and the input specification for scoring models.
- Next lesson: the selected features feed into [subsequent lesson topic].

---

**Learning Objectives** (testable, action-verb):

1. Transform skewed numeric features using log scaling and standardization, and print before/after distributions to confirm the transformation.
2. Engineer interaction and temporal features from raw columns, and explain which domain assumption each engineered feature encodes.
3. Implement a variance-threshold filter and correlation-based feature removal, and output the list of dropped features with the drop reason.
4. Rank features by permutation importance using an sklearn model, and select the top-k features that account for 90% of cumulative importance.
5. Compare model performance using all features versus a selected subset, and quantify the accuracy-to-complexity tradeoff.