# Decision Trees and Random Forests

## Hook It
A single decision tree partitions feature space into axis-aligned rectangles by greedily selecting the split that maximizes impurity reduction. The mechanism is interpretable but unstable — small data changes produce different trees. Random forests reduce this variance by averaging predictions across many de-correlated trees trained on bootstrap samples with random feature subsets.

## Map It
Cover three mechanisms in sequence: (1) recursive binary splitting with Gini impurity and entropy as splitting criteria, (2) bootstrap aggregating (bagging) to generate decorrelated training sets, and (3) feature subsampling at each split to ensure tree diversity. The bias-variance decomposition explains why a forest outperforms a single tree: bias stays low, variance drops through averaging. Introduce the `scikit-learn` implementations (`DecisionTreeClassifier`, `RandomForestClassifier`) after the math is clear.

## Build It
Build a decision tree on a synthetic lead-conversion dataset with features like `company_size`, `page_views`, `email_opens`, and `time_on_site`. Print the tree rules using `export_text`. Then build a random forest on the same data and compare. Extract feature importances and display them. All code runs in terminal with printed output — no plots.

- **Easy**: Train a `DecisionTreeClassifier` on the dataset, print the top-level split rule and Gini impurity at the root.
- **Medium**: Train a `RandomForestClassifier` with 100 trees, print ranked feature importances, and compare train vs test accuracy against the single tree.
- **Hard**: Write a loop that sweeps `max_depth` from 1 to 15, records train and test accuracy for both models, and prints the results as a table to identify where overfitting begins.

## Test It
Validate model behavior through cross-validation and explicit checks. Confirm that the random forest's cross-validated accuracy exceeds the single tree's. Verify that feature importances are consistent across folds. Test edge cases: what happens when `max_depth=1` (a stump) versus unrestricted depth.

- **Easy**: Run 5-fold cross-validation on both models, print mean and std of accuracy.
- **Medium**: Perturb 5% of training labels with noise, retrain both models, and print the change in test accuracy to demonstrate the forest's variance resistance.
- **Hard**: Implement out-of-bag (OOB) error estimation by setting `oob_score=True`, compare OOB accuracy to held-out test accuracy, and print both.

## Use It
This is the **lead scoring and account prioritization** mechanism in GTM Zone 01 (Target). Feature importances from a random forest trained on historical conversion data tell you which signals actually predict closed-won deals — not which signals you *assume* matter. The ranked feature list becomes your qualification criteria. The trained model becomes a batch scorer for new accounts. [CITATION NEEDED — concept: Clay implementation of lead scoring with feature importance]

- **Easy**: Train a forest on the conversion dataset, print the top 3 features, and write one sentence interpreting each as a GTM signal.
- **Medium**: Build a batch inference pipeline: load a CSV of new accounts, run them through the trained forest, output a sorted list of account IDs by predicted conversion probability.
- **Hard**: Train two forests — one on won/lost deals, one on engaged/disengaged accounts — compare their feature importance rankings, and print a table showing where the signals diverge.

## Ship It
Persist the trained model with `joblib.dump`. Build an inference function that accepts a list of dictionaries (one per account), constructs a feature matrix, and returns predictions with probabilities. Add a guard: if any feature has a missing value, print a warning and skip that row. Document the expected feature schema in a printed summary.

- **Easy**: Save a trained random forest to disk and load it back; print the predicted class of a single new record to confirm round-trip works.
- **Medium**: Write a `score_accounts(input_csv, model_path, output_csv)` function that reads raw account data, applies the model, and writes account_id + predicted_probability to a new CSV.
- **Hard**: Add a data drift check: compute mean and std of each feature in the training set, store as a baseline, then compare incoming batch statistics and print a warning if any feature's mean has shifted more than 2 standard deviations.

---

## Learning Objectives

1. Implement a decision tree classifier and extract the splitting rules as readable text.
2. Compare single-tree vs random-forest accuracy on the same dataset using cross-validation.
3. Extract and rank feature importances from a trained random forest.
4. Diagnose overfitting by evaluating train/test accuracy across a range of `max_depth` values.
5. Build a batch inference pipeline that scores new accounts and writes results to CSV.