# ML Pipelines

## Hook It
A model trained in a notebook with ad-hoc preprocessing will produce different scores at inference time when that preprocessing isn't applied identically. Pipelines exist to solve this: chain every transformation and prediction into a single serializable object so training and inference never drift. The hook is the silent failure mode — "my model works in dev but not in prod" — and the moment a practitioner realizes they need a pipeline.

## Define It
An ML pipeline is a directed acyclic graph of executable steps where each step consumes an output from the previous step. In the scikit-learn API specifically, a Pipeline is an ordered list of (name, transformer) tuples ending with an estimator, exposing a single `.fit()` / `.predict()` interface that applies all transformations in sequence. The key mechanisms: `fit_transform` on intermediate steps, `fit` only on the final estimator, and state isolation between steps.

## Explain It
Walk through the mechanism of how data flows through a pipeline: during `fit`, each transformer calls `fit_transform` and passes output to the next step; during `predict`, each transformer calls only `transform` (no fitting). Show how this prevents data leakage by ensuring fitted parameters (imputation values, scaling factors, vocabulary) are learned only from training data and frozen for test data. Demonstrate `Pipeline` from scikit-learn, then `make_pipeline` as a convenience. Show `ColumnTransformer` for applying different transformers to different feature subsets. Compare in-process pipelines (scikit-learn) vs. orchestrator-level pipelines (Airflow, ZenML) and when each is appropriate.

Code concepts to demonstrate:
- A pipeline with imputation → scaling → model that prints identical coefficients when re-run vs. manual approach that drifts
- A `ColumnTransformer` with different transformers for numeric vs. categorical columns
- Saving and loading a pipeline with `joblib` to confirm inference behavior is preserved

## Use It
GTM redirect to Zone 3 (Scoring & Prioritization): this is the pipeline pattern behind building a repeatable account-fit or lead-score model. When a practitioner says "enrich these firmographics, compute features, score against our ICP model," that is a pipeline — and if the enrichment preprocessing isn't chained, the score at inference won't match the score from training. Build a pipeline that takes raw company data, applies imputation for missing employee counts, one-hot encodes industry, scales revenue, and feeds into a classifier predicting conversion. This maps directly to how scoring models should be shipped in GTM stacks.

[CITATION NEEDED — concept: specific GTM cluster name for Zone 3 scoring pipeline application]

## Ship It
Covers pipeline serialization (`joblib.dump` / `joblib.load`), versioning strategies (timestamp + metric tagging), input schema validation at inference time (what happens when a new category appears), and monitoring for feature drift. Demonstrates a minimal production pattern: train pipeline, save with metadata, load and assert input shape before predicting.

Exercise hooks:
- (Easy) Serialize a trained pipeline, load it in a new process, and confirm predictions match
- (Medium) Add a custom transformer that validates input schema and raises on unexpected categories
- (Hard) Build a pipeline that logs input distributions on each `predict` call and flags when feature means drift beyond a threshold from training

## Quiz It
Assessment hooks targeting:
- The order of `fit_transform` vs `transform` calls during `fit` vs `predict`
- Why data leakage occurs when preprocessing is done outside a pipeline
- The behavior difference between `Pipeline` and `make_pipeline`
- What happens when a categorical value appears at inference that wasn't in training
- When to use `ColumnTransformer` vs. a single transformer

---

**Learning Objectives (draft):**
1. Build a scikit-learn Pipeline that chains preprocessing and model training into a single reproducible object
2. Detect and prevent data leakage by comparing manual preprocessing vs. pipeline-wrapped preprocessing
3. Configure a ColumnTransformer to apply different transformations to different feature subsets
4. Serialize and load a trained pipeline, confirming inference output is preserved across processes
5. Compare in-process pipelines (scikit-learn) to orchestrator-level pipelines (Airflow, ZenML) and identify when each is appropriate