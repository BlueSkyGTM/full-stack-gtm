# MLOps 01 — Experiment Tracking

---

## Beat 1: Hook It

You trained three models this week. Two outperformed the baseline. You cannot remember which learning rate you used for either of them. This beat establishes the reproducibility crisis that experiment tracking solves — not as a theoretical concern, but as an operational failure you have already experienced.

---

## Beat 2: Explain It

An experiment tracker is a structured write-ahead log for model training. Every training run records three categories of information: parameters (inputs you set), metrics (outputs you measure), and artifacts (files you produce). The mechanism is append-only logging — the tracker never mutates past records, only adds new ones. This beat covers the minimal schema (run ID, timestamp, parameters, metrics, artifacts) and the comparison operation it enables: sorting runs by metric to find the best configuration.

---

## Beat 3: Build It

Write a Python script that trains a simple classifier on a synthetic dataset, logs hyperparameters and accuracy to a local MLflow tracking server, and prints the top 5 runs sorted by accuracy. MLflow is introduced as the tool here — it solves the problem of "where do I write these records" by providing a local SQLite-backed store and a comparison API. The code runs unmodified in a terminal and prints observable output (a ranked table of runs).

**Exercise hooks:**
- *(Easy)* Modify one hyperparameter and re-run. Confirm the new run appears in the printed results.
- *(Medium)* Add a second metric (F1 score) to the logging. Print a comparison table with both metrics.
- *(Hard)* Log a confusion matrix image as an artifact for each run. Retrieve and display the artifact path for the best-performing run.

---

## Beat 4: Use It

GTM cluster: **ICP Scoring & Enrichment** (Zone 2 — Enrichment & Scoring). When you build predictive models for lead scoring, you run dozens of experiments with different feature sets and thresholds. Experiment tracking lets you answer "which feature configuration produced the highest conversion-rate lift?" without guessing. The redirect: this is not a theoretical connection — every scoring model you ship to Clay or a CRM went through an experiment phase, and if you didn't track it, you cannot defend the choice or improve it.

---

## Beat 5: Ship It

Promote the best run from your local experiment tracker to a model registry with a status label ("staging", "production"). This beat covers the transition from tracking (recording what happened) to registry (declaring what should happen). You will register your top-performing model and print its registry entry with version and status.

**Exercise hooks:**
- *(Easy)* Register the best run and print its version number.
- *(Medium)* Transition a model from "staging" to "production" via the MLflow API. Print confirmation.
- *(Hard)* Write a script that queries the registry, finds the current "production" model, and prints its parameters — simulating what a serving endpoint would do at inference time.

---

## Beat 6: Prove It

Five quiz questions grounded in the mechanisms above. Questions target: the three categories of logged data, the append-only property, the comparison operation, the difference between tracking and registry, and the retrieval of artifact paths. No trivia — every question traces to a specific objective in the lesson.

**Learning objectives (for `docs/en.md`):**
1. Configure an MLflow tracking server and log parameters, metrics, and artifacts for a training run.
2. Compare multiple experiment runs by sorting on a target metric and retrieving the best configuration.
3. Register a trained model in a model registry and assign it a lifecycle stage.
4. Explain the difference between experiment tracking (recording) and model registry (deployment declaration).
5. Retrieve a logged artifact path from a completed experiment run.