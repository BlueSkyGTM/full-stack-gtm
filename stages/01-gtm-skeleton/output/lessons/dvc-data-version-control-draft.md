# DVC: Data Version Control

## Hook

Git tracks code. It chokes on datasets. DVC extends Git's content-addressed storage model to files that don't fit in a repository—CSVs, model artifacts, feature stores. Without it, "which training set produced this model?" becomes an archaeological dig.

## Concept Introduction

DVC attaches a `.dvc` pointer file to each tracked artifact. The pointer stores a content hash and remote location. Git tracks the pointer; DVC manages the bytes. This section covers the hash-based addressing mechanism, how `dvc push` / `dvc pull` sync with remote storage, and how `dvc.yaml` defines pipeline stages with dependency graphs.

## Guided Walkthrough

Build a minimal DVC pipeline from scratch: initialize a repo, track a dataset, define a single-stage pipeline in `dvc.yaml`, reproduce it, and inspect `dvc.lock`. Every command produces observable output—file creation, hash printing, diff reporting.

**Exercise hooks:**
- *Easy:* Initialize a DVC project and track one file; verify the `.dvc` pointer contains a matching MD5 hash.
- *Medium:* Define a two-stage pipeline in `dvc.yaml` (extract → transform) and run `dvc repro`; confirm `dvc.lock` captures both stages.
- *Hard:* Change the source data, re-run `dvc repro`, and use `dvc dag` plus `dvc metrics diff` to observe which stages re-executed and why.

## Use It

**GTM Redirect:** This is foundational for Zone 1 (Data Engineering). Any GTM workflow that retrains models on updated lead data—intent signals, enrichment pipelines, scoring models—needs reproducible data versioning. Without it, a model regression cannot be traced to a data change. DVC provides the mechanism: hash-linked artifacts, pipeline DAGs, and experiment comparison.

**Exercise hooks:**
- *Medium:* Version a lead-scoring training dataset with DVC; modify one row, commit the change, and use `dvc diff` to confirm the hash changed.
- *Hard:* Build a `dvc.yaml` pipeline that runs a feature transform on CRM export data and outputs a training-ready `.parquet` file; tag the experiment with `dvc exp run` and compare metrics across two parameter sets.

## Ship It

Integrate DVC into a CI workflow: on push, `dvc pull` restores cached data, `dvc repro` rebuilds the pipeline, and `dvc push` stores new artifacts. Wire `dvc metrics show` into PR comments so reviewers see whether accuracy improved or regressed—all without manual file juggling.

**Exercise hooks:**
- *Hard:* Write a GitHub Actions workflow (or equivalent CI script) that runs `dvc pull`, `dvc repro`, and `dvc push` on every push to `main`; print `dvc metrics show` to stdout as observable CI output.

## Evaluate

Assessment covers: hash-based addressing vs. Git LFS, pointer-file structure, pipeline stage dependencies, reproduction semantics (when does a stage re-run?), and remote storage configuration.

**Exercise hooks:**
- *Medium:* Given a broken `dvc.yaml` where a stage dependency is mislabeled, identify the error and fix it so `dvc repro` succeeds and prints stage completion.
- *Hard:* A tracked dataset was modified manually (not through DVC). Explain why `dvc checkout` restores the old version, and demonstrate the restoration with observable output.