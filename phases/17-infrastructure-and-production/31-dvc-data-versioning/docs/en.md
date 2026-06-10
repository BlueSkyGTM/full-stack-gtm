# DVC: Data Version Control

> Your model accuracy dropped 4% overnight. The training code did not change. The model architecture did not change. The hyperparameters did not change. The data changed — a pipeline upstream added 12,000 mislabeled rows and nobody noticed because data files are not tracked in Git. DVC is what you add so that "what data produced this model" has a definitive answer.

**Type:** Build
**Languages:** Python, Bash (DVC CLI)
**Prerequisites:** Phase 17/29 (MLflow), Git fundamentals
**Time:** ~70 minutes

## Learning Objectives

- Explain why Git alone cannot version large data files, and what DVC's content-addressable cache solves.
- Initialize a DVC project, track a dataset with `dvc add`, and reproduce a two-stage pipeline with `dvc repro`.
- Read a `dvc.lock` file and explain what each field guarantees about reproducibility.
- Connect a DVC remote (local filesystem or S3) and push/pull data independently of code.

## The Problem

Git versions code. It does not version data. Committing a 5GB CSV to a Git repository makes the repository slow for everyone on the team forever. Storing the data outside Git means the link between "this commit" and "this dataset" exists only in someone's memory or a Slack message.

The result: you cannot reproduce a model. You cannot audit which data version caused a performance regression. You cannot roll back to last week's clean dataset after a pipeline bug corrupts the current one. MLflow (Lesson 29) tracks what happened during training; DVC tracks what the training consumed.

## The Concept

### Content-addressable storage

DVC replaces large files with small pointer files. When you run `dvc add data/train.csv`:
1. DVC computes the MD5 hash of `data/train.csv`
2. Copies the file into `.dvc/cache/` under a path derived from the hash
3. Writes a `data/train.csv.dvc` pointer file containing the hash and size
4. Adds `data/train.csv` to `.gitignore`

You commit the `.dvc` pointer file to Git. The actual data lives in the cache (local or remote). Two identical files share one cache entry — no duplication.

```
data/train.csv.dvc:
  md5: a3b4c5d6...
  size: 52428800
  path: train.csv
```

This file is what you version in Git. When a colleague checks out your commit and runs `dvc pull`, DVC fetches the exact bytes that correspond to that hash from the remote.

### Pipelines and `dvc.yaml`

DVC pipelines define a DAG (directed acyclic graph) of stages. Each stage has inputs (`deps`), outputs (`outs`), and a command:

```yaml
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - src/preprocess.py
      - data/raw/train.csv
    outs:
      - data/processed/train.parquet

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/train.parquet
    outs:
      - models/classifier.pkl
    metrics:
      - metrics/scores.json
```

`dvc repro` runs only the stages whose dependencies have changed. If `data/raw/train.csv` is unchanged and `preprocess.py` is unchanged, the preprocess stage is a cache hit and is skipped. If you change `train.py`, only the train stage reruns.

### `dvc.lock` — the reproducibility record

After `dvc repro`, DVC writes `dvc.lock` with the MD5 hashes of every dependency and output:

```yaml
schema: '2.0'
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
    - path: data/raw/train.csv
      md5: a3b4c5d6...
    outs:
    - path: data/processed/train.parquet
      md5: f7e8d9c0...
```

Commit `dvc.lock` to Git alongside the code. Any future run of `git checkout <commit> && dvc repro` will reproduce the exact pipeline state — same data, same outputs — because every hash is pinned.

### Remotes

A DVC remote is where the cached data actually lives. Without a remote, data stays on your local machine only:

```bash
# Add a local directory as remote (for this lesson)
dvc remote add -d myremote /tmp/dvc-remote

# Add an S3 remote (production)
dvc remote add -d s3remote s3://my-bucket/dvc-cache

# Push all tracked data to the remote
dvc push

# Pull data on a new machine after git clone
dvc pull
```

The remote does not need to be Git. It can be S3, GCS, Azure Blob, SSH, or a local path. The remote stores the content-addressed cache; Git stores the pointer files and lock files.

### DVC + MLflow

DVC and MLflow are complementary, not competing:
- **DVC** tracks what went in: data versions, pipeline stages, output hashes
- **MLflow** tracks what came out: parameters, metrics, model artifacts

A complete audit trail combines both: the MLflow run ID tells you the accuracy; the `dvc.lock` commit tells you the exact dataset and pipeline that produced it.

## Build It

Build a two-stage DVC pipeline: a preprocess stage that filters and splits a dataset, and a train stage that trains a classifier and writes metrics.

`code/main.py` runs the full demo end-to-end:
1. Generates a synthetic dataset (`data/raw/dataset.csv`)
2. Initializes a DVC project (`dvc init`)
3. Tracks the raw data (`dvc add data/raw/dataset.csv`)
4. Defines a two-stage pipeline in `dvc.yaml`
5. Runs `dvc repro` to execute both stages
6. Reads `dvc.lock` and prints the hash of every dependency and output
7. Runs `dvc repro` again to show the cache-hit behavior (nothing reruns)

The test suite validates the pipeline logic by importing the stage functions directly, without running DVC CLI commands.
