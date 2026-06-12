# Skill: DVC Data Version Control

## What you built

A two-stage DVC pipeline (preprocess → train) with content-addressed data tracking. Every dependency and output is hashed in `dvc.lock`. Re-running with no changes produces cache hits with zero re-execution.

## Reuse pattern

```bash
# Track a dataset
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc .gitignore
git commit -m "track raw dataset"

# Run the pipeline
dvc repro

# Commit the lock file
git add dvc.lock dvc.yaml
git commit -m "repro: first pipeline run"

# Push data to remote
dvc remote add -d s3remote s3://your-bucket/dvc-cache
dvc push

# On a new machine
git clone <repo>
dvc pull   # fetches exact cached bytes from S3
dvc repro  # all stages are cache hits
```

## dvc.yaml stage template

```yaml
stages:
  your-stage:
    cmd: python src/your_script.py
    deps:
      - src/your_script.py
      - data/input/
    outs:
      - data/output/
    metrics:
      - metrics/scores.json:
          cache: false
```

## Combined with MLflow

Record the DVC commit SHA as an MLflow tag for a complete audit trail:
```python
import subprocess, mlflow
sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
with mlflow.start_run():
    mlflow.set_tag("dvc_commit", sha)
    mlflow.log_metric("val_accuracy", acc)
```
