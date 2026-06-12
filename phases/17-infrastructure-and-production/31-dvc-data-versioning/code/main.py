"""
DVC data versioning — Phase 17, Lesson 31.

Demonstrates a full DVC workflow:
  - dvc init in a temp project directory
  - dvc add to track a raw dataset
  - dvc.yaml pipeline with preprocess + train stages
  - dvc repro to execute and cache the pipeline
  - dvc.lock inspection to verify hashes
  - Second dvc repro to show cache-hit (no re-execution)

Run:
    pip install dvc scikit-learn pandas numpy
    python code/main.py
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Stage functions (also importable for tests) ────────────────────────────────

def generate_dataset(output_path: Path, n_samples: int = 500) -> None:
    """Generate a synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=n_samples, n_features=10, n_informative=5,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    df["label"] = y
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Generated {len(df)} rows -> {output_path}")


def preprocess(input_path: Path, train_out: Path, test_out: Path) -> None:
    """Split raw dataset into train/test and save as parquet."""
    df = pd.read_csv(input_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_out, index=False)
    test_df.to_parquet(test_out, index=False)
    print(f"  Preprocessed: {len(train_df)} train, {len(test_df)} test")


def train(train_path: Path, test_path: Path,
          model_out: Path, metrics_out: Path) -> None:
    """Train a classifier and write accuracy metrics to JSON."""
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    X_train = train_df.drop("label", axis=1).values
    y_train = train_df["label"].values
    X_test = test_df.drop("label", axis=1).values
    y_test = test_df["label"].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_train, y_train)

    train_acc = float(clf.score(X_train, y_train))
    test_acc = float(clf.score(X_test, y_test))
    metrics = {"train_accuracy": train_acc, "test_accuracy": test_acc}

    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    import pickle
    model_out.write_bytes(pickle.dumps((scaler, clf)))
    metrics_out.write_text(json.dumps(metrics, indent=2))
    print(f"  Trained: train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")


# ── DVC orchestration helpers ──────────────────────────────────────────────────

def run(cmd: str, cwd: Path) -> str:
    """Run a shell command and return stdout."""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr.strip()}")
        result.check_returncode()
    return result.stdout.strip()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def write_pipeline(project: Path) -> None:
    """Write dvc.yaml defining the two-stage pipeline."""
    dvc_yaml = textwrap.dedent("""
        stages:
          preprocess:
            cmd: python pipeline.py preprocess
            deps:
              - pipeline.py
              - data/raw/dataset.csv
            outs:
              - data/processed/train.parquet
              - data/processed/test.parquet

          train:
            cmd: python pipeline.py train
            deps:
              - pipeline.py
              - data/processed/train.parquet
              - data/processed/test.parquet
            outs:
              - models/classifier.pkl
            metrics:
              - metrics/scores.json:
                  cache: false
    """).strip()
    (project / "dvc.yaml").write_text(dvc_yaml)


def write_pipeline_script(project: Path) -> None:
    """Write pipeline.py that DVC calls for each stage."""
    script = textwrap.dedent("""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from code.main import preprocess, train

        stage = sys.argv[1]
        if stage == "preprocess":
            preprocess(
                Path("data/raw/dataset.csv"),
                Path("data/processed/train.parquet"),
                Path("data/processed/test.parquet"),
            )
        elif stage == "train":
            train(
                Path("data/processed/train.parquet"),
                Path("data/processed/test.parquet"),
                Path("models/classifier.pkl"),
                Path("metrics/scores.json"),
            )
    """).strip()
    (project / "pipeline.py").write_text(script)


def parse_lock(project: Path) -> dict:
    """Parse dvc.lock and return a summary of hashes."""
    import yaml
    lock_path = project / "dvc.lock"
    if not lock_path.exists():
        return {}
    with open(lock_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        project = Path(tmpdir) / "dvc-demo"
        project.mkdir()

        # Step 1: Git init (DVC requires a Git repo)
        print("\n[1] Initializing Git + DVC...")
        run("git init", project)
        run("git config user.email demo@example.com", project)
        run("git config user.name Demo", project)
        run("dvc init", project)
        print("    DVC initialized")

        # Step 2: Generate and track raw data
        print("\n[2] Generating dataset and tracking with DVC...")
        raw_path = project / "data" / "raw" / "dataset.csv"
        generate_dataset(raw_path)
        run("dvc add data/raw/dataset.csv", project)
        raw_hash = md5_file(raw_path)
        print(f"    dataset.csv MD5: {raw_hash[:16]}...")

        # Step 3: Write pipeline definition
        print("\n[3] Writing dvc.yaml pipeline...")
        write_pipeline(project)
        write_pipeline_script(project)
        print("    dvc.yaml written with preprocess + train stages")

        # Step 4: First dvc repro
        print("\n[4] Running dvc repro (first run — all stages execute)...")
        run("dvc repro", project)
        metrics = json.loads((project / "metrics" / "scores.json").read_text())
        print(f"    Metrics: {metrics}")

        # Step 5: Inspect dvc.lock
        print("\n[5] Inspecting dvc.lock...")
        lock = parse_lock(project)
        if lock and "stages" in lock:
            for stage_name, stage_data in lock["stages"].items():
                print(f"    Stage: {stage_name}")
                for dep in stage_data.get("deps", []):
                    h = dep.get("md5", dep.get("hash", "?"))
                    print(f"      dep  {dep['path']}: {str(h)[:16]}...")
                for out in stage_data.get("outs", []):
                    h = out.get("md5", out.get("hash", "?"))
                    print(f"      out  {out['path']}: {str(h)[:16]}...")

        # Step 6: Second dvc repro — all stages cached
        print("\n[6] Running dvc repro again (no changes — all stages cached)...")
        result = subprocess.run(
            "dvc repro", shell=True, cwd=project,
            capture_output=True, text=True,
        )
        output = result.stdout + result.stderr
        if "Stage" not in output and ("cached" in output.lower() or
                                       "unchanged" in output.lower() or
                                       output.strip() == ""):
            print("    All stages cached — nothing reran.")
        else:
            # DVC may say "Stage 'X' didn't change, skipping"
            for line in output.splitlines():
                if line.strip():
                    print(f"    {line}")

        print("\nDone. Key files created:")
        print(f"  data/raw/dataset.csv.dvc  — pointer file (commit to Git)")
        print(f"  dvc.yaml                  — pipeline definition (commit to Git)")
        print(f"  dvc.lock                  — reproducibility record (commit to Git)")
        print(f"  .dvc/cache/               — content-addressed cache (gitignored)")


if __name__ == "__main__":
    main()
