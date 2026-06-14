## Ship It

**Easy:** Add a fourth evaluator to the demo pipeline that checks output length is within a configurable range. Define a function that returns 1.0 if the output's character count falls between `min_len` and `max_len`, 0.0 otherwise. Wire it into `EVALUATOR_CONFIG` with a weight and threshold, then confirm the exit code changes when you set the threshold high enough that it fails.

```python
def length_range_eval(case: TestCase, min_len=20, max_len=200) -> float:
    n = len(case.output)
    return 1.0 if min_len <= n <= max_len else 0.0

length_entry = ("length_range", lambda c: length_range_eval(c, 20, 250), 0.2, 0.9)
```

Add `length_entry` to `EVALUATOR_CONFIG`, re-run the pipeline, and observe the report now shows four rows. Change the threshold to 1.0 (meaning every single case must pass) and confirm the exit code becomes 1.

**Medium:** Replace the hard-coded `DATASET` with a JSONL file loaded from disk. Write a function that reads test cases from `golden_dataset.jsonl` where each line is `{"input": "...", "expected": "...", "output": "..."}`. Add a `--calibrate` flag that runs the pipeline against this file, computes the 10th percentile score for each evaluator, and writes those values to `thresholds.json`. Then run the pipeline normally using the calibrated thresholds.

```python
import argparse
import json
import os
from percentile import percentile

def load_dataset(path):
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cases.append(TestCase(row["input"], row["expected"], row["output"]))
    return cases

def calibrate(dataset, config, percentile_value=10):
    thresholds = {}
    for name, eval_fn, weight, _ in config:
        scores = [eval_fn(c) for c in dataset]
        thresholds[name] = round(percentile(scores, percentile_value / 100), 3)
    with open("thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Calibrated thresholds written to thresholds.json: {thresholds}")
    return thresholds

def load_thresholds(path="thresholds.json"):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="golden_dataset.jsonl")
parser.add_argument("--calibrate", action="store_true")
args = parser.parse_args()

dataset = load_dataset(args.dataset)
saved_thresholds = load_thresholds()

if args.calibrate:
    calibrate(dataset, EVALUATOR_CONFIG)
    sys.exit(0)

effective_config = []
for name, eval_fn, weight, default_thresh in EVALUATOR_CONFIG:
    t = saved_thresholds.get(name, default_thresh) if saved_thresholds else default_thresh
    effective_config.append((name, eval_fn, weight, t))
```

Note: `percentile` here refers to `numpy.percentile` in practice. If you want zero dependencies, implement linear interpolation between sorted values manually. The mechanism — take the 10th percentile of historical scores as the floor — is what matters, not the library call.

**Hard:** Implement regression detection. Store each pipeline run's evaluator scores in a SQLite database with a timestamp. Add a `--baseline` flag that computes the rolling mean of the last N runs (default 5) and fails the build if the current run's aggregate drops more than 10% below that baseline. Print a comparison table showing current score, baseline score, delta, and pass/fail.

```python
import sqlite3
import time
from datetime import datetime

DB_PATH = "eval_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS run_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            evaluator_name TEXT,
            mean_score REAL,
            aggregate_score REAL
        )
    """)
    conn.commit()
    return conn

def store_run(conn, results, agg_score):
    ts = datetime.now().isoformat()
    for r in results:
        conn.execute(
            "INSERT INTO run_scores (timestamp, evaluator_name, mean_score, aggregate_score) VALUES (?, ?, ?, ?)",
            (ts, r.name, r.mean, agg_score)
        )
    conn.commit()

def get_baseline(conn, window=5):
    rows = conn.execute("""
        SELECT evaluator_name, AVG(mean_score) as baseline
        FROM (
            SELECT evaluator_name, mean_score, timestamp,
                   ROW_NUMBER() OVER (PARTITION BY evaluator_name ORDER BY timestamp DESC) as rn
            FROM run_scores
        ) WHERE rn <= ?
        GROUP BY evaluator_name
    """, (window,)).fetchall()
    return {row[0]: row[1] for row in rows}

def check_regression(results, baseline, max_drop_pct=0.10):
    regressions = []
    for r in results:
        if r.name in baseline:
            b = baseline[r.name]
            drop = (b - r.mean) / b if b > 0 else 0
            if drop > max_drop_pct:
                regressions.append((r.name, r.mean, b, drop))
    return regressions

def print_regression_table(results, baseline):
    print("\n" + "=" * 75)
    print(f"{'EVALUATOR':<20} {'CURRENT':>8} {'BASELINE':>10} {'DELTA':>8} {'STATUS':>10}")
    print("-" * 75)
    for r in results:
        b = baseline.get(r.name, r.mean)
        delta = r.mean - b
        status = "OK" if abs(delta) <= 0.10 * b else "REGRESS"
        print(f"{r.name:<20} {r.mean:>8.3f} {b:>10.3f} {delta:>+8.3f} {status:>10}")
    print("=" * 75)
```

Wire these into `main()`: call `init_db()` at the top, `store_run()` after computing scores, and `check_regression()` before the exit decision. When `--baseline` is passed, load the last 5 runs, compute the rolling mean per evaluator, and fail if any evaluator drops more than 10% below its baseline. The regression check is a second gate on top of the absolute threshold — a run can pass all thresholds and still fail regression if quality is decaying.