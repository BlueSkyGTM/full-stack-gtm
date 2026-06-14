## Ship It

I/O redirection and exit-code handling are the mechanism behind every production cron job and CI pipeline that runs inference for GTM data workflows. The script below is a complete inference runner: it creates a virtual environment, executes a Python script, splits stdout and stderr into timestamped log files, and exits with a nonzero code if inference fails — which is exactly the contract a CI runner or cron scheduler checks.

First, the inference script it calls. Save this as `inference.py` in your working directory:

```python
import sys
import json
import random
import time

time.sleep(0.3)

if len(sys.argv) > 1 and sys.argv[1] == "--fail":
    print("Loading model weights...", flush=True)
    print("FATAL: model checkpoint not found at specified path", file=sys.stderr)
    sys.exit(1)

labels = ["enterprise", "mid-market", "startup"]
results = []
for i in range(5):
    label = random.choice(labels)
    score = round(random.uniform(0.72, 0.99), 4)
    results.append({"id": i, "label": label, "confidence": score})

for r in results:
    print(json.dumps(r), flush=True)

summary = {"total": len(results), "status": "complete"}
print(json.dumps(summary), flush=True)
```

Now the runner script. Save this as `run_inference.sh` alongside `inference.py`, then execute `bash run_inference.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STDOUT_LOG="${LOG_DIR}/stdout_${TIMESTAMP}.log"
STDERR_LOG="${LOG_DIR}/stderr_${TIMESTAMP}.log"
SUMMARY_LOG="${LOG_DIR}/summary.log"

mkdir -p "${LOG_DIR}"

if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

echo "Python: $(which python3)"
echo "Timestamp: ${TIMESTAMP}"
echo "stdout log: ${STDOUT_LOG}"
echo "stderr log: ${STDERR_LOG}"
echo "----------------------------------------"

set +e
python3 "${SCRIPT_DIR}/inference.py" > "${STDOUT_LOG}" 2> "${STDERR_LOG}"
EXIT_CODE=$?
set -e

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "[FAIL] Inference exited with code ${EXIT_CODE} at ${TIMESTAMP}" | tee -a "${SUMMARY_LOG}"
    echo "--- stdout (last 5 lines) ---"
    tail -n 5 "${STDOUT_LOG}" 2>/dev/null || echo "(empty)"
    echo "--- stderr ---"
    cat "${STDERR_LOG}"
    exit ${EXIT_CODE}
fi

TOTAL_RECORDS=$(grep -c '"id"' "${STDOUT_LOG}" || echo "0")
echo "[OK] Inference succeeded at ${TIMESTAMP} — ${TOTAL_RECORDS} records" | tee -a "${SUMMARY_LOG}"
echo "--- output ---"
cat "${STDOUT_LOG}"

exit 0
```

Run it and observe the output. The script creates a `.venv` on first run, splits output into timestamped logs, appends a one-line summary to `summary.log`, and exits `0` on success. To test the failure path, change `python3 "${SCRIPT_DIR}/inference.py"` to `python3 "${SCRIPT_DIR}/inference.py" --fail` and rerun — you will see `[FAIL]` printed, the stderr log displayed, and a nonzero exit code returned.

The `set -euo pipefail` at the top enforces three safety properties: the script exits on any error (`-e`), treats unset variables as errors (`-u`), and fails the pipeline if any stage fails (`-o pipefail`). The `set +e` / `set -e` bracket around the Python call is deliberate — you want to capture the exit code, not abort before you can log it. This is the pattern used in production cron jobs that run nightly batch inference for lead scoring, firmographic enrichment, or intent classification across a CRM database. [CITATION NEEDED — concept: production cron jobs running AI inference for GTM enrichment]