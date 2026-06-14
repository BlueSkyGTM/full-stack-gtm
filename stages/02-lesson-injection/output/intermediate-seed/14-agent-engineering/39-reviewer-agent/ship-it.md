## Ship It

Production wiring adds three things to the builder-marker pair: logging, score persistence, and a fallback path for borderline outputs. Logging means every builder-marker round is written to a JSONL file so you can audit decisions later. Score persistence means the scores are stored alongside the artifacts — you need this for A/B testing builder strategies and for tracking marker drift over time. The fallback path handles the gray zone: scores that are above the reject threshold but below the approve threshold.

```python
import time

LOG_FILE = "builder_marker_log.jsonl"
APPROVE_THRESHOLD = 6
REJECT_THRESHOLD = 4
GRAY_ZONE = (REJECT_THRESHOLD, APPROVE_THRESHOLD)

def log_round(task, artifact, score, decision, attempt):
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task": task[:200],
        "artifact": artifact,
        "score": score,
        "decision": decision,
        "attempt": attempt,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def production_gate(task, max_retries=2):
    for attempt in range(max_retries + 1):
        artifact = builder(task)
        score = marker(artifact)
        total = score["total"]

        if total >= APPROVE_THRESHOLD:
            decision = "approved"
        elif total < REJECT_THRESHOLD:
            if attempt < max_retries:
                decision = "retry"
                task = f"Previous attempt scored {total}/8. Notes: {score['notes']}. Fix these issues. Task: {task}"
            else:
                decision = "rejected"
        else:
            decision = "human_review"

        log_round(task, artifact, score, decision, attempt)

        if decision != "retry":
            break

    return {
        "artifact": artifact,
        "score": score,
        "decision": decision,
        "attempts": attempt + 1,
    }

tasks = [
    "Write to Sarah Chen, VP Eng at Datadog, about on-call alert fatigue reduction.",
    "Write to Marcus Webb, Head of RevOps at Gong, about pipeline forecasting accuracy.",
    "Write to Priya Patel, CTO at Vercel, about edge function performance monitoring.",
]

results = [production_gate(task) for task in tasks]

print(f"{'Task':<50} {'Score':>6} {'Decision':<15} {'Attempts':>8}")
print("-" * 85)
for task, result in zip(tasks, results):
    short_task = task[:48] + ".."
    print(f"{short_task:<50} {result['score']['total']:>4}/8 {result['decision']:<15} {result['attempts']:>8}")

print(f"\nFull round-trip logs written to {LOG_FILE}")
```

The gray zone routing matters because the marker is not infallible. A score of 5 out of 8 is ambiguous — it might be a genuinely mediocre output, or it might be a good output that the marker scored conservatively. Routing these to human review is cheaper than sending bad emails and safer than auto-approving. In a production outbound system running on secondary domains, the human review queue becomes your calibration mechanism: a human spot-checks gray-zone outputs, and their decisions inform whether you need to adjust the threshold or retrain the rubric.