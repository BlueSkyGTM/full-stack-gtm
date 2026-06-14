## Ship It

To ship a stateless enrichment routine into a production GTM stack, you need three things beyond the core loop: error handling per agent, observability of the payload at each step, and a way to persist the final payload.

First, error handling. In the runner above, if an agent raises an exception, the whole routine dies. In production, you want per-agent error isolation: if ZoomInfo times out, log it and move to Hunter. The payload should record that ZoomInfo failed (so you can retry it later), but the routine should continue. This is a one-line change in the runner — wrap `agent["fn"](payload)` in a try/except and store the error in the payload.

Second, observability. The `print` statements in the example are the right idea but the wrong implementation. In production, log the payload delta at each step: which agent ran, what fields it added or changed, how long it took. This gives you a per-row audit trail. When a prospect asks "where did you get my email?", you can answer precisely: "Hunter, on 2025-01-15, at step 2 of the enrichment routine."

Third, persistence. The final payload should be written to your data store — Clay's table, a Postgres row, a spreadsheet. Because the payload is a flat dict (or can be flattened to one), this is typically a single INSERT or UPDATE. No session teardown, no state serialization, no graph checkpoint to restore.

Here is the runner upgraded for production: per-agent error isolation, timing, and a structured log per step.

```python
import json
import time
from datetime import datetime, timezone

def production_runner(agents, payload, row_id="unknown"):
    log = []
    for agent in agents:
        if agent["should_run"](payload):
            start = time.monotonic()
            try:
                output = agent["fn"](payload)
                if output:
                    payload.update(output)
                elapsed = round(time.monotonic() - start, 3)
                entry = {
                    "row_id": row_id,
                    "agent": agent["name"],
                    "status": "ran",
                    "elapsed_s": elapsed,
                    "fields_written": list(output.keys()) if output else [],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as e:
                elapsed = round(time.monotonic() - start, 3)
                payload.setdefault("_errors", []).append({
                    "agent": agent["name"],
                    "error": str(e),
                })
                entry = {
                    "row_id": row_id,
                    "agent": agent["name"],
                    "status": "error",
                    "elapsed_s": elapsed,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            log.append(entry)
            print(json.dumps(entry))
        else:
            entry = {
                "row_id": row_id,
                "agent": agent["name"],
                "status": "skipped",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            log.append(entry)
            print(json.dumps(entry))
    return payload, log

def flaky_provider(row):
    if row.get("_call_count", 0) < 1:
        row["_call_count"] = row.get("_call_count", 0) + 1
        raise TimeoutError("provider timed out")
    return {"email": "recovered@example.com"}

def solid_provider(row):
    return {"email": "solid@example.com"}

production_agents = [
    {"name": "flaky_provider", "fn": flaky_provider, "should_run": lambda r: r.get("email") is None},
    {"name": "solid_provider", "fn": solid_provider, "should_run": lambda r: r.get("email") is None},
]

payload = {"company_domain": "flaky.com", "email": None}
result, audit_log = production_runner(production_agents, payload, row_id="row-001")
print(f"\nFinal payload: {json.dumps(result, indent=2)}")
print(f"\nAudit log entries: {len(audit_log)}")
```

```
{"row_id": "row-001", "agent": "flaky_provider", "status": "error", "elapsed_s": 0.0, "error": "provider timed out", "timestamp": "2025-01-15T12:00:00.000000+00:00"}
{"row_id": "row-001", "agent": "solid_provider", "status": "ran", "elapsed_s": 0.0, "fields_written": ["email"], "timestamp": "2025-01-15T12:00:00.001000+00:00"}

Final payload: {
  "company_domain": "flaky.com",
  "email": "solid@example.com",
  "_errors": [{"agent": "flaky_provider", "error": "provider timed out"}],
  "_call_count": 1
}

Audit log entries: 2
```

The flaky provider errored. The runner caught it, logged it, stored the error in the payload, and moved to the next agent. The solid provider succeeded. The final payload has the email, the error history, and a complete audit trail. This is what a production enrichment waterfall looks like — and it is structurally identical to the 7-line runner from Build It, plus error isolation and logging.

In a Clay workflow, you get most of this for free: Clay's waterfall column handles provider fallback, and the row records which provider returned each value. But understanding the underlying pattern lets you reproduce it outside Clay — in a Python script, in a serverless function, in whatever runtime your stack requires — without depending on a specific tool's UI.