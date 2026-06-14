## Ship It

When you deploy a durable enrichment pipeline to production, two operational concerns dominate: partial failure handling and checkpoint storage lifecycle.

**Partial failure and the saga pattern.** When step 4 of 5 fails, steps 1–3 have already committed. In a Clay enrichment context, this means you may have written data to a CRM, created a task in a sequencer, and sent a webhook — all before the final draft step failed. The saga pattern defines a compensating action per step: if "send webhook" succeeded but "create task" failed, the compensation is "delete the webhook delivery" or "mark it as retriable." Without compensation, you have orphaned state — a CRM contact with no associated sequence, or a webhook payload that no task references. Implement compensation as a `compensate` function registered alongside each step:

```python
import sqlite3
import json
import time

DB_PATH = "/tmp/saga.db"

def init_store():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS saga_state (
            saga_id TEXT, step_name TEXT, result TEXT,
            status TEXT, completed_at REAL,
            PRIMARY KEY (saga_id, step_name)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS saga_compensations (
            saga_id TEXT, step_name TEXT, compensate_fn TEXT, committed INTEGER,
            PRIMARY KEY (saga_id, step_name)
        )
    """)
    conn.commit()
    return conn

def step_write_crm(data):
    print(f"    [EXECUTE] writing to CRM: {data['company']}")
    return {"crm_id": "crm_12345", "status": "written"}

def compensate_write_crm(data, result):
    print(f"    [COMPENSATE] deleting CRM record {result['crm_id']}")
    return {"deleted": result["crm_id"]}

def step_create_task(crm_result):
    print(f"    [EXECUTE] creating task for CRM record {crm_result['crm_id']}")
    raise RuntimeError("task API returned 503")

def compensate_create_task(data, result):
    print(f"    [COMPENSATE] (no-op: task was never created)")
    return {"deleted": None}

def step_send_webhook(task_result):
    print(f"    [EXECUTE] sending webhook for task {task_result.get('task_id', 'unknown')}")

def run_saga(conn, saga_id, company_data):
    steps = [
        ("write_crm", step_write_crm, compensate_write_crm, lambda: company_data),
        ("create_task", step_create_task, compensate_create_task, lambda: prev["result"]),
        ("send_webhook", step_send_webhook, None, lambda: prev["result"]),
    ]

    prev = {"result": None}
    completed = []

    for step_name, step_fn, comp_fn, input_fn in steps:
        cached = conn.execute(
            "SELECT result, status FROM saga_state WHERE saga_id = ? AND step_name = ?",
            (saga_id, step_name)
        ).fetchone()

        if cached and cached[1] == "committed":
            print(f"  [REPLAY] {step_name} -> {cached[0][:50]}")
            prev["result"] = json.loads(cached[0])
            completed.append((step_name, comp_fn))
            continue

        if comp_fn:
            conn.execute(
                "INSERT OR REPLACE INTO saga_compensations VALUES (?, ?, ?, 0)",
                (saga_id, step_name, comp_fn.__name__)
            )
            conn.commit()

        try:
            result = step_fn(input_fn())
            conn.execute(
                "INSERT OR REPLACE INTO saga_state VALUES (?, ?, ?, ?, ?)",
                (saga_id, step_name, json.dumps(result), "committed", time.time())
            )
            conn.commit()
            prev["result"] = result
            completed.append((step_name, comp_fn))
        except Exception as e:
            print(f"\n  [FAILED] {step_name}: {e}")
            print(f"  [SAGA] rolling back {len(completed)} committed steps...\n")
            for done_name, done_comp in reversed(completed):
                if done_comp:
                    done_comp(company_data, prev["result"])
                    conn.execute(
                        "UPDATE saga_state SET status = 'compensated' WHERE saga_id = ? AND step_name = ?",
                        (saga_id, done_name)
                    )
                    conn.commit()
                    print(f"  [COMPENSATED] {done_name}")
                else:
                    print(f"  [SKIP] {done_name} (no compensation registered)")
            return None

    print(f"\n  [SAGA COMPLETE] all steps committed")
    return prev["result"]

if __name__ == "__main__":
    conn = init_store()
    saga_id = "saga_acme_001"
    company_data = {"company": "Acme Fintech", "domain": "acmefintech.io"}

    print(f"Running saga {saga_id} for {company_data['company']}\n")
    result = run_saga(conn, saga_id, company_data)

    if result is None:
        print(f"\nSaga failed and was compensated.")
    else:
        print(f"\nSaga result: {json.dumps(result, indent=2)}")

    conn.close()
```

Output:

```
Running saga saga_acme_001 for Acme Fintech

    [EXECUTE] writing to CRM: Acme Fintech
    [EXECUTE] creating task for CRM record crm_12345

  [FAILED] create_task: task API returned 503
  [SAGA] rolling back 1 committed steps...

    [COMPENSATE] deleting CRM record crm_12345
  [COMPENSATED] write_crm

Saga failed and was compensated.
```

The CRM write was committed, but the task creation failed. The saga ran the compensating action (`compensate_write_crm`) in reverse order, deleting the CRM record. The `create_task` step's compensation was a no-op because the task was never created. This is the saga pattern: forward progress with defined rollback.

**Checkpoint storage lifecycle.** Checkpoints accumulate. A 10,000-row Clay enrichment table with 5 waterfall providers per row generates 50,000 checkpoint records. In production, set a retention policy: checkpoints older than N days are archived or deleted, and completed workflows are marked as terminal so they are not replayed on restart. SQLite handles this for development; for production scale, use a database with TTL support or a scheduled cleanup job.