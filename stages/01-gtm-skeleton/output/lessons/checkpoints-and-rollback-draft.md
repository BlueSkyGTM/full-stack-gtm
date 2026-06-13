# Checkpoints and Rollback

## Hook (Why this matters)

Long-running processes fail. Training jobs die mid-epoch. Agent workflows crash after 47 API calls. Without checkpoints, you start from zero. With checkpoints, you resume from the last known good state. This lesson covers the mechanism of state capture and recovery—the difference between a 10-minute retry and a 10-hour rerun.

## Concept (The mechanism)

**State snapshotting as a recovery primitive.** A checkpoint is a serialized snapshot of process state (model weights, optimizer momentum, step counter, random seed) written to durable storage at defined intervals. Rollback is the inverse: deserialize the last valid checkpoint and resume execution from that point. The core tradeoff is checkpoint frequency vs. I/O overhead—every checkpoint costs disk write time, but missing a checkpoint costs the compute since the last save. We cover full checkpoints, incremental checkpoints, and the coordinator pattern for distributed state.

## Code (Working examples)

**Three progressive scripts, all runnable in terminal:**

1. **Basic checkpoint/save-resume pattern.** A mock training loop that saves `state.json` every N steps. Kill it mid-run. Re-run it. Observe it resumes from the last checkpoint instead of step 0.

2. **Checkpoint validation with rollback.** A loop that writes checkpoints, then simulates corruption (bad loss spike). The recovery script detects the anomalous checkpoint, rolls back to the previous one, and reports which checkpoint was rejected.

3. **Incremental checkpointing with size comparison.** Two strategies run side-by-side on the same workload: full state serialization vs. diff-based incremental. Script prints checkpoint size and write time for each, demonstrating the storage/compute tradeoff.

All scripts use only `json`, `os`, `time`, and `pathlib` from the standard library. No external dependencies.

## Use It (GTM connection)

**GTM Cluster: Data Enrichment Pipelines (Zone 2 — Data Infrastructure)**

Enrichment pipelines that call 3+ data providers per record (Clay waterfall pattern: Clearbit → Apollo → Hunter → manual) face the same failure/recovery problem as training loops. If the pipeline dies at provider 2 of 4 on record 847 of 2,000, you need to know where you stopped and what you already fetched. The checkpoint pattern here is: write `(record_id, provider_completed, enriched_data)` to persistent storage after each provider call. On restart, the pipeline reads the checkpoint log and skips completed work. Without this, you burn API credits re-fetching data you already have, or worse, create duplicate records downstream.

**Exercise hooks:**
- *Easy:* Write a checkpoint function that saves the last completed record index to a file. Modify a provided mock enrichment loop to call it after each record.
- *Medium:* Add validation logic that detects partial writes (checkpoint exists but is malformed JSON) and rolls back to the previous checkpoint.
- *Hard:* Implement a checkpoint strategy that tracks per-provider completion status for each record. Given a mock pipeline that crashes randomly, demonstrate recovery that resumes from the exact provider call that failed, not from the beginning of the record.

## Ship It (Production considerations)

**What breaks at scale and how to handle it:**

- **Checkpoint storage contention.** Writing checkpoints to the same NFS path from 8 concurrent workers creates write contention. Solution: per-worker checkpoint files with a manifest, or a checkpoint database (SQLite works for single-node, DynamoDB for distributed).
- **Stale checkpoints.** Code changes between the checkpoint write and the resume. The model architecture changed but the checkpoint has the old weight shapes. Solution: checkpoint metadata must include a schema version or code hash. On load, compare against current code. Mismatch = reject checkpoint.
- **Checkpoint cleanup.** Keeping every checkpoint fills disk. Implement a retention policy: keep last N, or keep best (lowest loss), or keep one per epoch. The script demonstrates a cleanup function that retains by policy.
- **Atomic writes.** A crash mid-checkpoint-write produces a corrupt file. Use write-to-temp-then-rename (`os.replace`) for atomicity. The code examples demonstrate this pattern.

**Exercise hooks:**
- *Easy:* Add a retention policy to the basic checkpoint script that deletes all but the last 3 checkpoints.
- *Medium:* Implement schema version checking. Modify the training loop's state structure, then demonstrate that loading an old checkpoint fails gracefully with a version mismatch error, not a cryptic KeyError.
- *Hard:* Build a checkpoint coordinator for 4 simulated workers. Each worker writes to its own checkpoint file. The coordinator reads all 4 and reports the global state: which workers completed, which failed, and what the recovery plan is.

## Quiz (Assessment)

**Questions target the mechanism, not trivia. All questions grounded in the code examples and concept section:**

- Q1: Given a training loop that saves a checkpoint every 100 steps, if the job crashes at step 347, what is the latest recoverable step? (Tests understanding of checkpoint interval vs. recovery point.)
- Q2: Two checkpoint strategies are available: full save (200ms write time, 50MB) every 100 steps, or incremental save (20ms write time, 5MB) every 10 steps. For a 10,000-step job that crashes on average once per run, which strategy minimizes total wall-clock time including recompute? Show the math. (Tests the frequency vs. overhead tradeoff.)
- Q3: A checkpoint file exists but is empty (0 bytes). Describe what happened and how atomic writes prevent this. (Tests the atomic write-to-temp-then-rename pattern.)
- Q4: Your enrichment pipeline checkpoints after each provider call. The pipeline crashes mid-write during the Apollo step for record 412. On restart, the checkpoint says record 412 completed Clearbit only. Explain why you do NOT re-call Clearbit for record 412. (Tests the GTM application of per-step checkpointing.)

---

**Learning Objectives:**
1. Implement a checkpoint-save and checkpoint-load cycle for a stateful loop.
2. Detect corrupted or stale checkpoints and trigger rollback to the last valid state.
3. Compare full vs. incremental checkpoint strategies given a cost budget and failure rate.
4. Configure checkpoint frequency based on the tradeoff between I/O overhead and recompute cost.
5. Explain how atomic writes prevent partial-checkpoint corruption.