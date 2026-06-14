## Ship It

To productionize checkpoint-rollback in a GTM enrichment pipeline, three decisions need to be made explicitly.

**Checkpoint storage location.** A local JSON file works for development and single-process pipelines. For production with multiple workers or serverless functions, checkpoints need to live in shared durable storage—a Postgres table, a Redis hash, or an S3 object with a key like `checkpoints/{pipeline_run_id}/{record_id}.json`. The storage choice determines your recovery granularity: file-per-record lets you retry individual records; a single log file requires sequential replay.

**Checkpoint retention policy.** Every checkpoint consumes storage. For a 2,000-record enrichment run checkpointing after each of 4 provider calls, that is 8,000 checkpoint entries. Define a retention window: keep checkpoints for 7 days post-completion, then delete. This matters for GDPR compliance—stale checkpoint files containing prospect emails are a data retention liability even if the production database has been cleaned.

**Recovery testing.** The checkpoint mechanism is useless if you never test recovery. The practice here is deliberate chaos injection: kill the enrichment pipeline mid-run in staging, verify it resumes from the correct checkpoint, verify no provider is double-called, verify no duplicate records appear downstream. This is the enrichment-pipeline equivalent of a chaos engineering practice. LangGraph's checkpoint-to-Postgres pattern makes this testable because the checkpoint state is queryable—you can inspect it with SQL before resuming, confirming the state matches expectations.

The atomic write + validation + backward-scan rollback pattern from the Build It scripts handles 90% of enrichment pipeline failure modes. The remaining 10%—network partitions, provider rate limits that persist across restarts, schema changes between checkpoint versions—require application-level logic that no generic checkpoint framework provides.