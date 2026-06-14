## Ship It

To deploy this in a real GTM stack, you need three changes to the demo: replace SQLite with Postgres (SQLite handles one writer at a time; Postgres handles concurrent workers), add a `FOR UPDATE SKIP LOCKED` clause so multiple workers can pull from the same queue without collisions, and wrap the enrichment + CRM write in a database transaction so they succeed or fail together.

The Postgres queue query that enables concurrent workers:

```sql
BEGIN;
SELECT task_id, payload 
FROM task_queue 
WHERE status = 'pending' 
ORDER BY task_id 
FOR UPDATE SKIP LOCKED 
LIMIT 1;

UPDATE task_queue 
SET status = 'processing', updated_at = NOW() 
WHERE task_id = :task_id;
COMMIT;
```

`FOR UPDATE SKIP LOCKED` is the key. When worker A locks task 1, worker B's query skips task 1 (it is locked) and gets task 2. This lets you run N workers in parallel against the same queue without double-processing. SQS and RabbitMQ implement this at the broker level (visibility timeouts). Postgres implements it at the row level. The mechanism is the same: the consumer gets a message, the message becomes invisible to other consumers, and if the consumer crashes without acknowledging, the visibility timeout expires and the message reappears.

For LangGraph specifically, the checkpoint mechanism is already built in. LangGraph's runtime writes a checkpoint after each super-step — each node execution in the graph — keyed by `thread_id`. The checkpoint stores the full graph state: which nodes ran, what they returned, what the current values of all state variables are. If a worker crashes mid-execution, the lease on that `thread_id` expires and another worker can resume from the last checkpoint. This means your multi-agent graph gets crash recovery for free, as long as you configure the Postgres checkpointer. The configuration is roughly:

```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph

checkpointer = PostgresSaver.from_conn_string("postgresql://...")
graph_builder = StateGraph(AgentState)
graph_builder.add_node("enrich", enrich_node)
graph_builder.add_node("score", score_node)
graph_builder.add_edge("enrich", "score")
graph = graph_builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "batch_2024_01_15_lead_8201"}}
result = graph.invoke(initial_state, config=config)
```

The `thread_id` is your checkpoint key. Set it to a deterministic identifier — `batch_{date}_lead_{lead_id}` — so that if the invocation crashes and you retry with the same `thread_id`, LangGraph resumes from the last completed super-step rather than restarting the graph. This is the same idempotency pattern as the `sync_log` table in the CRM pipeline: the identifier determines whether you get a fresh run or a resume.

When do you outgrow hand-rolled Postgres + FastAPI? Bedi's argument is that you probably don't, and the evidence supports it for most GTM workloads. A Postgres-backed queue with `FOR UPDATE SKIP LOCKED` handles thousands of tasks per minute. A single Postgres instance with reasonable hardware processes enrichment batches of 50k records in under an hour. You need Temporal or a dedicated workflow engine when you have complex retry policies (exponential backoff with jitter across multiple downstream APIs), long-running human-in-the-loop steps (an SDR needs to approve a draft before sending), or cross-service transactions that span multiple databases. For straight enrichment + CRM sync, Postgres is enough. Reach for heavier infrastructure when you can articulate the specific failure mode that Postgres does not handle — not before.

The one piece you must build regardless of stack: monitoring on queue depth. If your queue grows faster than your worker drains it, leads sit unenriched for hours. A simple alert — "queue depth exceeds 1,000 pending tasks" — catches this before your SLA breaches. CloudWatch, Datadog, or a cron job that queries `SELECT COUNT(*) FROM task_queue WHERE status = 'pending'` and pages you on threshold — the mechanism does not matter, the visibility does.

[CITATION NEEDED — concept: recommended queue depth alerting thresholds for Clay enrichment runs]