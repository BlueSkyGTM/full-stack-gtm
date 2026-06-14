## Ship It

The production deployment shape is straightforward: Mem0 as a Docker container (self-hosted) or the managed API, backed by Postgres for the KV + vector store (via pgvector), and either Neo4j or Mem0's bundled graph store for relationships. The ingestion hook wires into whatever agent or workflow emits text — a Clay webhook, a Slack listener, a call transcript pipeline. The retrieval hook wires into your agent's prompt construction: before each LLM call, run `client.search()` and inject the top-k memories as context.

Latency budgeting is the critical operational concern. The fan-out query hits three stores in parallel, so wall-clock latency is dominated by the slowest store. Graph traversal on Neo4j for a 2-hop query against a moderately sized graph (100K nodes) runs in single-digit milliseconds. Vector similarity on pgvector with an HNSW index is comparable. The KV lookup is negligible. Realistic end-to-end search latency is 20-80ms for a warm cache, 100-200ms cold. The extraction step on ingestion is 500-2000ms because it requires an LLM call — that is acceptable for async ingestion but unacceptable for synchronous request paths.

The cache strategy for hot entities is entity-level memoization. If "Acme Corp" appears in 40% of queries for a given user, cache the graph subgraph for that entity and invalidate on write. This matters because enrichment workflows repeatedly query the same company, person, or deal across multiple sequence steps.

```python
import time
import statistics

try:
    from mem0 import MemoryClient
except ImportError:
    print("pip install mem0ai to run this benchmark")
    exit(1)

import os
if not os.environ.get("MEM0_API_KEY"):
    print("export MEM0_API_KEY=your_key")
    exit(1)

client = MemoryClient()

batch = [
    {"text": f"Prospect {i} from company {i % 20} discussed budget concerns in the last call.", "user_id": f"user_{i % 5}"}
    for i in range(50)
]
batch.extend([
    {"text": f"Company {i % 20} uses tool_{i % 10} as their primary platform.", "user_id": f"user_{i % 5}"}
    for i in range(50)
])

t0 = time.time()
results = client.add(batch, batch_size=25)
batch_ingest_ms = (time.time() - t0) * 1000
print(f"[BATCH INGEST] {len(batch)} memories in {batch_ingest_ms:.0f}ms")
print(f"               avg per memory: {batch_ingest_ms / len(batch):.0f}ms")

benchmark_queries = [
    ("budget concerns", "user_0"),
    ("company 3", "user_2"),
    ("primary platform", "user_4"),
    ("what tools does company 10 use", "user_1"),
    ("budget", "user_3"),
]

latencies = []
for q, uid in benchmark_queries:
    t0 = time.time()
    res = client.search(query=q, user_id=uid)
    ms = (time.time() - t0) * 1000
    latencies.append(ms)
    print(f"[SEARCH] q='{q}' uid={uid} | {len(res)} hits | {ms:.0f}ms")

print(f"\n[LATENCY REPORT]")
print(f"  min:    {min(latencies):.0f}ms")
print(f"  median: {statistics.median(latencies):.0f}ms")
print(f"  p95:    {sorted(latencies)[int(len(latencies) * 0.95)]:.0f}ms")
print(f"  max:    {max(latencies):.0f}ms")
```

This benchmark ingests 100 memories in two batched calls (batch_size=25), then runs five queries that exercise different stores. The latency report at the end tells you whether your deployment is within the budget for synchronous agent calls. If the median exceeds 100ms, investigate the graph store index — Neo4j without proper indexing on node labels degrades sharply at scale. If the p95 is 3x the median, you likely have a cold-cache problem and should add entity-level caching for hot prospects.

The cost optimization rule from Zone 14 applies directly: every enrichment credit and every extraction LLM call is a token cost. Batch your ingestion, cache your hot entities, and measure latency per query class — not just an aggregate average. A p95 of 200ms on graph-heavy queries while vector queries run at 20ms tells you to optimize the graph index, not the overall system.