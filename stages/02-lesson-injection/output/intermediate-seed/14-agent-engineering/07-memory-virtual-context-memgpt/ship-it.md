## Ship It

To deploy virtual context in a production GTM agent, you need three components and one discipline.

**Component 1: Core memory store.** A persistent key-value store that gets injected into every system prompt. In production this is often a JSON blob in a database row — keyed by account or session ID. The agent reads it at the start of each turn and can write to it via tool calls.

**Component 2: Archival store with search.** A vector database (Pinecone, Weaviate) or even a simple SQLite full-text search table for lower volumes. The agent inserts raw tool outputs here after extracting summaries for core memory. When it needs a detail, it calls a search function that returns ranked chunks.

**Component 3: Memory tools exposed to the model.** The agent's function-calling interface must include `core_memory_replace`, `core_memory_append`, `archival_memory_insert`, and `archival_memory_search`. These are the paging operations. The model decides when to call them — that is the design choice MemGPT makes.

**The discipline: audit what the model remembers.** Model-controlled memory means the model can forget to remember. It can fail to promote a critical fact to core memory, or it can over-promote noise. In a GTM context, this means periodically inspecting core memory state after enrichment runs. Did the agent persist the rejection reason? Did it archive the full technographic dump or lose it? If the agent consistently forgets a specific fact, add a deterministic check — after step N, verify that fact X exists in core memory before proceeding.

Here is a production-style memory audit that checks whether the right facts survived a simulated enrichment run:

```python
import json

def audit_core_memory(core_memory, required_facts):
    missing = []
    for fact_name, check_fn in required_facts.items():
        value = core_memory.get(fact_name, "")
        if not check_fn(value):
            missing.append(fact_name)
    return {
        "passed": len(missing) == 0,
        "missing": missing,
        "core_memory_snapshot": dict(core_memory),
    }

core_memory_after_run = {
    "account": "Acme Corp",
    "icp_status": "qualified",
    "technographic": "Snowflake, Segment, dbt",
    "funding": "",
    "prior_outreach": "cold Jan 2025 - no response",
    "routing": "warm outbound, reference funding trigger",
}

required = {
    "account": lambda v: len(v) > 0,
    "icp_status": lambda v: v == "qualified",
    "technographic": lambda v: "Snowflake" in v,
    "funding": lambda v: len(v) > 0,
    "prior_outreach": lambda v: "Jan" in v,
    "routing": lambda v: "warm" in v,
}

result = audit_core_memory(core_memory_after_run, required)
print(json.dumps(result, indent=2))

if not result["passed"]:
    print(f"\nAUDIT FAILED: model forgot to persist {result['missing']}")
    print("Action: add a deterministic post-step check to force these into core memory.")
else:
    print("\nAUDIT PASSED: all required facts in core memory.")
```

Run it. The audit catches that `funding` is empty — the model forgot to promote the funding detail from archival to core. This is exactly the failure mode that model-controlled memory introduces, and exactly what deterministic guardrails catch.

For cost optimization, track the ratio of core memory tokens to total context tokens. If core memory is 10% of context but archival searches return results that never get used, the agent is over-archiving. If core memory is 80% of context, the agent is under-archiving and context pressure will degrade reasoning. A healthy ratio for enrichment workflows is 20-40% core memory, 40-60% recent conversation, and the remainder for system instructions.