# Parallel / Swarm / Networked Architectures

---

## Beat 1: Hook

A single LLM call is a single-threaded worker. The moment you need to research 200 accounts, enrich 50 contacts, and draft personalized outreach simultaneously, serial execution collapses. Parallel and swarm architectures distribute cognitive work across multiple agents that run concurrently, share state, and coordinate toward a goal. This is the difference between a script and a system.

---

## Beat 2: Concept

**Parallel execution** runs independent tasks simultaneously—each agent has no awareness of the others. Fan-out, process, fan-in.

**Swarm architecture** dispatches work to heterogeneous agents that hand off control to each other based on capability. OpenAI's Swarm implements this pattern: a router agent evaluates input, transfers execution to a specialist agent, which may transfer again. No centralized orchestrator; control is passed like a baton.

**Networked architectures** connect agents through shared channels (message queues, event streams, shared memory). Each agent subscribes to events, produces outputs, and reacts to others' outputs. This is actor-model computation applied to LLM agents.

Key mechanisms:
- **Fan-out / fan-in**: Distribute N subtasks, collect N results, synthesize. Map-reduce for agents.
- **Handoff protocol**: Agent A wraps its context into a transfer message, Agent B receives it and continues. Swarm uses this.
- **Pub/sub event bus**: Agents emit events, other agents react. Decouples producers from consumers.
- **Shared scratchpad**: Multiple agents read/write to a common state store. Requires conflict resolution.
- **Convergence**: How parallel branches reconcile contradictions. Last-write-wins, voting, or a designated "judge" agent.

Tradeoff matrix: Parallel is simplest but agents can't coordinate. Swarm enables coordination via handoffs but is sequential across handoffs. Networked is most flexible but requires coordination overhead—deadlocks, race conditions, and divergent state are real failure modes.

---

## Beat 3: Demo

Build three progressively complex architectures using the OpenAI Agents SDK (or raw async Python with an LLM client):

1. **Parallel fan-out/fan-in**: Research 5 companies simultaneously. Each sub-agent returns a structured summary. A synthesis agent merges findings into a single report.

2. **Swarm with handoffs**: A triage agent classifies an inbound support request, hands off to a billing agent or technical agent. The specialist resolves or escalates back.

3. **Networked pipeline**: A research agent emits findings to a queue. A drafting agent consumes findings and produces copy. A review agent consumes copy and approves or rejects (rejection loops back to drafting).

All examples print observable output confirming coordination worked.

```python
import asyncio
import os
import json
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

async def research_company(name: str) -> dict:
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a company research agent. Return JSON with keys: name, industry, one_paragraph_summary."},
            {"role": "user", "content": f"Research this company: {name}"}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

async def synthesize(results: list[dict]) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Synthesize multiple company research summaries into a brief comparative report."},
            {"role": "user", "content": json.dumps(results)}
        ]
    )
    return response.choices[0].message.content

async def main():
    companies = ["Acme Corp", "TechFlow", "DataBridge"]
    
    print("=== Fan-out: Researching companies in parallel ===")
    results = await asyncio.gather(*[research_company(c) for c in companies])
    for r in results:
        print(f"  {r['name']}: {r['industry']}")
    
    print("\n=== Fan-in: Synthesizing ===")
    report = await synthesize(results)
    print(report[:500])

asyncio.run(main())
```

---

## Beat 4: Use It

**GTM Cluster: Zone 02 — Enrichment Waterfalls**

The enrichment waterfall pattern is a parallel architecture. When Clay (or any enrichment tool) runs multiple data providers against a contact, those lookups execute concurrently. The waterfall collects results, applies priority rules, and populates the record.

More advanced: multi-agent account research. A parallel architecture fans out one research agent per target account, each enriching firmographic, technographic, and intent data. A convergence agent normalizes results into a consistent format for scoring.

The swarm pattern maps to multi-step outreach orchestration: a triage agent routes inbound leads to a research specialist, which hands off to a copy-drafting specialist, which hands off to a sequencing specialist.

**Exercise hooks:**
- *Easy*: Modify the fan-out demo to research 10 accounts in parallel and print per-account latency.
- *Medium*: Build a two-agent swarm where a "classifier" agent reads a lead description and hands off to either an "SMB specialist" or "Enterprise specialist" agent, each producing tailored outreach.
- *Hard*: Build a networked enrichment pipeline: one agent scrapes a domain, one agent enriches from a data provider mock, one agent scores the combined result. Wire them through an asyncio Queue with retry logic.

---

## Beat 5: Ship It

Production considerations for multi-agent systems:

- **Rate limiting**: N parallel agents means N concurrent API calls. Implement token bucket rate limiting across the agent pool, not per-agent.
- **Error isolation**: One agent failure should not collapse the swarm. Wrap each agent in a circuit breaker. Log failures but continue.
- **Observability**: Every agent handoff and every parallel branch needs trace IDs. Use structured logging with `trace_id`, `agent_id`, `step` fields.
- **Cost accounting**: Parallel execution multiplies token spend linearly with agent count. Budget per-branch and total.
- **Idempotency**: Networked agents may receive duplicate events. Design handlers to be idempotent on `event_id`.
- **State management**: For swarm handoffs, serialize full agent context (conversation history + scratchpad) into the transfer payload. For networked agents, use an external state store (Redis, SQLite) rather than in-memory dicts—agents may run in separate processes.

```python
import asyncio
import uuid
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s trace=%(trace_id)s agent=%(agent_id)s %(message)s")

class Agent:
    def __init__(self, name: str, trace_id: str):
        self.name = name
        self.trace_id = trace_id
        self.logger = logging.LoggerAdapter(logging.getLogger(), {"trace_id": trace_id, "agent_id": name})

    async def run(self, task: str) -> dict:
        start = time.time()
        self.logger.info(f"Starting task: {task}")
        await asyncio.sleep(0.5)  # simulate LLM call
        result = {"agent": self.name, "task": task, "status": "complete"}
        self.logger.info(f"Completed in {time.time() - start:.2f}s")
        return result

class Orchestrator:
    def __init__(self, max_concurrency: int = 3):
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.results = []

    async def dispatch(self, agent: Agent, task: str):
        async with self.semaphore:
            try:
                result = await agent.run(task)
                self.results.append(result)
            except Exception as e:
                logging.getLogger().error(f"Agent {agent.name} failed: {e}")

    async def fan_out(self, tasks: list[tuple[str, str]], trace_id: str):
        agents = [Agent(name, trace_id) for name, _ in tasks]
        coros = [self.dispatch(agent, task) for agent, (_, task) in zip(agents, tasks)]
        await asyncio.gather(*coros)
        print(f"\n=== Results ({len(self.results)}/{len(tasks)} succeeded) ===")
        for r in self.results:
            print(f"  {r['agent']}: {r['status']}")

async def main():
    trace_id = str(uuid.uuid4())[:8]
    orch = Orchestrator(max_concurrency=2)
    tasks = [("researcher", "enrich Acme Corp"), ("researcher", "enrich TechFlow"), ("researcher", "enrich DataBridge")]
    await orch.fan_out(tasks, trace_id)

asyncio.run(main())
```

---

## Beat 6: Evaluate

**Exercise hooks:**
- *Easy*: Explain in one paragraph why a parallel architecture cannot guarantee consistent shared state without a coordination mechanism. Provide a concrete failure scenario.
- *Medium*: Given a swarm with three agents (triage → research → draft), trace the handoff sequence for an input that gets escalated twice. How many LLM calls occur? What context does the final agent receive?
- *Hard*: Design a networked architecture for real-time competitive intelligence monitoring. Specify: what events each agent emits, what it subscribes to, how you handle a burst of 500 events in 10 seconds, and how you prevent duplicate processing.

**Self-check questions:**
1. When does parallel execution produce worse results than sequential? (Answer: when later steps depend on earlier outputs—dependencies force serialization.)
2. What is the failure mode specific to swarm handoffs that does not exist in simple parallel? (Answer: infinite handoff loops—agent A transfers to B, B transfers back to A.)
3. Why does a networked architecture need idempotent handlers more than a parallel one? (Answer: pub/sub delivery can produce duplicate events; parallel workers receive distinct inputs by design.)