# LangGraph: Stateful Graphs and Durable Execution

---

## Beat 1: Hook

**Why this, why now.** Standard LLM chains are linear and forgetful—they execute top-to-bottom and lose state after completion. Real workflows branch, loop, wait for external input, and need to survive failures. LangGraph's `StateGraph` implements a checkpointed finite-state machine where each node transition is persisted. This is the same execution model that powers multi-step GTM orchestration: enrichment waterfalls with fallbacks, multi-channel outreach sequences with waiting periods, and lead routing that branches on scoring thresholds.

---

## Beat 2: Concept

**The mechanism, stripped down.** Three primitives to cover:

1. **State schema** — A typed dictionary (typically `TypedDict` or Pydantic model) that all nodes read from and write to. This is the shared memory. Each node returns a partial update to the state, not the full state.

2. **Graph topology** — Nodes are functions. Edges are transitions. Conditional edges route based on current state. Unlike DAGs, cycles are allowed: a node can route back to itself or to an earlier node. This is how you implement retry loops and iterative refinement.

3. **Checkpointing and durability** — A `Checkpointer` (e.g., `SqliteSaver`, `PostgresSaver`) serializes state after every node execution. This enables: (a) resuming a graph after a crash or restart, (b) "time travel" to replay from any prior checkpoint, and (c) human-in-the-loop patterns where execution pauses mid-graph and resumes on external signal.

Key distinction to call out: LangGraph is not LangChain. LangChain implements chains (linear composition). LangGraph implements state machines (arbitrary topology with persistence). The dependency is historical, not architectural.

---

## Beat 3: Demo

**Running code, observable output.** Two demonstrations:

**Demo A: Stateful graph with branching and loop**
- Build a `StateGraph` with 3 nodes: `classify`, `enrich`, `retry_check`
- `classify` sets a `category` field in state
- Conditional edge routes based on `category`
- `retry_check` increments an `attempt_count` in state and loops back if under threshold
- Print state at each node transition to show mutation

**Demo B: Durable execution with checkpointing**
- Same graph, add `MemorySaver` checkpointer
- Execute to midpoint, then interrupt
- Resume from checkpoint using `thread_id`
- Print checkpoint contents to show serialized state
- Print resumed execution to show continuation from exact pause point

Both demos produce terminal output confirming state transitions and checkpoint recovery. No browser. No external APIs.

---

## Beat 4: Use It

**GTM redirect.** The LangGraph `StateGraph` with conditional edges is the execution pattern behind multi-step enrichment waterfalls. In Clay terms: a row triggers a waterfall that tries data provider A, checks if the result is sufficient, and if not, falls through to provider B. The `StateGraph` with checkpointing is what makes this reliable at scale—if the Clay runtime restarts mid-waterfall, the row resumes from the last completed provider check, not from zero.

Specific mapping: This is the mechanism for **Zone 2 — Enrichment waterfall with fallback logic** from the GTM topic map. [CITATION NEEDED — concept: enrichment waterfall fallback pattern in Clay]

If the enrichment-waterfall connection is too forced, the redirect is: "foundational for Zone 2 enrichment orchestration."

**Exercise hooks:**
- *Easy:* Modify the conditional edge thresholds in Demo A to change routing behavior. Observe different paths through the graph.
- *Medium:* Add a new node that represents a second enrichment source. Implement fallback logic: if `enrich` returns empty, route to `enrich_backup` instead of exiting.
- *Hard:* Implement a checkpointer-backed graph that pauses after `classify`, simulates an external approval (manual state injection), and resumes. Print state before and after the pause.

---

## Beat 5: Ship It

**Production constraints.** Three things to address:

1. **Storage backend** — `MemorySaver` is in-memory and ephemeral. Production requires `PostgresSaver` or equivalent. State schema changes between deployments require migration or versioning strategy. Undocumented: how LangGraph handles schema backwards compatibility when you add fields to state mid-deployment. Test this before shipping.

2. **Thread management** — Each concurrent workflow instance needs a unique `thread_id`. In GTM context, this is typically one thread per prospect/row. At 10K+ concurrent threads, checkpoint storage becomes a bottleneck. Monitor checkpoint write latency.

3. **Cost of durability** — Every node transition writes a checkpoint. For a 5-node graph processing 50K rows, that's 250K checkpoint writes. Calculate storage and I/O cost before deploying. Consider whether every transition needs persistence or only critical ones (LangGraph does not currently support selective checkpointing—this is a known limitation).

**Exercise hooks:**
- *Easy:* Swap `MemorySaver` for `SqliteSaver` in Demo B. Confirm checkpoints persist to disk across separate Python process invocations.
- *Medium:* Write a loop that spawns 100 graph instances with unique `thread_id`s. Profile total checkpoint storage size and write time.
- *Hard:* Deliberately change the state schema between two runs (add a required field). Observe and document the failure mode. Propose a mitigation.

---

## Beat 6: Review

**What to retention-check.**

Core mechanisms to verify:
- The difference between a chain (linear, stateless between runs) and a stateful graph (arbitrary topology, persistent state)
- How state mutation works: nodes return partial updates, not full replacement
- What checkpointing enables: resume, replay, human-in-the-loop
- The production cost of durable execution: storage, I/O, schema migration

**Assessment hooks (not full quiz text, but testable claims):**
- Given a node's return value, predict the resulting state (tests partial-update mechanism)
- Given a graph topology with a cycle, predict exit condition behavior (tests conditional edge logic)
- Given a checkpoint at step N of M, describe what happens on resume (tests durability mental model)
- Compare LangGraph `StateGraph` to a standard LangChain `SequentialChain`: which properties hold for each (branching, looping, persistence, resumability)?

---

## Learning Objectives

1. **Build** a `StateGraph` with typed state schema, multiple nodes, and conditional edges that produces observable state transitions in terminal output.
2. **Implement** checkpointed execution using `MemorySaver` or `SqliteSaver`, and resume a paused graph from a stored checkpoint.
3. **Explain** the difference between LangChain's linear chain composition and LangGraph's stateful finite-state machine model.
4. **Configure** a graph with a cycle (retry loop) that exits based on a state-derived condition.
5. **Evaluate** the storage and I/O cost implications of durable execution at GTM scale (10K+ concurrent threads).