# Shared Memory and Blackboard Patterns

## Hook

You've built three agents that each know something different about a prospect. Agent one found their tech stack. Agent two scraped their job posts. Agent three analyzed their funding. Now what? Without a coordination pattern, you have three isolated outputs. With a blackboard, you have one unified account brief that each agent contributes to and reads from. This lesson covers the two most common multi-agent state-sharing mechanisms: shared memory (unstructured, any-agent-writes) and blackboard (structured, controller-mediated).

## Concept

**Shared Memory Pattern**: A common data store that all agents read from and write to. No central controller. Agents decide for themselves when to read and what to write. Collision risk is real. Solution: atomic operations or eventually-consistent models.

**Blackboard Pattern**: A structured shared workspace plus a controller. The controller inspects the blackboard state and decides which agent to activate next. Agents are specialists that read what they need, write their contribution, and hand control back. This is the same architecture Hearsay-II used for speech recognition in 1973 — nothing new under the sun.

**Key distinction**: Shared memory is a *data structure*. Blackboard is a *control flow mechanism built on top of a shared data structure*.

## Demo

A minimal Python implementation of both patterns:

- **Shared Memory**: A dictionary-based store with threading locks. Three agents write concurrently. Show output with and without locks to demonstrate race conditions.

- **Blackboard**: A blackboard class with a controller loop. Three specialist agents (company-enrichment, intent-signal, fit-scorer) each check the blackboard, contribute their piece, and yield. Controller terminates when a completeness condition is met.

All output is terminal-printed. No browser. No external APIs.

## Use It

**GTM Redirect → Zone 1 Enrichment Waterfall**

The Clay waterfall is a blackboard pattern. The shared record is the blackboard. Each enrichment provider is a specialist agent. The waterfall controller checks which fields are null and routes to the next provider that can fill them. When you build a multi-agent enrichment pipeline — say, fetching firmographic data, then tech stack, then intent signals into one unified account record — you are implementing a blackboard. The alternative (each agent passes output to the next via message-passing) is a pipeline, not a blackboard. The difference matters when agents need to read each other's outputs to decide what to do next.

**Exercise hooks**:
- **Easy**: Build a shared dictionary that three enrichment functions write to. Print the final state.
- **Medium**: Implement a blackboard controller that runs three agents in sequence, where agent 2 reads agent 1's output to decide what to look up.
- **Hard**: Add a completeness check — the controller keeps invoking agents until all required fields are filled or all agents have been tried. Handle the case where no agent can fill a field.

## Ship It

**Production considerations for shared-state multi-agent systems**:

- Concurrency: If agents run in parallel, you need locks, queues, or CRDTs. If sequential, you don't — but you lose speed.
- Schema enforcement: Unstructured shared memory becomes a maintenance nightmare. Use typed schemas (Pydantic models, JSON Schema) to constrain what agents can write.
- Stale reads: An agent reads state, starts processing, and by the time it writes, the state has changed. Solutions: versioned writes, compare-and-swap, or just run sequentially.
- Observability: Log every blackboard write with agent ID, timestamp, and field modified. Without this, debugging a 5-agent system with inconsistent outputs is guesswork.

**Exercise hooks**:
- **Easy**: Add Pydantic validation to the shared memory store. Reject writes that don't match the schema.
- **Medium**: Instrument the blackboard with a write log. Print the full history of state changes after the controller completes.
- **Hard**: Implement compare-and-swap for a parallel-agent blackboard. Simulate a conflict where two agents try to write the same field simultaneously — one should succeed, one should retry.

## Evaluate

Assessment targets for this lesson:

1. **Compare shared memory vs. blackboard**: Given a scenario where three agents must contribute to a shared result, identify which pattern applies and why. Testable via scenario-based question.

2. **Detect a race condition**: Given the output of two agents writing to shared memory without locks, identify the corrupted state and explain the fix.

3. **Implement a controller**: Given three specialist agents and a blackboard schema, write the controller loop that terminates when the record is complete.

4. **Evaluate tradeoffs**: Given a requirement for real-time enrichment (sub-2-second latency) with five data sources, argue for parallel shared-memory with locks vs. sequential blackboard with a controller. Both are defensible — the question tests reasoning, not the "right" answer.

5. **Map to GTM**: Given a Clay waterfall enrichment config, identify which parts correspond to the blackboard (the record), the specialists (the providers), and the controller (the waterfall router).

---

**Learning Objectives (action-verb, testable)**:

1. Implement a shared memory store with concurrent writes and conflict detection in Python.
2. Implement a blackboard controller that activates specialist agents based on blackboard state.
3. Compare shared memory and blackboard patterns and identify when each is appropriate.
4. Add schema validation and write logging to a multi-agent shared state system.
5. Map a GTM enrichment waterfall to the blackboard pattern components.