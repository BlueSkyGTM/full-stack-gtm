# Claude Agent SDK: Subagents and Session Store

## Hook
You've hit the wall with single-agent architectures. One prompt tries to research, score, and draft — and the context window turns into soup. Subagents are the fix: spawn specialists, collect their output, and stitch it together. Session store is how those specialists remember what happened.

## Concept
The subagent pattern: a parent agent delegates work to child agents with scoped instructions, receives structured output, and aggregates. Session store provides a key-value persistence layer that survives across agent turns. Mechanism: parent calls `Agent.run()` or orchestrates via tool calls; children write to and read from a shared `SessionStore` object. State is namespaced per session, not per agent — which means coordination and collision are both possible.

## Walkthrough
Build a minimal orchestrator that spawns three subagents in sequence: one fetches data, one scores it, one formats output. Each subagent reads from and writes to session store. Show how session keys persist between agent calls and how to inspect the store. Demonstrate what happens when two agents write to the same key (last-write-wins behavior). Print the full session state at the end to confirm persistence.

## Use It
This is the architecture behind multi-step enrichment pipelines. Map to **Zone 01 — Enrichment** and **Zone 02 — Scoring** in the GTM topic map: one subagent pulls firmographic data, another scores ICP fit, a third drafts a research summary. Session store holds the accumulated account profile so each agent builds on prior work instead of starting from scratch. This is how you build the Clay waterfall pattern without Clay — orchestrated agents with shared state.

## Ship It
Implement a production-grade orchestrator with error handling for subagent failures, session store TTL management, and structured output validation. Exercise hooks:
- **Easy**: Modify the walkthrough example to add a fourth subagent that writes a log entry to session store
- **Medium**: Add retry logic — if a subagent returns malformed output, re-invoke it with the error message and the session context
- **Hard**: Build a parallel execution pattern where two subagents run simultaneously on different session namespaces, then a reducer agent merges results

## Extend It
Subagent orchestration at scale: token budget allocation across agents, session store size limits, and the tradeoff between sequential (dependable context) vs. parallel (faster, stale read risk). Compare to LangGraph's state graph and CrewAI's memory system — same pattern, different primitives. [CITATION NEEDED — concept: subagent token budget management in Claude Agent SDK]. Point to multi-agent debugging: how to trace which agent wrote which session key and when.