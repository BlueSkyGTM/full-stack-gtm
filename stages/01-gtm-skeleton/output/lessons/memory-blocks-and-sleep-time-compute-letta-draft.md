# Memory Blocks and Sleep-Time Compute (Letta)

## Hook
Agents that forget everything between turns are useless for sustained workflows. This lesson covers the mechanism Letta uses to give agents durable, editable memory and the compute pattern that lets them consolidate that memory during idle cycles rather than in-line during inference.

## Concept
Two mechanisms, one design principle. First: **memory blocks** — named, editable segments of text that live inside the agent's context window (core memory) or in a searchable vector store (archival memory). The agent can `core_memory_append`, `core_memory_replace`, `archival_memory_insert`, and `archival_memory_search` via tool calls it issues to itself. This is the operating-system memory hierarchy pattern (registers → RAM → disk) applied to LLM context management. Second: **sleep-time compute** — the agent runs background jobs during idle periods to reorganize, summarize, and migrate information between memory tiers. Instead of burning context tokens on bookkeeping during a user conversation, the agent defers that work to a separate compute pass. The agent effectively "dreams" — replaying recent interactions, extracting entities, updating its persona block, and archiving stale context.

## Use It
**GTM Redirect: Zone 2 — Enrichment, persistent account intelligence.** In a multi-touch outbound workflow, an agent researching an account accumulates fragments across sessions: 10-K filings, LinkedIn activity, technographic signals, past email replies. Memory blocks are how you store that accumulated intelligence in a queryable form. Sleep-time compute is how you keep that intelligence from rotting — the agent re-summarizes and re-indexes during off-hours. This is the mechanism behind "account intelligence that improves with every touch" rather than a blank-slate enrichment run each time. [CITATION NEEDED — concept: Letta memory blocks applied to Clay enrichment workflows]

## Build It
**Exercise hooks only:**

- **Easy**: Spawn a local Letta agent, insert two facts into core memory via tool calls, and print the resulting memory block state to confirm the edits landed.
- **Medium**: Implement a two-tier memory flow — load conversation excerpts into archival memory, run a search query, and demonstrate retrieval of a previously archived fact.
- **Hard**: Configure a sleep-time compute job that scans recent messages, extracts named entities, and writes them into a designated core memory block. Print before/after state of the block to show the consolidation happened.

## Ship It
Production considerations: memory block sizing (core memory burns context tokens; overstuff it and you degrade response quality), archival recall latency (embedding search adds round-trip time), sleep-time job scheduling (how often, what triggers it, what happens if the job fails mid-consolidation), and the fundamental tension that the agent is editing its own prompt — which means it can corrupt its own state. Mitigation: checkpoint memory state before each sleep-time pass. GTM redirect: this is the infrastructure that keeps an account intelligence agent from going stale across a 14-touch outbound cadence.

## Extend It
- Compare Letta's memory tiering to LangGraph's checkpointing and persistent state mechanism — what problems does each solve, and where do they overlap?
- Read the original MemGPT paper and trace how the memory management prompt (the "inner monologue" where the agent decides to swap memory in/out) evolved into the current tool-call interface.
- Investigate the failure mode: what happens when an agent's core memory fills with contradictory facts? Design a reconciliation strategy.
- [CITATION NEEDED — concept: sleep-time compute failure modes and recovery patterns in production Letta deployments]