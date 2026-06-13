# Lesson: Handoffs and Routines — Stateless Orchestration

---

## Beat 1: Hook

When one agent finishes its work and another picks up, something has to carry the context. If that something is a shared database or a session object, you've coupled the agents together. Stateless orchestration is the pattern where the message itself carries everything the next agent needs — no shared memory, no side channels, no assumptions about what happened before. This is how Clay's enrichment waterfall works: each enrichment step receives the same row payload, checks what's already populated, and either fills the gap or passes it along. The row *is* the state.

---

## Beat 2: Concept

Define three terms precisely:

**Handoff** — the moment one agent or routine transfers execution to another. The transfer carries a payload. The receiving agent does not know or care who called it.

**Routine** — a named, repeatable sequence of handoffs. Routines are declarative: "run A, then B, then C, exit on first success." They are not scripts with branching logic embedded in code.

**Stateless** — the orchestrator does not maintain memory between steps. Each step receives its inputs, produces outputs, and terminates. If step 3 needs something from step 1, it must be in the payload, not in a session store.

Contrast with stateful orchestration (e.g., a LangGraph graph with a persistent `State` object that accumulates across nodes). The tradeoff: stateless is easier to retry, parallelize, and debug; stateful is easier when you need cumulative reasoning.

---

## Beat 3: Mechanism

The mechanism is a **payload-passing loop with exit conditions**.

1. **Define the payload schema.** This is typically a JSON object with typed fields. Some fields are required inputs; some are outputs that start as `null`.
2. **Define the routine as an ordered list of agents or functions.** Each agent has: an input contract, an output contract, and an exit condition (e.g., "skip if `email` is already populated").
3. **The orchestrator iterates the list.** For each agent: check the exit condition. If not met, call the agent with the payload. Merge the agent's output back into the payload. Move to the next agent.
4. **After the list is exhausted, return the final payload.**

This is exactly the Clay enrichment waterfall pattern: provider A is called for email, if it returns null, provider B is called, and so on. The orchestrator doesn't care what the providers do internally — it only manages the payload and the exit conditions.

Code example: a minimal Python implementation of a stateless routine runner that accepts a list of agents (functions), a payload (dict), and runs them in sequence, printing the payload state at each step.

---

## Beat 4: Use It

**GTM Cluster: Enrichment Waterfalls (Zone 1 — Research & Enrichment)**

The Clay waterfall *is* a stateless routine. When you configure a Clay enrichment column to "find email" by trying ZoomInfo → Hunter → Snov.io → Apollo in order, you are defining a handoff routine. Each provider receives the same row payload (`first_name`, `last_name`, `domain`), checks whether `email` is already populated (exit condition), and if not, attempts to populate it. The orchestrator does not remember which provider succeeded — it only passes the payload forward.

The generalization: any GTM workflow where you need to try multiple data sources, tools, or agents in sequence with fallback logic is a stateless routine. Examples:
- Enrichment waterfalls (email, phone, technographic data)
- Multi-channel outreach sequences (email → LinkedIn → phone, exit on reply)
- Account research routines (scrape website → check Crunchbase → summarize with LLM)

If you can express your GTM workflow as "try A, then B, then C, exit on condition," you can model it as a stateless routine.

---

## Beat 5: Ship It

Build a working stateless orchestrator that runs a three-step enrichment routine on a list of mock prospects. The routine: (1) check if email is present, (2) check if company is present, (3) generate a summary using an LLM call. Each step is a function that receives the full payload and returns a modified payload. The orchestrator prints the payload state after each step.

Deliverable: a single Python script that runs unmodified, processes 3 mock prospects, and prints the before/after payload for each.

---

## Beat 6: Drill It

**Easy:** Add a fourth step to the routine that checks for phone number and populates it from a mock lookup. Observe how the orchestrator doesn't need modification — only the step list changes.

**Medium:** Implement a branching routine where step 2 depends on the output of step 1. If `company` is `"enterprise"`, run the enterprise enrichment path; if `"smb"`, run the SMB path. This tests whether statelessness holds under conditional routing.

**Hard:** Implement retry logic within the stateless constraint. If a step fails, retry it up to 2 times with exponential backoff, but do not persist any retry state between attempts — each retry receives the same input payload. Log all attempts and final outcomes.

---

## Learning Objectives

1. **Implement** a payload-passing orchestrator that runs a sequence of agents with exit conditions and no shared state.
2. **Detect** when a workflow requires stateful orchestration (cumulative reasoning, multi-turn conversation) versus stateless (fallback chains, enrichment waterfalls).
3. **Configure** a multi-step GTM enrichment routine modeled as a stateless handoff sequence.
4. **Compare** stateless orchestration to stateful graph-based orchestration (e.g., LangGraph) in terms of retry behavior, debuggability, and coupling.
5. **Evaluate** whether a given GTM workflow can be expressed as a stateless routine or requires persistent state.

---

## GTM Redirect Rules for This Lesson

- In **Use It**: "This is the Clay waterfall pattern — the enrichment waterfall is a stateless routine where each provider receives the same row payload and the orchestrator exits on first success."
- In **Ship It**: "The routine you build maps directly to how Clay configures multi-provider enrichment columns: ordered providers, shared payload, exit conditions."
- If a student asks about workflows that *don't* fit this pattern (e.g., multi-turn sales conversations with memory), redirect to: "That requires stateful orchestration, which is a different pattern. This lesson covers stateless routines — foundational for Zone 1 enrichment workflows."