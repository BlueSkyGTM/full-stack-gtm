# Orchestration Patterns: Supervisor, Swarm, Hierarchical

---

## Hook It

You've got three agents that each do one thing well. Now you need them to work together without stepping on each other. This lesson covers the three core topologies for multi-agent coordination — supervisor (one brain, many hands), swarm (peer handoffs, no boss), and hierarchical (nested supervisors for scale).

---

## Ground It

**Supervisor pattern:** A single controller agent receives the task, delegates to specialist workers, and synthesizes their output. The supervisor holds the plan. Workers are stateless executors. Mechanism: supervisor loops over a task list, dispatches to workers by capability match, collects results, and either finishes or re-delegates.

**Swarm pattern:** No central controller. Each agent can transfer control to any other agent via a handoff function. The agent currently holding the baton decides who goes next. Mechanism: each agent's tool set includes a `transfer_to_*` function for every other agent. The LLM decides when to call it. OpenAI's Agents SDK implements this exact mechanism.

**Hierarchical pattern:** Supervisors nested inside supervisors. A top-level supervisor delegates to mid-level supervisors, each of which manages their own workers. Mechanism: recursive composition — every "worker" at level N can itself be a supervisor with its own sub-workers. This is how you scale from 3 agents to 30 without the top-level supervisor's context window becoming the bottleneck.

**Tradeoffs:**
- Supervisor: simple to reason about, but the supervisor is a single point of failure and a context bottleneck.
- Swarm: resilient and flat, but harder to predict/debug because control flow emerges from LLM decisions.
- Hierarchical: scales well, but adds latency (multiple LLM hops) and complexity in state propagation.

---

## Show It

Build all three patterns as runnable Python scripts that print their control flow. Each example uses plain functions and print statements — no framework yet — so the mechanism is visible.

**Supervisor demo:** A main loop that picks the next worker, calls it, and appends the result. Print each delegation with indentation showing depth.

**Swarm demo:** Each agent function returns either a final answer or the name of the next agent. The runner loop checks the return value and routes accordingly. Print the handoff chain.

**Hierarchical demo:** A top-level supervisor delegates to a mid-level supervisor, which delegates to workers. Print the full tree traversal.

Then show the same three patterns implemented using the OpenAI Agents SDK, which provides `Agent` classes and `Runner` methods for all three topologies out of the box. Print the control flow trace to confirm the pattern matches the manual implementation.

---

## Try It

**Easy:** Run the supervisor demo with three workers. Modify the task so the supervisor must call the same worker twice. Observe the control flow output.

**Medium:** Implement a swarm of four agents where each agent can hand off to exactly two others (not all-to-all). Run a task that requires at least three handoffs. Print the path taken.

**Hard:** Build a hierarchical system: top-level supervisor → two mid-level supervisors → two workers each (6 agents total). The task requires output from all four leaf workers, synthesized at the mid level, then merged at the top. Print the full execution tree.

---

## Use It

**GTM redirect — Autonomous SDR Agent (Zone 2 cluster: LinkedIn Automation & ABM, Cold Calling Infrastructure, multi-agent orchestration):**

Your SDR agent needs all three patterns depending on scale:

- **Supervisor:** Route a single lead through enrich → score → write email → send. One agent orchestrates the pipeline. This is your task router from the GTM hook — the supervisor decides what fires next based on lead state.

- **Swarm:** When a lead replies, the conversation agent hands off to the objection handler, which hands off to the booking agent. No supervisor needed — the agents pass the baton based on conversation state. This is how multichannel sequencing works: LinkedIn agent hands off to email agent when a connection is accepted.

- **Hierarchical:** At scale, you run 30+ agents across territories, channels, and intent signals. A top-level supervisor routes to territory supervisors, each managing channel-specific workers. This is the architecture for running 10,000 leads/month with proper suppression and human escalation.

[CITATION NEEDED — concept: specific production examples of hierarchical multi-agent SDR systems in GTM tools]

---

## Ship It

**Failure modes to handle before production:**

Supervisor: if the supervisor's context overflows, it loses the plan. Solution: summarize completed subtasks and only keep active tasks in context.

Swarm: agents can enter handoff loops (A→B→A→B forever). Solution: set a max handoff count and force-terminate with the last agent's output.

Hierarchical: state propagation between levels is fragile — a mid-level supervisor might not surface a critical detail to the top. Solution: each level returns a structured summary with required fields, not free-text.

**Monitoring:** Log every delegation, handoff, and result. In production, you need to reconstruct why Agent C was called after Agent B. Without trace logs, swarm debugging is guesswork.

**Exercise hook:** Add trace logging to your hierarchical implementation from "Try It." Each log entry includes: timestamp, calling agent, target agent, input summary, output summary. Run the full task and print the trace. Confirm you can reconstruct the decision path from logs alone.