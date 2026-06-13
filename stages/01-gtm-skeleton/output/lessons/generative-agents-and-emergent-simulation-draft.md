# Lesson Title: Generative Agents and Emergent Simulation

---

## Beat 1: Hook

A 2023 Stanford simulation placed 25 LLM-powered agents in a virtual town. Within two simulated days, agents organized a Valentine's Day party—with invitations, planning, and social coordination that no code explicitly programmed. This is emergent behavior from four primitives: observe, retrieve, reflect, plan. This lesson dissects those primitives.

---

## Beat 2: Concept — The Architecture

Four mechanisms, in execution order:

1. **Memory Stream** — An append-only log of observations, each timestamped and scored for importance (0–10). This is the agent's raw experience buffer.

2. **Retrieval** — Three scoring functions multiplied together: *recency* (exponential decay by time), *importance* (the score assigned at encoding), *relevance* (cosine similarity between query embedding and memory embedding). Top-K memories are surfaced.

3. **Reflection** — Periodically, the agent synthesizes recent memories into higher-order abstractions ("I enjoy talking to Maria about research"). Reflections are stored as memories themselves, making them retrievable for future reasoning. Triggered when the cumulative importance of recent memories exceeds a threshold.

4. **Planning** — A hierarchical day plan (hour → sub-actions) stored in memory. Plans are re-planned when new observations contradict expectations. Plans are also retrievable memories.

The loop: observe environment → retrieve relevant memories + reflections + plans → decide action → append observation to memory stream → periodically reflect → revise plan.

No single component is complex. Emergence happens when multiple agents run this loop in a shared environment and observe each other's actions.

[CITATION NEEDED — concept: exact recency decay constant and importance scoring prompt from the Generative Agents paper, Park et al. 2023]

---

## Beat 3: Code — Minimal Generative Agent

A single-agent implementation of all four primitives using only stdlib + an LLM call. The agent maintains a memory stream, retrieves by recency × importance × relevance, generates reflections when cumulative importance exceeds a threshold, and maintains/revises a plan.

**Exercise hooks:**

- **Easy:** Run the provided agent for 20 simulated ticks. Print the memory stream after each reflection trigger. Count how many reflections were generated.
- **Medium:** Add a second agent that shares the environment. Both agents observe each other's actions. Log the first instance of Agent B retrieving a memory about Agent A's action.
- **Hard:** Remove the reflection component entirely. Run the same 20-tick simulation. Compare task completion accuracy with and without reflection. Print the delta.

---

## Beat 4: Use It — GTM Redirect

This maps to **Zone 3 (Research)** and the **ICP Simulation** cluster.

The mechanism is direct: generative agents simulate buying committee members. Each agent is seeded with a persona (VP Engineering, security concerns, budget authority), given a memory stream of market observations, and placed in a shared simulation. You observe:

- Which messages propagate between personas via agent-agent observation
- Which objections surface during reflection
- How plans (purchase evaluations) revise when new information enters the environment

This is not a persona interview. This is a Monte Carlo simulation of social information diffusion through a buying committee. The GTM application: test messaging strategies against 100 simulated committee runs before spending pipeline dollars.

Foundational for Zone 3 account research and Zone 5 messaging strategy. The memory-retrieval-reflection architecture also underpins personalized outbound systems that maintain state across touchpoints.

---

## Beat 5: Ship It — Production Exercise

**Exercise hook:** Build a 5-agent buying committee simulation. Each agent represents a different stakeholder (economic buyer, technical evaluator, champion, blocker, end user). Seed each with persona-specific memories. Inject a product message into one agent's observation stream. Run 50 simulated interactions. Measure: did the message reach the economic buyer? Did the blocker generate negative reflections? Output a propagation graph and a JSON log of every agent's plan revisions.

---

## Beat 6: Evaluate

Quiz questions grounded in the four mechanisms:

1. Given a memory with importance=3, recency score=0.5, relevance score=0.8, compute the retrieval score. If another memory has importance=9, recency=0.1, relevance=0.6, which is retrieved first?

2. A reflection is triggered when _________ exceeds a threshold. (Fill in: cumulative importance of recent unreflected memories.)

3. Why is a reflection stored in the memory stream rather than acted on immediately? (Answer: reflections must be retrievable for future retrieval-weighted reasoning; they are compressed abstractions, not actions.)

4. You observe emergent coordination between two agents but neither was explicitly programmed to coordinate. Name the mechanism that enables this. (Answer: shared environment observations entering each agent's memory stream via the observe step.)

**Exercise hook:** Write a 3-sentence explanation of why removing reflection would degrade multi-step planning but not single-turn retrieval. Submit as a PR comment on the lesson repo.