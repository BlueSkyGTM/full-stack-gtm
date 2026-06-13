# The Shift from Chatbots to Long-Horizon Agents

## Beat 1: Hook — "Ask vs. Task"

The request-response ceiling. A chatbot answers a question; an agent completes a job. Describe the failure mode: when a practitioner asks a chatbot to "research this account and draft a personalized email," the single-turn context window fills, the model hallucinates or stops early, and the practitioner ends up copying/pasting between turns. This is the threshold where chatbot architecture breaks and agent architecture becomes necessary.

## Beat 2: Concept — The Planning-Execution Loop

Explain the mechanism that separates chatbots from agents. A chatbot follows: `user_input → model → output`. An agent follows: `goal → plan → step → observe → replan → next_step → ... → done`. Define the three components: (1) a planner that decomposes a goal into ordered subtasks, (2) an executor that runs one subtask and returns an observation, (3) a state manager that tracks completed/failed/pending steps. Name the pattern: a planning-execution loop with memory. Reference: ReAct (Yao et al., 2022) as one canonical implementation. [CITATION NEEDED — concept: planning-execution loop origin and alternatives].

## Beat 3: Code — Chatbot vs. Agent in 40 Lines

Build a minimal working comparison. First, a single-prompt chatbot that tries to do account research in one call — show it truncate or hallucinate. Second, a 3-step agent loop that: (1) searches for company info, (2) searches for contact info, (3) synthesizes both into a summary. The agent loop uses a hardcoded plan (`steps = ["company_research", "contact_research", "synthesis"]`) and a state dictionary. Print the plan, each step's observation, and the final output to terminal. Uses `anthropic` SDK with tool-use or structured output. All code runs without modification and prints observable state transitions.

## Beat 4: Use It — Enrichment Waterfalls as Agent Loops

GTM redirect: this is Zone 02 (Enrich). An enrichment waterfall is a long-horizon agent with a fixed plan: try data source A, if empty try B, if empty try C, then synthesize. Map the planning-execution loop directly to the Clay waterfall pattern — each enrichment step is an executor subtask, the "fall through" logic is the replanner, and the contact record is the state manager. Concrete example: enriching a lead record with company size, tech stack, and funding stage. The chatbot approach would paste the lead's LinkedIn into a prompt and hope; the agent approach runs each enrichment as a discrete step, validates the output, and backfills only what's missing.

## Beat 5: Ship It — Observability for Long-Running Agents

Production agents fail in ways chatbots don't: they loop infinitely, they get stuck on one step, they silently drop state. Ship a wrapper that logs every planning-execution cycle: step attempted, tool called, observation returned, tokens consumed, and whether the plan changed. Print a structured JSON log for each cycle. This is the minimum observability needed before an agent touches a production CRM. Mention: this logging pattern is what agent frameworks (LangGraph, CrewAI) implement internally, but the mechanism is the same — instrument the loop, not the model.

## Beat 6: Check It — Exercises

- **Easy**: Modify the 3-step agent to add a fourth step ("competitor_analysis") and print the updated plan.
- **Medium**: Add a simple "replan" branch — if a step returns empty data, skip to the next step instead of stopping. Print the replan decision.
- **Hard**: Replace the hardcoded plan with an LLM-generated plan. Give the model the goal and available tools, ask it to output an ordered list of steps, then execute that list. Compare output quality to the hardcoded version.

---

## Learning Objectives (3–5, action verbs only)

1. **Compare** the architectural difference between a request-response chatbot and a planning-execution agent loop.
2. **Implement** a minimal agent loop that decomposes a goal into ordered subtasks and executes them sequentially.
3. **Map** the planning-execution pattern to a GTM enrichment waterfall, identifying planner, executor, and state manager components.
4. **Add observability logging** to an agent loop that records each step's input, output, and plan modifications.
5. **Evaluate** when a task exceeds chatbot architecture and requires an agent approach, based on step count, state requirements, and error-recovery needs.

## GTM Redirect Rules

- **Use It** redirects to: Zone 02 (Enrich) — enrichment waterfalls as fixed-plan agent loops.
- **Ship It** is foundational for all Zones — any multi-step GTM workflow (research, outreach, qualification) requires agent observability.