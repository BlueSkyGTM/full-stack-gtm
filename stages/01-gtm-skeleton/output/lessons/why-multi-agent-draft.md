# Why Multi-Agent?

## Hook It
A single LLM call hits a capability ceiling fast: context overflow, role confusion, sequential bottlenecks. Multi-agent architectures decompose work so each agent holds a narrow scope, clear instructions, and terminates independently. This beat opens with a concrete failure mode—asking one agent to research a company, score intent, and draft outreach in a single prompt—and shows where it breaks.

## Ground It
Define three mechanisms that make multi-agent different from "just calling the API three times": **task decomposition** (split by responsibility boundary, not arbitrary chunks), **agent autonomy** (each agent owns its prompt, tools, and stop condition), and **orchestration topology** (sequential chains, parallel fan-out, or supervisor-routing). Introduce the single-agent baseline as the control case. Reference ReAct reasoning loops as the per-agent primitive, then show how orchestration sits above that primitive.

**Learning Objectives:**
1. Compare single-agent versus multi-agent execution on a task with three distinct responsibility boundaries.
2. Diagram the flow of a sequential multi-agent chain and identify where state must be passed between agents.
3. Evaluate when multi-agent adds overhead rather than value, using latency and failure-mode analysis.
4. Implement a two-agent sequential chain where Agent 1 produces structured output that Agent 2 consumes without modification.

## Show It
Live code: a two-agent pipeline in Python using direct API calls (no framework yet—mechanism before tool). Agent 1 analyzes a company's description and outputs a JSON-enforced industry classification. Agent 2 consumes that JSON and generates a positioning angle. Print each agent's raw output and the handoff payload to expose the contract between them. Show what happens when Agent 1 returns malformed output—the chain breaks at the boundary, which is the teachable moment.

**Exercise hooks:**
- Easy: Run the provided two-agent chain with three different company descriptions. Observe where the handoff contract holds or breaks.
- Medium: Add error handling at the agent boundary. If Agent 1's JSON is malformed, retry once before failing.
- Hard: Convert the sequential chain to a parallel fan-out where Agent 1 feeds two independent Agent 2 variants simultaneously, then compare their outputs.

## Use It
GTM redirect: **Zone 02 — Enrichment Waterfall**. The multi-agent sequential chain is the mechanism behind Clay's waterfall enrichment: Agent 1 checks data source A, Agent 2 fills gaps from source B, Agent 3 resolves conflicts. Each agent in the chain owns one enrichment step and passes a structured payload forward. Show a minimal enrichment waterfall using three agents that each simulate a different data source lookup on the same company record. [CITATION NEEDED — concept: Clay waterfall enrichment step architecture and state-passing mechanism]

**Exercise hooks:**
- Easy: Trace the enrichment state (a single dictionary) through three sequential agents. Print the state after each step.
- Medium: Add a "gap detection" agent that inspects the enriched record and flags which fields are still missing.
- Hard: Implement a cost-aware waterfall where Agent 2 only runs if Agent 1's confidence score is below a threshold—mimicking paid-versus-free data source prioritization.

## Ship It
Production considerations for multi-agent systems: **observability at agent boundaries** (log input/output pairs per agent, not just final result), **failure isolation** (one agent crash should not lose the work of upstream agents), and **state persistence** (if Agent 3 fails, you resume from Agent 3, not Agent 1). Introduce the pattern of writing intermediate outputs to a store (file, database, queue) rather than holding everything in memory. Mention LangGraph and CrewAI as frameworks that implement these patterns, but only after showing the manual version.

**Exercise hooks:**
- Easy: Add file-based logging to each agent in the enrichment waterfall. Each agent appends its input/output to a JSONL file.
- Medium: Implement checkpointing. After each agent completes, write the full pipeline state to disk. On restart, detect the last checkpoint and resume from there.
- Hard: Add retry logic with exponential backoff at each agent boundary. Track total latency and cost across the full pipeline. Print a summary report.

## Extend It
Where multi-agent goes next: **supervisor topologies** where a router agent inspects the task and dispatches to specialized sub-agents, **hierarchical delegation** where agents spawn child agents for subtasks, and **the boundary problem**—deciding what constitutes an "agent" versus a "function call." Preview the next lesson (Orchestration Patterns) and point to readings on MapReduce as a predecessor pattern, and on CrewAI/LangGraph as concrete implementations of the topologies introduced here.

**Exercise hooks:**
- Easy: Sketch (in comments or a text file) a supervisor topology for a four-stage GTM workflow: research → score → draft → review. Label each agent's responsibility boundary.
- Medium: Implement a minimal supervisor agent that reads a task description, classifies it into one of three categories, and routes to the correct specialized agent.
- Hard: Build a self-evaluating loop: Agent 1 produces output, Agent 2 critiques it against a rubric, and if the score is below threshold, the loop repeats with the critique appended to Agent 1's context. Cap at three iterations.