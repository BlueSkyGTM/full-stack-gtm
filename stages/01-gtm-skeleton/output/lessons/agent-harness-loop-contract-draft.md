# Agent Harness Loop Contract

## Define It
The agent harness loop contract specifies the schema and control flow guarantees between the orchestrator (harness) and the agent's iterate-decide-act cycle. Covers input/output shape, tool registration protocol, termination conditions, and state accumulation semantics.

## Ground It
Build a minimal harness that runs a 3-step loop against a mock LLM responder — prints state at each iteration to show the contract boundaries (input → decision → tool call → result → next input). Observable output confirms the loop respected its termination clause.

## Map It
Position the harness loop contract relative to: (1) single-shot LLM calls, (2) ReAct-style reasoning loops, (3) DAG-based orchestration. Compare to OpenAI Agents SDK tool loop, LangGraph `StateGraph` cycle nodes, and Anthropic's tool-use loop pattern. Identify which guarantees each framework enforces vs. leaves to the developer.

## Use It
GTM redirect: **Zone 2 — Enrichment Waterfall**. The Clay enrichment waterfall is an agent harness loop: iterate over data providers, invoke, evaluate confidence, continue or terminate. The loop contract determines when enrichment stops (field filled, max providers exhausted, confidence threshold met). Exercise: configure a mock enrichment harness that loops through three providers and terminates on first successful fill — print the contract state at each step.

## Ship It
Implement a full agent harness loop contract as a reusable Python class with: typed input/output schema (Pydantic models), tool registry, max-iteration guard, and a termination predicate callback. Run it against a mock "company research" agent that calls 2–3 tools and halts when it accumulates enough data. All output printed to terminal — no external dependencies beyond `pydantic` and `anthropic` (or mock). Medium exercise: add error-recovery retry logic inside the loop. Hard exercise: add a human-in-the-loop confirmation gate before tool calls that mutate external state.

## Extend It
Examine failure modes when the contract is violated: infinite loops from missing termination guards, state drift when harness mutates shared context, tool schema mismatches that surface only at runtime. Discuss formal verification of loop contracts (preconditions, postconditions, invariants) and how property-based testing can catch contract breaks. Note: comprehensive formal verification tooling for agent loops is nascent — [CITATION NEEDED — concept: formal verification of agent loop contracts].