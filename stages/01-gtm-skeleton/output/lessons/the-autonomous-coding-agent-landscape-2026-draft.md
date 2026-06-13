# The Autonomous Coding Agent Landscape (2026)

## Hook It

You've been told an agent is "just an LLM with tools." That's wrong, and the difference is why some coding agents complete multi-hour refactors while others loop forever on a typo. This lesson maps the architectural decisions that separate agents that ship code from agents that burn tokens.

## Map It

Survey the agent architecture space: single-turn completion → retrieval-augmented chat → tool-calling loops → plan-then-execute → multi-agent delegation. For each tier, identify what changes in the control flow, what failure modes appear, and what engineering constraints emerge. Covers Claude Code, Cursor Agent Mode, Aider, Devin, Codex CLI, and open-source frameworks (OpenHands, SWE-Agent). Each positioned by its loop architecture, not its marketing page.

**Exercise hooks:**
- Easy: Classify five coding tools by agent tier from their documented behavior
- Medium: Trace the control flow of a tool-calling loop and identify where infinite loops can occur
- Hard: Compare two agents' sandboxing strategies and document which failure modes each prevents

## Build It

Implement a minimal coding agent loop from scratch: prompt → LLM call → tool execution → observation → next LLM call. Add a file-read tool, a file-write tool, and a bash-run tool. Run the agent against a concrete task (fix a failing test in a provided repo). Instrument token usage, iteration count, and success/failure. This is the mechanism every commercial coding agent wraps.

**Exercise hooks:**
- Easy: Add logging to each step of the agent loop and print the full trace
- Medium: Add a retry-with-different-strategy mechanism when the agent loops on the same error twice
- Hard: Implement a simple planner that generates a step list before execution and evaluates success against the plan

## Use It

Coding agents are infrastructure for building GTM tooling. The mechanism: define a specification for a data pipeline, enrichment script, or integration, then delegate implementation to an agent with sandboxed execution. Maps to **Zone 02 — Enrichment** and **Zone 04 — Orchestration** in the GTM topic map. Specifically: using a coding agent to generate Clay webhook payloads, build Apollo API wrappers, or construct research scraping pipelines from a natural-language spec.

**Exercise hooks:**
- Easy: Write a spec for a lead enrichment script and run a coding agent against it
- Medium: Build a validation gate that checks agent-generated code against a GTM schema before execution
- Hard: Chain an enrichment agent output into a Clay waterfall input format and verify the data shape matches

## Ship It

Deploying coding agents in production means handling: context window exhaustion during long tasks, cost controls for token-heavy loops, permission boundaries for file system and network access, and evaluation frameworks for code quality. Covers guardrail patterns: approval gates, diff-review hooks, token budgets, and rollback mechanisms. Addresses the "who is responsible when the agent ships a bug" question with concrete accountability patterns.

**Exercise hooks:**
- Easy: Add a token budget to your agent loop that halts execution with a summary
- Medium: Implement a diff-review gate that shows proposed changes and requires approval before write
- Hard: Build an evaluation harness that runs your agent against five tasks and scores success, token cost, and iteration count

## Extend It

Multi-agent coding systems: when one agent plans, another executes, and a third reviews. The delegation problem, the context-sharing problem, the orchestration problem. Pointer to research on SWE-bench leaderboards and what the top-scoring architectures do differently. Also: where agents still fail consistently — multi-file refactors with implicit dependencies, novel API usage without documentation, and tasks requiring taste judgments.

**Exercise hooks:**
- Easy: Read one SWE-bench leaderboard submission and map its architecture to the agent tiers from Map It
- Medium: Implement a two-agent system where one writes code and another reviews it, with a revision loop
- Hard: Design an evaluation benchmark for coding agents specific to GTM pipeline tasks and justify the scoring rubric