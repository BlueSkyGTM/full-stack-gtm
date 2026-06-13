# Roots and Elicitation — Scoping and Mid-Flight User Input

## Beat 1: Hook

Opening scenario: you build an agent that runs straight to execution without confirming what the user actually wants. It hallucinates requirements, skips constraints, and ships the wrong thing. The fix isn't more context — it's structured elicitation at two critical points: before you start (scoping) and while you run (mid-flight).

---

## Beat 2: Concept

Two mechanisms explained:

**Scoping elicitation** — a structured pre-flight check where the agent gathers constraints, validates intent, and locks scope before any generation begins. This is a "root" pattern: it anchors all downstream behavior.

**Mid-flight elicitation** — the agent pauses execution when it hits ambiguity, missing data, or a decision point, asks the user a targeted question, and incorporates the answer before continuing.

Key distinction: scoping is proactive (ask before you act), mid-flight is reactive (ask when you're stuck). Both prevent the agent from inventing answers to questions it never asked.

---

## Beat 3: Demo

Two runnable code examples:

1. **Scoping elicitation**: A Claude API call where the system prompt forces the model to output a JSON scoping object (intent summary, constraints, missing info) before producing any deliverable. Print the JSON to terminal.

2. **Mid-flight elicitation**: A multi-turn loop where the agent generates a draft, detects uncertainty markers in its own output, surfaces a clarifying question, and re-generates with the user's answer injected. Print each turn to terminal.

Both examples use the Messages API with `claude-sonnet-4-20250514`, run in terminal via `anthropic` Python SDK, and produce observable printed output.

---

## Beat 4: Use It

GTM cluster: **ICP Research & Enrichment** ([CITATION NEEDED — concept: mapping elicitation patterns to GTM research workflows])

The scoping pattern maps directly to how enrichment tools gather firmographic data before scoring accounts. When a Clay waterfall hits a missing field — say, employee count — it doesn't guess. It either moves to the next provider or flags the gap. That's elicitation logic: identify what's missing, then resolve it before proceeding.

Mid-flight elicitation maps to async research workflows where a human reviewer approves or corrects an AI-generated account classification before it enters the CRM.

---

## Beat 5: Ship It

Three exercise tiers:

**Easy**: Modify the scoping example to add one additional required field to the JSON output and validate it's present before continuing.

**Medium**: Build a two-phase agent where Phase 1 produces a scoping document, prints it, waits for user input via `input()`, and Phase 2 generates the deliverable using that input.

**Hard**: Implement a mid-flight loop that runs a generation, scans the output for uncertainty markers ("I'm not sure", "it depends", "might be"), extracts a clarifying question, gets user input, and re-runs with the clarification appended. Loop until no uncertainty markers remain or 3 rounds elapse.

---

## Beat 6: Evaluate

Three assessment angles:

1. **Detect**: Given a transcript of agent-user interaction, identify which turns are scoping elicitation vs. mid-flight elicitation vs. neither.

2. **Compare**: Two system prompts — one with elicitation instructions, one without. Predict which produces more accurate output on an underspecified task and explain why.

3. **Design**: Given a task description with 3 known ambiguities, write a scoping prompt that surfaces all 3 before execution begins. Testable by running the prompt and checking whether the output flags each ambiguity.

---

## Learning Objectives

1. **Configure** a system prompt that forces scoping elicitation before task execution.
2. **Implement** a mid-flight elicitation loop that pauses on ambiguity and incorporates user clarification.
3. **Detect** uncertainty markers in agent output that signal the need for mid-flight user input.
4. **Compare** the error profiles of agents with and without structured elicitation.
5. **Map** elicitation patterns to GTM enrichment workflows where missing data requires resolution before proceeding.