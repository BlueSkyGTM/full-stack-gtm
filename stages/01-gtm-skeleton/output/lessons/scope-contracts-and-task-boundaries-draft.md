# Scope Contracts and Task Boundaries

## Beat 1: Hook

A scope contract is a declarative specification of what a task handler will and will not do — expressed in a format the model can parse and your pipeline can enforce. Without one, every edge case becomes a runtime surprise. Describe a real failure mode: a lead-scoring agent that started writing outreach emails because the task boundary was implicit, not explicit.

## Beat 2: Concept

Three-part mechanism. **Scope contract schema**: a structured declaration with `IN_SCOPE`, `OUT_OF_SCOPE`, and `ESCALATION` fields. **Boundary enforcement**: guard clauses that check task input against the contract before execution. **Handoff protocol**: when a task falls outside scope, the contract specifies where it routes instead of silently failing or improvising. Contrast with "just write better prompts" — a contract is testable and auditable, a vibe check is not.

## Beat 3: Demo

Build a minimal scope contract as a Python dictionary. Write a function that accepts a task description, checks it against `IN_SCOPE` and `OUT_OF_SCOPE` keyword sets, returns a decision (`EXECUTE`, `REJECT`, `ESCALATE`), and prints the routing decision with a reason. All output observable in terminal.

**Exercise hook (Easy)**: Add a new task type to `IN_SCOPE` and verify it routes to `EXECUTE`.

## Beat 4: Use It

GTM redirect: **Zone 02 — ICP & Account Intelligence**. When you run an agent that enriches accounts, the scope contract determines whether it should answer "what does this company do?" (in scope) vs. "should we acquire them?" (out of scope — that's a strategic decision, not an enrichment task). This is the boundary between data retrieval and decision-making. Clay's waterfall enriches fields; it does not evaluate strategic fit. That line must be explicit, not assumed. [CITATION NEEDED — concept: Clay scope boundaries in enrichment workflows]

**Exercise hook (Medium)**: Write a scope contract for an ICP scoring agent. Define three tasks that are in scope, three that are out of scope, and one escalation path. Test all seven against your enforcement function.

## Beat 5: Ship It

Implement a multi-agent routing system where each agent has a scope contract. An orchestrator receives a task, checks it against each agent's contract, routes to the correct handler or escalates. Print the full routing trace. This is the pattern used in production agent architectures where "who handles what" must be deterministic, not emergent.

GTM redirect: This is foundational for Zone 05 (Outbound & Inbound Execution) where multiple agents — research, copy, delivery scheduling — must respect boundaries or you get agents stepping on each other's work.

**Exercise hook (Hard)**: Add a contract version field. When the orchestrator loads agents, it logs which contract version it used for routing. Introduce a contract update mid-script and show the routing decision changing for the same input.

## Beat 6: Evaluate

Three assessment angles: (1) Given a scope contract and five task descriptions, classify each as `EXECUTE`, `REJECT`, or `ESCALATE` — automated grader checks against answer key. (2) Given an agent that oversteps its scope (provided as code), identify the missing boundary and write the contract clause that prevents it. (3) Given two agents with overlapping scope contracts, find the ambiguity that causes double-routing and fix it.

No quiz bank without corresponding `docs/en.md`. Objectives for this lesson:

1. Write a scope contract with `IN_SCOPE`, `OUT_OF_SCOPE`, and `ESCALATION` fields for a given agent role.
2. Implement a boundary enforcement function that routes tasks to `EXECUTE`, `REJECT`, or `ESCALATE` with printed reasons.
3. Diagnose scope violations in multi-agent systems by tracing routing decisions.
4. Compare implicit task boundaries (prompt-only) with explicit contracts (structured + testable) and identify failure modes of each.