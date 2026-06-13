# ReWOO and Plan-and-Execute: Decoupled Planning

---

## Hook: When ReAct Gets Stuck in the Loop

ReAct interleaves thinking and acting, which means every tool call feeds back into the next reasoning step. That round-trip is expensive and brittle — one bad observation derails the entire chain. ReWOO and Plan-and-Solve both answer the same question: what if the planner never sees the observations until the end?

---

## Concept: Two Mechanisms for Decoupling

**Plan-and-Execute** — A planner LLM generates an ordered list of tasks. An executor LLM runs each task sequentially, passing results forward. A replanner can revise remaining steps after each execution. LangGraph ships a reference implementation of this pattern.

**ReWOO (Reasoning Without Observation)** — A planner LLM generates a fixed plan with placeholder slots (e.g., `#E1`, `#E2`). Workers fill each slot independently. A solver LLM receives the original question plus all filled slots and produces the final answer. The planner is called exactly once; it never sees tool outputs during planning. Published by Xu et al. (2023) [CITATION NEEDED — concept: ReWOO paper title and arxiv ID].

Key structural difference: Plan-and-Execute can replan mid-execution. ReWOO commits to the full plan upfront. This makes ReWOO cheaper (one planner call) but less adaptive to failed tool calls.

---

## Use It: Enrichment Waterfalls as Decoupled Plans

GTM cluster: **Enrichment Waterfalls** (Zone 1 — Intelligence)

A Clay waterfall is a Plan-and-Execute instance. The plan: "get email from provider A, if null try provider B, if null try provider C." The execution: run each step, check the output, decide whether to continue. The replanner is the waterfall's conditional logic — it decides which provider to hit next based on whether the previous one returned data.

ReWOO maps to a different GTM pattern: multi-source account research where all lookups are independent. Plan: "get firmographic data from Clearbit, get tech stack from BuiltWith, get funding from Crunchbase." All three execute in parallel. Solver: combine into an account brief. No replanning needed because the lookups don't depend on each other.

---

## Build It: Implement Both Patterns Side by Side

Construct a Plan-and-Execute agent using LangGraph's `plan-and-execute` pattern. Then construct a ReWOO agent that generates a plan with slot variables, fills them with tool calls, and solves. Run the same research task through both and compare: total LLM calls, token usage, handling of a tool that returns null.

- **Easy:** Implement Plan-and-Execute with a two-tool research task. Print the plan, each execution step, and the replan decisions.
- **Medium:** Implement ReWOO with three independent tool slots. Inject a null return from one tool and observe how the solver handles the missing slot.
- **Hard:** Run the same five-step enrichment task through both patterns. Measure and compare: total token count, wall-clock time, and final answer accuracy when one tool fails. Print a structured comparison table.

---

## Ship It: When to Commit vs. When to Replan

Production decisions for decoupled planning:

- **Use Plan-and-Execute** when steps have dependencies (waterfall logic: "if this fails, try that") or when tool failure rates are high and replanning is necessary.
- **Use ReWOO** when steps are independent (parallel enrichment lookups), when you want to minimize LLM calls, or when you need deterministic plan structure for auditing.
- **Token cost tradeoff:** ReWOO calls the planner once. Plan-and-Execute calls the planner once plus a replanner after every execution step. For a 5-step plan, that's 1 planner call vs. 1 + 5 replanner calls.
- **Failure mode:** ReWOO cannot recover from a bad plan. If the planner chooses the wrong tool, the entire chain runs anyway. Plan-and-Execute can replan around failures but costs more.
- **Observability:** ReWOO's plan is inspectable before execution begins — you can validate or reject the plan before spending on tool calls. Plan-and-Execute reveals the plan incrementally.

---

## Review It

Assessment targets:

1. **Compare** ReWOO and Plan-and-Execute on replanning capability and planner call count.
2. **Identify** which pattern maps to an enrichment waterfall and why (dependency between steps).
3. **Predict** failure behavior: given a scenario where tool 2 of 5 fails, describe what each pattern does next.
4. **Calculate** approximate token costs for both patterns given a fixed step count and replanner token overhead.
5. **Evaluate** which pattern to use for a specific GTM workflow (parallel lookups vs. conditional waterfall).