# Action Budgets, Iteration Caps, and Cost Governors

## Hook

Introduces the failure mode: an agent loop that won't stop. A ReAct chain that calls a tool, gets a bad result, retries with a slightly different prompt, loops again, and 47 API calls later has burned $12 in tokens with no useful output. This is the problem that budget mechanisms solve.

## Concept

Defines the three control surfaces. An **action budget** is a hard ceiling on discrete tool calls or operations. An **iteration cap** limits how many loop cycles an agent may complete before being forced to yield. A **cost governor** tracks cumulative spend (tokens, dollars, latency) and terminates the process when a threshold is breached. These are independent dials that compose together.

## Mechanism

Explains each control as an algorithm, then shows implementation patterns. Action budgets decrement a counter on each tool invocation. Iteration caps wrap the agent loop in a bounded `for` or `while` with a guard condition. Cost governors accumulate token usage from API responses and compare against a float threshold. Covers composition: how to layer all three so that hitting *any* ceiling triggers graceful shutdown. Addresses the subtlety of partial results — what to return when you're cut off mid-reasoning. Code examples use pure Python with no framework dependencies.

## Use It

Builds a minimal agent loop with all three controls wired in. The agent performs a multi-step research task (e.g., "find 3 competitors for a given company") and the controls prevent runaway behavior. Exercise hooks: (Easy) Modify the action budget to allow more tool calls and observe behavior. (Medium) Add a latency-based governor that terminates if a single step exceeds a time threshold. (Hard) Implement a "soft landing" that lets the current tool call finish but prevents new iterations when the cost governor triggers.

## Ship It

Demonstrates how these controls translate to a production agent endpoint. Shows a FastAPI route that accepts a task, enforces per-request cost caps, logs budget consumption, and returns either a completed result or a `budget_exhausted` payload with partial output. Exercise hooks: (Easy) Add a response header that reports remaining budget. (Medium) Write a middleware that enforces a global hourly cost cap across all requests. (Hard) Implement budget borrowing: allow a request to exceed its per-request cap if the global hourly budget has surplus capacity.

## Evaluate

Presents scenarios where each control type is the correct primary constraint and the practitioner must choose and justify. Covers observability: what to log, how to detect when agents are hitting caps too early (budgets too tight) vs. too late (budgets too loose). Discusses the tradeoff between cost control and agent quality — aggressive caps produce cheaper but less capable agents. Points to the GTM application: in Clay waterfalls and multi-step enrichment workflows, iteration caps prevent cascading API waste when a single enrichment step fails repeatedly. Exercise hooks: (Easy) Given a log of budget exhaustion events, identify the failing step. (Medium) Tune iteration caps for a specific agent task to minimize cost while maintaining a quality threshold. (Hard) Design a budget allocation strategy for a multi-tenant system where different customers have different spend tiers.

---

**GTM Redirect:** In outbound enrichment workflows (Zone 1, enrichment waterfall), iteration caps prevent a single stale data source from burning through your entire Apollo/PeopleDataLabs API allocation. Cost governors enforce per-account or per-campaign spend ceilings so a malformed query doesn't cascade into budget overruns across your sequence.