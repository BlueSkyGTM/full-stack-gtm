# Plan-Execute Control Flow

## Hook

The LLM generates a plan, then you execute it step by step. If step 3 fails, you don't need to re-plan steps 1 and 2. This separation—plan once, execute many—is the backbone of reliable multi-step agent behavior.

## Concept

Two-phase control flow: (1) LLM outputs a structured list of tasks, (2) a runtime loop iterates through tasks, executes each one, collects results, and optionally triggers re-planning if execution reveals a dead end. Contrast with ReAct's interleaved think-act cycle—plan-execute trades adaptability for predictability and debuggability. Key mechanisms: task queue, step-level failure handling, re-plan trigger conditions.

## Demonstration

Build a minimal plan-execute loop in Python. LLM call generates a JSON plan (list of dicts with `task` and `tool` fields). Execution loop pops tasks, dispatches to tool functions, appends results to a context window. If a step fails, mark it failed and continue—or trigger re-plan if `consecutive_failures > threshold`. Print the plan, then print each step's result as it executes.

## Use It

**GTM cluster: Zone 1 — Data & Enrichment (Waterfall Pattern)**

Clay's waterfall enrichment implements this exact control flow. The plan: "try provider A, then B, then C until you get a work email." The execution: sequential API calls with early exit on success. [CITATION NEEDED — concept: Clay waterfall step-level retry configuration]. Building your own plan-execute loop outside Clay lets you handle enrichment chains across tools Clay doesn't support—custom scrapers, internal databases, or multi-hop research where step 2 depends on step 1's output.

## Ship It

- **Easy:** Extend the demo to accept a natural language goal, generate the plan, and execute three mock tool calls. Print plan and results.
- **Medium:** Add re-planning. If any step returns `{"status": "need_more_info"}`, feed execution history back to the LLM to generate a revised plan for remaining steps.
- **Hard:** Build a parallel executor. Parse the plan into a DAG (some steps depend on prior steps, some don't). Execute independent steps concurrently. Detect cycles. Print execution timeline.

## Evaluate

- Diagram question: given a sample ReAct trace and a plan-execute trace for the same goal, identify which steps could have been parallelized in plan-execute but not in ReAct.
- Debugging question: a plan-execute loop re-plans on every step even when steps succeed. Identify the bug in the termination condition.
- Design question: you're building a three-provider enrichment waterfall. The second provider is rate-limited and returns errors 30% of the time. Write the failure-handling logic (retry vs skip vs re-plan) and justify the choice.

---

**Learning Objectives:**
1. Implement a two-phase plan-execute control flow with structured task generation and sequential execution.
2. Compare plan-execute and ReAct architectures on axes of predictability, debuggability, and adaptability.
3. Detect and handle step-level failures without discarding the entire execution context.
4. Configure a re-plan trigger that activates only when execution output invalidates remaining tasks.
5. Map the plan-execute pattern onto a multi-provider enrichment waterfall with early-exit semantics.