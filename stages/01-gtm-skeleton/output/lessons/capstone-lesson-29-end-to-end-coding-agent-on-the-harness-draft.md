# Capstone Lesson 29: End-to-End Coding Agent on the Harness

## Hook

You've built the harness (Lesson 27). You've wired tool calls (Lesson 28). Now you chain them into a single autonomous loop: the agent receives a task, writes code, executes it on the harness, reads the result, and self-corrects until the task is done or the budget is exhausted. This is the capstone — every mechanism from the last four lessons in one runnable system.

## Concept

Describe the plan → generate → execute → observe → repair loop as a finite state machine with three terminal conditions: pass, fail-max-retries, fail-budget. Map each state to the components already built (harness sandbox, tool schema, LLM call). Explain the retry budget, the observation parser (stdout/stderr/exit code), and the prompt construction that feeds prior attempt context back into the next generation call. Mechanism first: the agent is a while loop with a termination condition, not magic.

## Live Code

Build the complete `CodingAgent` class in a single file. It takes a task string, a max_retries integer, and a harness instance. Each iteration: (1) construct a messages array with system prompt, task, and prior attempts; (2) call the LLM with tool use enabled; (3) extract the code block from the tool call; (4) execute on the harness; (5) parse the result; (6) either return the output or append the failure to context and loop. Print every iteration's attempt number, exit code, and output so the practitioner can watch the agent self-correct in real time.

## Use It

This is the architecture behind Clay's enrichment waterfall and automated research agents — an LLM loop that generates structured actions, executes them, and adapts based on results. The specific GTM cluster is **Zone 2: Signal Detection & Enrichment**. [CITATION NEEDED — concept: Clay waterfall agent loop internals]. A coding agent that writes, runs, and fixes its own data-transformation scripts is the same mechanism that powers automated lead research and enrichment pipelines at scale.

## Drill

- **Easy:** Run the agent on a task that passes on the first attempt (e.g., "write a function that returns 42"). Observe the single-iteration path.
- **Medium:** Give the agent a task with a deliberate off-by-one error in the prompt (e.g., "generate the first 10 primes" but the correct answer is 11 primes including 2). Watch it self-correct within 3 retries.
- **Hard:** Add a token budget counter. Track cumulative prompt + completion tokens across retries. Terminate early if the budget exceeds a threshold. Print the total token spend at exit.

## Ship It

The practitioner extends the agent with a persistent artifact store: each successful code write is saved to a file named by task hash. They then run the agent against a battery of 5 tasks of increasing difficulty (string manipulation, file I/O, API parsing, data transformation, algorithm implementation) and produce a run report: task name, retries used, token spend, pass/fail. This report is the ship artifact — evidence the agent works end-to-end.

---

**GTM Redirect Rules for this lesson:**
- Redirect target: Zone 2 (Signal Detection & Enrichment)
- Specific framing: "this is the autonomous loop pattern behind enrichment waterfalls"
- Foundational fallback: if the Clay waterfall internals cannot be cited, the redirect becomes "foundational for Zone 2 — the same generate-execute-observe loop that enrichment pipelines require"