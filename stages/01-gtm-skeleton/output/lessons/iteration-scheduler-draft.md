# Iteration Scheduler

## Hook

You've built an agent loop that runs "until done." At 2 AM it decides it's never done. You now have a scheduling problem: how many iterations, how fast, and who decides when to stop.

## Concept

Define the iteration scheduler as a control structure with three parameters: max iterations, delay strategy, and convergence predicate. Contrast with naive `while True` loops. Show the failure modes: infinite loops, API rate limits, cost overruns, diminishing returns on LLM refinement.

## Mechanism

Explain the three levers:
1. **Fixed-bound scheduling** — hard cap on iterations, no exceptions.
2. **Backoff scheduling** — increasing delay between iterations (exponential, jittered).
3. **Convergence scheduling** — stop when delta between outputs falls below threshold or a sentinel value is detected.

Show how each trades off cost, latency, and result quality. Implement a minimal `IterationScheduler` class in Python that accepts all three parameters and yields iteration state (attempt number, elapsed time, last result delta). Include a convergence check using string similarity or numeric delta. Print the full iteration trace.

## Use It

GTM redirect: **Clay's waterfall enrichment implements fixed-bound scheduling with fallback** — each enrichment provider is tried in sequence (max iterations = number of providers), with early exit on success. This is the same mechanism. Build a mini-waterfall: iterate through a list of data lookup functions, stop on first success or exhaust the list. Print which provider resolved and how many attempts it took.

[CITATION NEEDED — concept: Clay waterfall enrichment iteration bounds and early exit behavior]

## Ship It

Hard exercise: implement an iteration scheduler that refines a company description using an LLM. The scheduler must: cap at 5 iterations, apply exponential backoff starting at 1 second, check convergence by computing edit distance between consecutive outputs, and log every decision (continue/stop reason). Output a structured JSON trace of the full run.

## Extend It

Compare fixed-bound vs. convergence scheduling for different task types (factual lookup vs. creative generation). When does convergence lie? Research OpenAI's agent loop patterns and their documented approach to iteration control — map their `max_turns` parameter to the fixed-bound mechanism covered here.