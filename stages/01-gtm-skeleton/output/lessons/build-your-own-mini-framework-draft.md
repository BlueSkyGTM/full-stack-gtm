# Build Your Own Mini Framework

## Hook

You've used LangChain. You've fought LangChain. Here's when to write 200 lines of your own orchestration code instead of importing 2,000 lines of someone else's abstraction leak.

## Concept

Three primitives—Node, Pipeline, Context—compose into a directed execution graph. Each node declares input/output schemas, the pipeline resolves execution order, and a shared context object carries state between nodes. This is the same mechanism underneath LangChain's RunnableSequence and Prefect's Task graph, minus the dependency weight.

## Demo

Build a working 200-line framework from scratch: a `Node` base class with `execute()`, a `Pipeline` class that chains nodes sequentially and handles retry/fallback, and a `Context` dict that accumulates state. Every code block runs in terminal and prints confirmation of the mechanism working.

## Use It

**GTM Redirect → Zone 1 (ICP & Account Intelligence)**

Wire three nodes into your framework: (1) fetch company data from an API, (2) pass the raw data to an LLM for enrichment/scoring, (3) format the output into a structured account record. This is the same pipeline pattern used in Clay's enrichment waterfall—except you own every line. *[CITATION NEEDED — concept: Clay enrichment waterfall internal architecture]*

## Ship It

Exercise hooks only:
- **Easy**: Add a `RetryNode` subclass that retries on `ConnectionError` with exponential backoff. Print retry count on success.
- **Medium**: Implement a `ConditionalBranch` node that routes to one of two downstream nodes based on a predicate function. Execute both branches in a single pipeline run and print which path was taken.
- **Hard**: Add an async `AsyncPipeline` that runs independent nodes concurrently using `asyncio.gather`. Print total execution time and compare to sequential execution.

## Evaluate

Write assertions against pipeline behavior: verify context state after each node, confirm retry logic fires the correct number of times, test that a failing node in the middle doesn't corrupt earlier context. Evaluate whether your mini-framework handles the three cases that made you reach for it— if it doesn't, identify the missing abstraction.

---

**Learning Objectives (for `docs/en.md`):**

1. Implement a composable pipeline framework using Node, Pipeline, and Context primitives
2. Configure retry and fallback behavior at the node level
3. Build a conditional branching mechanism that routes execution based on runtime data
4. Compare the architecture of your mini-framework to production orchestration tools
5. Evaluate when a custom framework is preferable to an existing library for a given workload