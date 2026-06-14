## Ship It

These are exercise hooks — specifications without solutions. Build them against your framework and verify behavior with assertions.

**Easy: Add a `RetryNode` subclass.**

Subclass `Node` to create a `RetryNode` that accepts a `should_retry` predicate function. The base `Node.run` method retries on any exception. Your `RetryNode` should additionally inspect the result of `execute` — if `should_retry(result)` returns `True`, treat it as a logical failure and retry. Print the total retry count on success. Test it with a node that returns a sentinel value on the first two calls and a valid value on the third.

**Medium: Implement a `ConditionalBranch` node.**

Build a node that accepts a `predicate` function and two downstream `Node` instances (`if_true` and `if_false`). During `execute`, evaluate the predicate against the current context, run the appropriate downstream node inline, and write the result back to context. This breaks the linear pipeline model — your `ConditionalBranch` is a mini-pipeline within a pipeline. Execute both a qualified and disqualified company through a pipeline that contains a `ConditionalBranch` and print which path each company took.

**Hard: Add an `AsyncPipeline` that runs independent nodes concurrently.**

Extend your framework with an `AsyncPipeline` class that uses `asyncio.gather` to run nodes with no input dependencies on each other simultaneously. This requires a dependency resolver: inspect each node's `inputs` list, build a dependency graph, identify nodes whose inputs are already satisfied by the context, and run them in parallel. Print total execution time and compare it to sequential execution of the same nodes. A realistic test case: three independent fetch nodes (company data, news articles, technographic data) that all feed into a single scoring node. The three fetches should run concurrently; the scoring node waits for all three.