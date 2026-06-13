# Async and Hogwild! Inference

## Hook

You're serving 10,000 inference requests per minute and your batch scheduler just became the bottleneck. The queue is backing up, latency is climbing, and someone suggests "just make it async." But async what, exactly? And what happens to your model's outputs when multiple threads race on shared state?

## Concept

**Asynchronous inference** decouples request submission from result retrieval via queues and worker pools. The mechanism: a producer pushes work to a queue, workers pull at their own pace, results land in a separate channel. No blocking. No waiting for the slowest batch member.

**Hogwild!** is a specific flavor of async execution for *shared-memory parameter updates*. Originally described for gradient descent (Niu et al., 2011) [CITATION NEEDED — concept: Hogwild! convergence guarantees for sparse updates], the core claim: when updates are sparse relative to the total parameter space, lock-free concurrent writes converge almost as well as locked updates — because collisions are rare enough that the error they introduce is noise, not bias.

The conditions where Hogwild! works:
- Updates touch a small fraction of total parameters (sparse)
- The system is "almost linear" — conflicting writes don't corrupt structure
- You accept approximate correctness in exchange for throughput

Where it breaks:
- Dense updates (most parameters change every step)
- Any requirement for exact ordering or deterministic output
- Shared mutable state that isn't numerically tolerant of races

## Demo

Working Python examples showing:

1. **Sync vs async inference with a shared model**: submit N requests serially, then submit the same N via `asyncio` with a semaphore-bounded worker pool. Print timing and output consistency for both.

2. **Hogwild! shared-state update race**: spawn multiple threads updating a shared numpy array (simulating embedding weights) without locks. Compare final state against single-threaded sequential updates. Print divergence metrics — L2 distance between locked and lock-free results.

3. **Collision rate visualization**: given a parameter space of size P and update sparsity S, print the expected collision probability and show how it scales. Demonstrate that at ~1% collision rate, Hogwild! output is nearly identical to locked; at ~40%, it diverges.

## Use It

**GTM Cluster: Enrichment Waterfall (Clay) and Parallel Lead Scoring**

In GTM enrichment pipelines — specifically the Clay waterfall pattern where multiple data providers are queried sequentially, falling back on failure — async execution is the throughput mechanism. Each enrichment step is an independent API call touching different fields of the same company/contact record. The updates are sparse: one provider fills firmographic data, another fills tech stack, another fills funding. Hogwild!-style lock-free writes work here because each provider updates non-overlapping fields.

Where Hogwild! specifically applies in GTM: **parallel scoring of accounts where the scoring model updates a shared feature store.** Multiple workers score different accounts, writing features to a shared cache. If the features are account-scoped (sparse by definition), lock-free writes are safe.

Where it does *not* apply: any enrichment step where order matters (e.g., provider B's query depends on provider A's output). That's a dependency graph, not a race-tolerant system — you need structured concurrency, not Hogwild!.

## Ship It

**Easy**: Convert a synchronous batch inference script to async using `asyncio` and a bounded semaphore. Measure and print throughput improvement.

**Medium**: Build a lock-free shared feature cache for parallel account scoring. Spawn 50 concurrent scoring workers, each writing to a shared dictionary. Compare results against a threading.Lock-protected version. Print timing and divergence.

**Hard**: Implement a Hogwild!-style parameter server for an online lead-priority model. Workers compute gradient updates on minibatches and apply them to shared weights without locks. Track loss over time and compare convergence against a locked baseline. Determine the sparsity threshold where divergence becomes unacceptable for your specific feature space.

## Evaluate

- Explain why Hogwild! convergence degrades as update density increases, using the collision probability formula.
- Compare the output of async-in-order vs async-with-race-conditions on a shared scoring model. Identify which GTM scenarios tolerate the latter.
- Implement a detection mechanism: given the outputs of a lock-free inference system, determine whether race conditions meaningfully affected results without access to the locked baseline.
- Evaluate whether a specific Clay waterfall enrichment step (multiple providers, overlapping fields) is safe for lock-free execution or requires serialization.