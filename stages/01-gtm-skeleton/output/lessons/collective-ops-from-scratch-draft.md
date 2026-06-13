# Collective Ops From Scratch

## Hook It
You have N workers and one result to compute. Naive approaches either bottleneck on one node or produce incorrect results. Collective operations are the coordination primitives that make distributed computation correct by construction.

## Ground It
Prerequisite concepts: array partitioning, single-node reduce/fold, the difference between "send to one" and "send to all." A working Python environment with `numpy` is assumed.

## Explain It
Define each collective op—broadcast, scatter, gather, reduce, all-reduce, all-gather, reduce-scatter—as a mathematical pattern first (input shape → output shape, data movement topology). Then implement each from scratch using only lists and loops, no libraries. After the mechanism is clear, name what MPI and PyTorch call these same operations.

## Use It
**GTM Redirect:** Enrichment pipelines over large account lists are a distributed scatter-gather problem. You scatter a company list across workers, each worker calls an enrichment API, then you gather and reduce (dedupe, merge, score). This is the same pattern as `all_gather` followed by a merge-reduce. Foundational for Zone 02 (Enrichment & Scoring) and Zone 04 (Multi-channel orchestration at scale). [CITATION NEEDED — concept: enrichment-as-scatter-gather pattern in GTM tooling]

## Ship It
Build a minimal `CollectiveOps` class with `scatter`, `gather`, `reduce`, and `all_reduce` methods. All methods run locally against partitioned lists. Every method prints its input/output shapes and the intermediate state at each step so the data movement is observable.

## Prove It
- **Easy:** Given a 12-element list and 4 workers, print the result of `scatter` and confirm each worker gets exactly 3 elements.
- **Medium:** Implement `all_reduce` using only `gather` + `broadcast` (two-step). Print intermediate state after each step. Compare element count against the one-step ring pattern.
- **Hard:** Implement `reduce_scatter` from scratch. Given 8 workers and an 8-partition matrix, each worker receives exactly one partition of the reduced result. Print and verify no partition is duplicated and none are missing.

---

**Learning Objectives (draft):**
1. Implement broadcast, scatter, gather, reduce, all-reduce, and reduce-scatter from scratch using only lists and loops.
2. Compare the message complexity of naive gather-broadcast all-reduce versus ring-pattern all-reduce.
3. Diagnose incorrect output in a collective op by inspecting intermediate partition state.
4. Map a GTM enrichment pipeline onto a scatter-gather-reduce pattern and identify which collective op corresponds to each stage.