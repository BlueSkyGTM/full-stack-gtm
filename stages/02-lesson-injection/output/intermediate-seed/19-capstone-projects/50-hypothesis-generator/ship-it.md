## Ship It

A production hypothesis generator needs one more capability: **chaining**. When a test runs and returns results, those results are new evidence. The generator should consume that evidence, update its queue, and either promote the next hypothesis or kill candidates that the new evidence disconfirms.

The chain works like this: the initial seed prompt produces Queue v1. You run a test on Rank 1. The test result arrives. The generator re-scores every remaining hypothesis against the new evidence, re-ranks the queue, and outputs Queue v2. If Rank 1 passed, hypotheses that depend on the same mechanism get promoted. If Rank 1 failed, hypotheses that share its assumptions get demoted or killed.

```python
def update_queue_with_evidence(
    queue: list,
    test_result: dict,
    penalty_for_shared_variables: float = 0.15,
) -> list:
    tested_hypothesis = None
    for h in queue:
        if h.id == test_result["hypothesis_id"]:
            tested_hypothesis = h
            break

    if not tested_hypothesis:
        print(f"  Hypothesis {test_result['hypothesis_id']} not found in queue")
        return queue

    tested_vars = set(tested_hypothesis.variables)
    outcome = test_result["outcome"]

    updated = []
    for h in queue:
        if h.id == test_result["hypothesis_id"]:
            h.status = outcome
            print(f"  [Rank {h.id}] TESTED: {outcome}")
            updated.append(h)
            continue

        shared = tested_vars & set(h.variables)
        if outcome == "FAILED" and shared:
            h.rank_score = max(0.0, h.rank_score - penalty_for_shared_variables * len(shared))
            print(f"  [Rank {h.id}] PENALIZED: shares {shared} with failed hypothesis (-{penalty_for_shared_variables * len(shared):.2f})")
        elif outcome == "PASSED" and shared:
            h.rank_score = min(1.0, h.rank_score + penalty_for_shared_variables * 0.5 * len(shared))
            print(f"  [Rank {h.id}] BOOSTED: shares {shared} with passed hypothesis (+{penalty_for_shared_variables * 0.5 * len(shared):.2f})")

        h.status = getattr(h, "status