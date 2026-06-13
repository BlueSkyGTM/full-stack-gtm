# Leaderboard Aggregation

## Hook

You have five leaderboards ranking the same ten models. Each uses different benchmarks, different scales, and different subsets of models. You need one answer: which model do you deploy? Leaderboard aggregation is the mechanism that turns conflicting ranked lists into a single defensible ranking.

## Concept

Cover the core algorithms for combining multiple ranked lists: Borda count (positional scoring), Copeland's method (pairwise majority), and Kemeny-Young (optimal consensus permutation — NP-hard, so you approximate). Address the two prerequisite normalization problems: scale reconciliation (min-max, z-score, percentile) and missing item handling (some models don't appear on every leaderboard). Distinguish between rank-based aggregation and score-based aggregation — they behave differently when lists have uneven coverage.

## Use It

Build a function that takes N ranked lists with optional scores, normalizes them, and outputs a consensus ranking using at least two aggregation methods. Print the input leaderboards side-by-side with the aggregated output so the practitioner can see where methods disagree. GTM redirect: this is the multi-signal scoring pattern used in Zone 2 enrichment — when three intent providers rank the same accounts differently, you face the identical aggregation problem. [CITATION NEEDED — concept: multi-provider intent signal aggregation in GTM enrichment workflows]

**Exercise hooks:**
- *Easy:* Aggregate three ranked lists using Borda count. Print the consensus ranking.
- *Medium:* Add z-score normalization and handle three models that appear on only 2 of 4 leaderboards. Compare Borda vs. Copeland output.
- *Hard:* Implement Kemeny-Young approximation for 8+ items. Benchmark runtime and compare consensus quality against Borda on the same input.

## Debug It

Three failure modes to diagnose. First, the coverage trap: a model appearing on 2 of 5 leaderboards scores higher than one appearing on all 5, because the missing entries aren't penalized correctly. Second, scale conflation: raw score averaging when Leaderboard A uses 0–100 accuracy and Leaderboard B uses 0.0–1.0 F1 — the numbers combine but the result is meaningless. Third, the dictatorship problem: one leaderboard with extreme variance in scores dominates the aggregated result after normalization. Each failure mode gets a reproducible input that produces the wrong ranking, plus the fix.

**Exercise hooks:**
- *Easy:* Given a buggy aggregation that ignores missing items, identify which model is unfairly penalized and fix the scoring.
- *Medium:* Diagnose scale conflation in a provided code snippet. Add normalization and show the ranking changes.

## Explain It

The practitioner writes a short explanation answering: "Why can't you just average the ranks? Give a concrete 3-leaderboard example where rank averaging produces a different winner than Borda count, and explain what property of Borda fixes the problem." This forces the practitioner to articulate the mechanism, not just implement it.

## Ship It

Build a CLI tool that reads leaderboard data from JSON files (one file per leaderboard), runs the practitioner's chosen aggregation method, and writes the consensus ranking to stdout as a sorted JSON array. The tool must accept a flag for aggregation method (`--method borda|copeland|kemeny`) and a flag for normalization (`--norm zscore|percentile|minmax`). GTM redirect: this is the scoring reconciliation layer in a multi-signal enrichment pipeline — swap "leaderboard" for "intent provider" and the code is identical. [CITATION NEEDED — concept: multi-signal scoring reconciliation in GTM enrichment architecture]

**Exercise hooks:**
- *Medium:* Ship the CLI tool with Borda and z-score as defaults. Test with 4 provided JSON leaderboard files containing 15 models with partial coverage.
- *Hard:* Add a `--weights` flag that accepts per-leaderboard confidence weights, and implement weighted Borda count. Document how changing weights shifts the top-3 models using your test data.

---

**Learning Objectives (for `docs/en.md`):**

1. Implement Borda count and Copeland's method to aggregate multiple ranked lists into a single consensus ranking.
2. Detect and correct the missing-item coverage bias when models appear on uneven numbers of leaderboards.
3. Compare rank-based aggregation vs. score-based aggregation and identify when each is appropriate.
4. Diagnose scale conflation bugs in aggregation code by inspecting intermediate normalized values.
5. Configure a CLI tool that reads multi-leaderboard JSON input and outputs a defensible consensus ranking with a chosen normalization and aggregation method.