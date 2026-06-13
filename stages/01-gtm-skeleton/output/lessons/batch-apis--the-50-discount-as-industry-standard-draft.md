# Batch APIs — the 50% Discount as Industry Standard

## Hook
API providers charge per-request overhead — connection setup, auth validation, rate-limit checks. Batching collapses N units of overhead into 1. That compression is why providers discount batch calls ~50%. If you're making serial API calls in a GTM pipeline, you're paying a tax for coordination you don't need.

## Concept
Explain the cost anatomy of a single API call vs. a batch call. Model why providers can offer 50% discounts — it's not generosity, it's their infrastructure cost dropping. Cover: request amortization, payload overhead, rate-limit token economics. Name specific providers who publish batch discounts (OpenAI, Anthropic, most enrichment APIs).

## Demo
Build a cost calculator that accepts a list-size and compares serial vs. batch pricing using real published rates from at least two providers. Output: a table showing unit cost, total cost, and savings percentage. Code runs in terminal, prints results, no scaffolding.

## Use It
Map to GTM enrichment workflows. When enriching a list of 10K accounts through an API (firmographic data, intent signals, scoring), serial calls at full price vs. batched calls at discount is the difference between a profitable and unprofitable outbound motion. This connects to the Clay waterfall pattern and bulk enrichment operations. [CITATION NEEDED — concept: batch enrichment pricing in Clay waterfall documentation]

## Ship It
Write a batch-aware API client that: (1) accepts a list of records, (2) chunks them into the provider's max batch size, (3) calls the batch endpoint, (4) retries on partial failure, (5) returns a flat list of results. Exercise hook: easy = call one batch endpoint with fixed data; medium = chunk dynamically based on provider limits; hard = handle partial failures and merge results.

## Edge Cases
Batch endpoints have failure modes that serial calls don't: partial failures (some items rejected), different rate-limit behavior (batch often has separate pools), stale data if batch processing is async, and ordering assumptions that break when responses come back in different order than requests. Cover detection and recovery strategies for each.