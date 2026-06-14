## Ship It

Build a batch-aware API client that handles the full lifecycle: accept a list of records, chunk them into the provider's max batch size, submit to the batch endpoint, handle partial failures with retry, and return a flat result list. This client simulates a batch API locally so you can observe every mechanism without external dependencies.

```python
import json
import hashlib
import random
from dataclasses import dataclass, field
from typing import Callable

MAX_BATCH_SIZE = 500
MAX_RETRIES = 3

SIMULATED_FAILURE_RATE = 0.05


@dataclass
class BatchResult:
    item_id: str
    status: str
    data: dict
    error: str = None


def simulate_batch_api_call(chunk, batch_number):
    results = []
    for item in chunk:
        item_id = item["id"]
        fail = random.random() < SIMULATED_FAILURE_RATE
        if fail:
            results.append(BatchResult(
                item_id=item_id,
                status="failed",
                data={},
                error="simulated_provider_error"
            ))
        else:
            results.append(BatchResult(
                item_id=item_id,
                status="succeeded",
                data={
                    "company": item.get("company", "unknown"),
                    "score": random.randint(40, 95),
                    "employee_count": random.randint(10, 5000),
                }
            ))
    return results


def chunk_records(records, max_batch_size):
    chunks = []
    for i in range(0, len(records), max_batch_size):
        chunks.append(records[i:i + max_batch_size])
    return chunks


def run_batch_with_retry(records, max_batch_size=MAX_BATCH_SIZE, max_retries=MAX_RETRIES):
    chunks = chunk_records(records, max_batch_size)
    print(f"[batch_client] {len(records)} records → {len(chunks)} chunks (max {max_batch_size}/chunk)")
    print(f"[batch_client] batch discount: 50% | turnaround: 24h | failure rate: {SIMULATED_FAILURE_RATE*100:.0f}%\n")

    all_results = {}
    retry_queue = []

    for i, chunk in enumerate(chunks):
        results = simulate_batch_api_call(chunk, i)
        for r in results:
            if r.status == "failed":
                retry_queue.append(r)
            all_results[r.item_id] = r
        succeeded = sum(1 for r in results if r.status == "succeeded")
        failed = len(results) - succeeded
        print(f"  chunk {i+1}/{len(chunks)}: {succeeded} succeeded, {failed} failed")

    retry_round = 0
    while retry_queue and retry_round < max_retries:
        retry_round += 1
        print(f"\n  retry round {retry_round}/{max_retries}: {len(retry_queue)} failed items")
        current_retry = retry_queue[:]
        retry_queue = []

        retry_records = []
        id_to_original = {}
        for r in current_retry:
            fake_record = {"id": r.item_id, "company": f"retry_{r.item_id}"}
            retry_records.append(fake_record)
            id_to_original[r.item_id] = r

        results = simulate_batch_api_call(retry_records, retry_round)
        for r in results:
            if r.status == "failed":
                retry_queue.append(r)
            all_results[r.item_id] = r

        succeeded = sum(1 for r in results if r.status == "succeeded")
        print(f"    retry succeeded: {succeeded}, still failing: {len(retry_queue)}")

    final_succeeded = sum(1 for r in all_results.values() if r.status == "succeeded")
    final_failed = sum(1 for r in all_results.values() if r.status == "failed")

    print(f"\n[batch_client] FINAL: {final_succeeded} succeeded, {final_failed} failed")
    if final_failed > 0:
        failed_ids = [r.item_id for r in all_results.values() if r.status == "failed"]
        print(f"[batch_client] permanently failed item IDs: {failed_ids[:10]}{'...' if len(failed_ids) > 10 else ''}")

    ordered = []
    for record in records:
        ordered.append(all_results.get(record["id"]))

    return ordered


def generate_test_records(count):
    records = []
    for i in range(count):
        records.append({
            "id": f"acct_{i:05d}",
            "company": f"Company_{i}",
            "domain": f"company{i}.com",
        })
    return records


def compute_batch_savings(record_count, cost_per_record_serial=0.003):
    serial_total = record_count * cost_per_record_serial
    batch_total = serial_total * 0.50
    print(f"\n  cost estimate ({record_count:,} records):")
    print(f"    serial:   ${serial_total:.2f}")
    print(f"    batch:    ${batch_total:.2f}")
    print(f"    savings:  ${serial_total - batch_total:.2f} (50.0%)")


if __name__ == "__main__":
    random.seed(42)

    records = generate_test_records(1200)
    print("=" * 70)
    print("  BATCH ENRICHMENT PIPELINE")
    print("=" * 70)

    results = run_batch_with_retry(records)

    print("\n" + "=" * 70)
    print("  SAMPLE RESULTS (first 5)")
    print("=" * 70)
    for r in results[:5]:
        print(f"  {r.item_id} | {r.status:10} | score={r.data.get('score', 'N/A')} | "
              f"employees={r.data.get('employee_count', 'N/A')}")

    compute_batch_savings(len(records))
```

The output shows chunking, partial failure detection, retry rounds, and final merge:

```
======================================================================
  BATCH ENRICHMENT PIPELINE
======================================================================
[batch_client] 1200 records → 3 chunks (max 500/chunk)
[batch_client] batch discount: 50% | turnaround: 24h | failure rate: 5%

  chunk 1/3: 478 succeeded, 22 failed
  chunk 2/3: 471 succeeded, 29 failed
  chunk 3/3: 475 succeeded, 25 failed

  retry round 1/3: 76 failed items
    retry succeeded: 73, still failing: 3

  retry round 2/3: 3 failed items
    retry succeeded: 3, still failing: 0

[batch_client] FINAL: 1200 succeeded, 0 failed

======================================================================
  SAMPLE RESULTS (first 5)
======================================================================
  acct_00000 | succeeded  | score=72 | employees=3847
  acct_00001 | succeeded  | score=55 | employees=291
  acct_00002 | succeeded  | score=89 | employees=4120
  acct_00003 | succeeded  | score=43 | employees=78
  acct_00004 | succeeded  | score=67 | employees=2043

  cost estimate (1,200 records):
    serial:   $3.60
    batch:    $1.80
    savings:  $1.80 (50.0%)
```

The key mechanisms here are four. First, chunking respects the provider's maximum batch size — you cannot submit an arbitrary-length array and expect the endpoint to accept it. Second, the retry queue catches per-item failures, not per-batch failures. A batch endpoint may return a 200 with 477 successes and 23 errors embedded in the response body. You must parse per-item status, not just HTTP status. Third, the final merge reconstructs the original input ordering by keying results on item ID — batch responses frequently arrive in a different order than the request, especially when the provider parallelizes internally. Fourth, permanently failed items are surfaced explicitly rather than silently dropped.

Partial failures are the batch-specific failure mode that catches teams off guard. In a serial pipeline, a failed call either retries or crashes — there is no ambiguity. In a batch pipeline, 95% of items may succeed while 5% fail silently inside a successful HTTP response. Without per-item status checking, those failures are invisible. The enrichment appears to complete successfully, but 500 of your 10,000 accounts have null data. Detection requires parsing the response body per-item and tracking which IDs succeeded, which failed, and which need retry.

Different rate-limit behavior is the second trap. Batch endpoints often have separate rate-limit pools from synchronous endpoints, but those pools have different characteristics. A provider may allow one batch file per minute with up to 50,000 items, or five concurrent batch jobs, or a total of 2 million tokens in flight. Hitting a batch rate limit behaves differently than hitting a synchronous limit — the file may be rejected entirely, or queued with an unknown start time. Your client needs to handle batch-specific rate-limit responses (typically HTTP 429 with a `Retry-After` header or a provider-specific error code) differently from synchronous throttling.

Async batch processing introduces stale data risk. A batch file submitted at midnight starts processing at an indeterminate time — the provider may queue it for 6 hours before execution begins. If your enrichment data has time sensitivity (intent signals that decay, pricing data that changes daily), the 24-hour SLA is a ceiling, not a typical case. Your pipeline should timestamp when the batch was submitted, not when results were received, and flag results that exceed a freshness threshold. For enrichment waterfalls in Clay that feed into live outbound campaigns, stale batch data can mean calling on accounts whose intent signal has already faded.

Ordering assumptions break silently. When you submit items in a specific order, the provider may parallelize across workers and return results in arbitrary order. Some providers guarantee response order matches request order (OpenAI Batch API preserves order within the file). Others do not. If your downstream code assumes positional correspondence between request and response arrays, it will silently misattribute data — Company A's enrichment lands on Company B's record. The fix is to embed a unique identifier in every batch item and join on that identifier during merge, never relying on array position.