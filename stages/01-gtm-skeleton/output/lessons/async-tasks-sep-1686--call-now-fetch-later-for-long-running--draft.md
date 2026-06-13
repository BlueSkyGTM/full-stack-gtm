# Async Tasks (SEP-1686) — Call-Now, Fetch-Later for Long-Running Work

## Beat 1: Hook — Why This Pattern Exists

Synchronous API calls assume the work finishes before the connection drops. Enrichment waterfalls, bulk scoring, and large exports don't play by that rule. The call-now/fetch-later pattern splits one request into two: initiate the job, then retrieve the result on your terms. Without it, long-running GTM operations timeout silently or block the thread entirely.

---

## Beat 2: Concept — The Async Task Lifecycle

Three phases, five states. Phase one: `POST /tasks` returns a `task_id` immediately while work runs server-side. Phase two: `GET /tasks/{id}` polls for status transitions (`pending → running → completed|failed`). Phase three: when status is `completed`, fetch the payload. Cover polling strategies (fixed interval, exponential backoff), timeout ceilings, and the difference between polling and webhook-based retrieval. Name the pattern before naming any tool — this is the mechanism Clay's enrichment waterfall uses to coordinate multi-provider lookups without blocking.

---

## Beat 3: Demo — Working Async Task Client

Build a minimal async task system in Python: a mock server that accepts task creation, simulates work with a delay, and tracks state. Then build the client side — create a task, poll with exponential backoff, handle all three terminal states (completed, failed, timeout). Print output at every state transition so the practitioner can observe the lifecycle end to end. No frameworks, no ORMs — just `http.server` and `requests` to keep the mechanism visible.

---

## Beat 4: Use It — GTM Redirect

**GTM cluster: Zone 01 — Enrichment Waterfall**

When Clay runs an enrichment waterfall across multiple data providers, it uses this exact pattern. You submit a row for enrichment, receive a job identifier, and the platform polls provider APIs until all resolve or the waterfall exhausts its list. The same pattern applies to bulk scoring exports and large prospecting list generation. If you've ever triggered a Clay table enrichment and watched rows populate over 30+ seconds, you've watched call-now/fetch-later in production. The practitioner should be able to recognize when a GTM workflow requires async handling (anything touching external APIs with variable latency) versus when synchronous calls are sufficient (local lookups, cached data).

[CITATION NEEDED — concept: Clay internal async task implementation for waterfall enrichment]

---

## Beat 5: Ship It — Integration Exercise

**Easy:** Poll a single async task to completion with fixed-interval retries. Print status at each poll.

**Medium:** Implement exponential backoff (starting 1s, max 5s, ceiling 60s total) and handle the `failed` state gracefully with the error message from the task payload.

**Hard:** Build a task orchestrator that fans out 5 async tasks concurrently, polls all of them with individual timeouts, and returns a summary: which succeeded, which failed, which timed out. This mirrors what happens when you enrich a batch of leads — each row is its own async task, and you need to reconcile the outcomes.

---

## Beat 6: Review — What to Carry Forward

Summarize the pattern as a decision tree: if work takes <2s, go synchronous. If work takes >2s or latency is unpredictable, go async with polling. If you need real-time updates across thousands of tasks, graduate to webhooks. The practitioner leaves with a reusable polling function, an understanding of state transitions, and the ability to recognize when a GTM workflow requires async handling versus when it's over-engineering. Assessment questions focus on: identifying when to use the pattern, diagnosing stuck tasks, and choosing between polling strategies based on workload characteristics.

---

## Learning Objectives (for `docs/en.md`)

1. **Implement** a call-now/fetch-later client that initiates an async task and polls to completion
2. **Configure** polling strategies (fixed interval, exponential backoff) with appropriate timeout ceilings
3. **Handle** all async task terminal states: completed, failed, and timeout
4. **Compare** synchronous vs. async task patterns and identify when each is appropriate for GTM workloads
5. **Diagnose** a stuck or stale async task using status codes and elapsed time