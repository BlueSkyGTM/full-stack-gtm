# Production Runtimes: Queue, Event, Cron

## 1. Hook

You built the enrichment workflow. It works in your terminal. Now it needs to run at 2:14 AM when a prospect's job changes, retry when the CRM API returns 429, and refresh your ICP scoring every Monday without you touching it. The gap between "works on my machine" and "works in production" is which runtime primitive you choose and whether you chose correctly.

## 2. Concept

Three distinct execution models, each solving a different failure mode. **Queue**: ordered, persistent, retryable — for work that must complete exactly once (enrichment writes, CRM updates). **Event**: reactive, fan-out, decoupled — for work triggered by state changes (signal detected → notify + score + enqueue). **Cron**: scheduled, idempotent — for work that runs on a clock regardless of state (weekly domain warmup, nightly stale-record sweeps). The mechanism that matters: each primitive has a different guarantee about delivery, ordering, and exactly-once semantics. Pick wrong and you either lose writes or double-send prospecting emails.

## 3. Walkthrough

Build all three primitives in a single Node.js script using BullMQ (Redis-backed queue), EventEmitter (in-process event bus), and node-cron (cron scheduler). Wire them together: a cron job emits an event, the event handler enqueues work, the queue processes with retry and dead-letter logging. Every step prints observable output — no mocks, no stubs. Demonstrates the actual control flow: schedule → trigger → enqueue → process → ack/fail.

## 4. Use It

This is your **Living GTM** production layer (Cluster 13). Your Clay enrichment tables need cron-triggered refreshes on a schedule, not manual button clicks. Your signal monitoring (job changes, funding rounds) is an event source that fans out to scoring, Slack alerts, and CRM creation. Your outbound sequencing with per-provider rate limits is a queue with backoff — otherwise you burn the sending domain. The deploy pipeline ships these three runtime configs alongside your SPF/DKIM/DMARC infrastructure. Without the right primitive, your GTM stack either drops prospects or sends duplicates. Neither is recoverable.

## 5. Ship It

- **Easy**: Configure a cron job that prints the current enrichment queue depth every 5 minutes. Confirm it fires on schedule with observable timestamps.
- **Medium**: Build an event emitter that listens for `signal.detected` events and enqueues a processing job with retry (3 attempts, exponential backoff). Simulate a failure and confirm the dead-letter output.
- **Hard**: Wire all three together — cron triggers a stale-record scan, scan emits events per stale record, events enqueue enrichment jobs, jobs process with rate-limit awareness. Print a full execution trace showing the chain from schedule to completion. This is the pattern that runs your Clay + n8n stack unattended.

## 6. Reflect

Before this lesson, your automation was something you ran. After this lesson, it's something that runs itself — and you think in terms of delivery guarantees, not just "does it work." The shift: from triggering workflows manually to designing systems where the trigger, execution, and failure handling are three separate decisions with three separate correctness criteria. That's the difference between a practitioner and an engineer.