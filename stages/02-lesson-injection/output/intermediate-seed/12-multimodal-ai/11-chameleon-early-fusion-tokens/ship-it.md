## Ship It

Deploying an early-fusion model — or deploying the tracing pattern it teaches — into a production GTM stack requires answering one question: where does the unified stream live? In Chameleon, the unified stream is the token sequence inside the transformer's context window. In a GTM pipeline, the unified stream is your observability layer — the system that ingests events from every tool (email platform, CRM, enrichment API, sequence orchestrator) and presents them as one traceable sequence.

Zone 12 defines this as living observability: not a static dashboard, but a real-time signal feed where reply rate drift, enrichment failure rate, and sequence step latency are all visible as events in a single stream. The connection to early fusion is structural. Late-fusion observability — checking each tool's dashboard separately and correlating manually — is the GTM equivalent of a vision encoder bolted to an LLM: two systems, a projector (your brain or a spreadsheet), and alignment friction. Early-fusion observability — one event stream, one vocabulary of event types, one tracing interface — is the Chameleon pattern applied to pipeline health.

Here is a minimal observability layer that implements the unified-stream pattern:

```python
from datetime import datetime, timedelta
import random
import json

random.seed(7)

EVENT_VOCAB = {
    "SEQ_START":        {"type": "sentinel", "zone": "12"},
    "SEQ_END":          {"type": "sentinel", "zone": "12"},
    "EMAIL_QUEUED":     {"type": "signal",  "zone": "2"},
    "EMAIL_SENT":       {"type": "signal",  "zone": "2"},
    "EMAIL_OPENED":     {"type": "signal",  "zone": "3"},
    "EMAIL_CLICKED":    {"type": "signal",  "zone": "3"},
    "REPLY_RECEIVED":   {"type": "signal",  "zone": "3"},
    "ENRICHMENT_FETCH": {"type": "action",  "zone": "1"},
    "ENRICHMENT_HIT":   {"type": "action",  "zone": "1"},
    "ENRICHMENT_MISS":  {"type": "action",  "zone": "1"},
    "CRM_PUSH":         {"type": "action",  "zone": "4"},
    "WATERFALL_START":  {"type": "sentinel", "zone": "1"},
    "WATERFALL_END":    {"type": "sentinel", "zone": "1"},
}


def emit_event(event_name, contact_id, extra=None):
    if event_name not in EVENT_VOCAB:
        raise ValueError(f"Unknown event: {event_name}")
    event = {
        "event": event_name,
        "contact_id": contact_id,
        "timestamp": datetime.now().isoformat(),
        "meta": EVENT_VOCAB[event_name],
        "payload": extra or {},
    }
    return event


def simulate_contact_journey(contact_id):
    events = []
    events.append(emit_event("SEQ_START", contact_id))

    t = datetime.now()
    steps = random.choices(
        ["EMAIL_SENT", "ENRICHMENT_FETCH", "EMAIL_OPENED",
         "EMAIL_CLICKED", "REPLY_RECEIVED", "CRM_PUSH",
         "ENRICHMENT_MISS"],
        weights=[20, 15, 12, 8, 5, 15, 10],
        k=random.randint(4, 10),
    )

    for step in steps:
        t += timedelta(seconds=random.randint(30, 3600))
        event = emit_event(step, contact_id, {"elapsed_s": int((t - datetime.now()).total_seconds())})
        events.append(event)

    events.append(emit_event("SEQ_END", contact_id))
    return events


def health_report(all_events):
    total = len([e for e in all_events if e["meta"]["type"] != "sentinel"])
    by_type = {}
    for e in all_events:
        if e["meta"]["type"] == "sentinel":
            continue
        by_type[e["event"]] = by_type.get(e["event"], 0) + 1

    enrichment_total = by_type.get("ENRICHMENT_FETCH", 0) + by_type.get("ENRICHMENT_HIT", 0) + by_type.get("ENRICHMENT_MISS", 0)
    enrichment_hit_rate = by_type.get("ENRICHMENT_HIT", 0) / enrichment_total if enrichment_total > 0 else 0

    emails_sent = by_type.get("EMAIL_SENT", 0)
    replies = by_type.get("REPLY_RECEIVED", 0)
    reply_rate = replies / emails_sent if emails_sent > 0 else 0

    opens = by_type.get("EMAIL_OPENED", 0)
    open_rate = opens / emails_sent if emails_sent > 0 else 0

    print("=== PIPELINE HEALTH (Unified Event Stream) ===")
    print(f"\nTotal non-sentinel events: {total}")
    print(f"\n--- Key Metrics ---")
    print(f"  Reply rate:      {reply_rate:.1%}  ({replies}/{emails_sent})")
    print(f"  Open rate:       {open_rate:.1%}  ({opens}/{emails_sent})")
    print(f"  Enrichment hit:  {enrichment_hit_rate:.1%}  ({by_type.get('ENRICHMENT_HIT', 0)}/{enrichment_total})")

    print(f"\n--- Event Distribution ---")
    for event_name, count in sorted(by_type.items(), key=lambda x: -x[1]):
        zone = EVENT_VOCAB[event_name]["zone"]
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  [{zone:>2}] {event_name:<20} {count:>4}  {pct:>5.1f}%  {bar}")

    print(f"\n--- Drift Check ---")
    if reply_rate < 0.03:
        print("  WARNING: Reply rate below 3% — possible content degradation")
    if enrichment_hit_rate < 0.5:
        print("  WARNING: Enrichment hit rate below 50% — check provider health")
    if open_rate < 0.15:
        print("  WARNING: Open rate below 15% — subject line or deliverability issue")
    if reply_rate >= 0.03 and enrichment_hit_rate >= 0.5 and open_rate >= 0.15:
        print("  All metrics within expected range.")


all_events = []
for cid in range(1001, 1021):
    journey = simulate_contact_journey(cid)
    all_events.extend(journey)

health_report(all_events)

print("\n=== SAMPLE TRACE (contact 1001) ===")
sample = simulate_contact_journey(1001)
for e in sample:
    print(f"  [{e['meta']['zone']:>2}] {e['timestamp'][-12:-4]}  {e['event']:<20}  type={e['meta']['type']}")
```

This is not a toy abstraction. The event vocabulary (`EVENT_VOCAB`) is the shared token set. The `emit_event` function is the tokenizer — it maps a GTM action to a discrete event type with metadata. The sentinel events (`SEQ_START`, `SEQ_END`, `WATERFALL_START`, `WATERFALL_END`) bracket segments exactly like Chameleon's `<image-start>` and `<image-end>`. The health report computes metrics by parsing the unified stream, not by querying individual tool APIs.

When you deploy this pattern — whether with OpenTelemetry, a custom event bus, or structured logging into a queryable store — you get the same structural benefit Chameleon gets from early fusion: one stream, one vocabulary, one place to detect drift. Reply rate drift becomes your model degradation signal in the same way that a spike in image-token probability at a text position would signal something unusual in Chameleon's output distribution.