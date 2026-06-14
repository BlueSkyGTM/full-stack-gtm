## Ship It

Deploying a multimodal enrichment pipeline into a GTM stack requires observability — without it, extraction quality drifts silently and your enrichment data degrades over time as websites change their layouts, add new visual elements, or switch to image-heavy designs that confuse the model's OCR pathway. Zone 12 (observability, logging, tracing) provides the monitoring infrastructure for this. The key insight is that in a living GTM system, model degradation is not an ML metric — it is a business metric. Reply rate drift, enrichment fill rate decline, and bounce rate increase are your model degradation signals.

For the multimodal enrichment pipeline, three tracing checkpoints matter. First, log the input: image resolution, domain, and a hash of the screenshot content so you can detect when a website changes its layout. Second, log the extraction output with its confidence score and the specific fields extracted. Third, log downstream impact: did this enrichment improve the account record, did the sales team use the extracted data, and did it correlate with engagement. This tracing setup monitors your enrichment pipeline performance in real time — reply rate drift is your model degradation signal, and confidence score decline is its leading indicator.

```python
import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List

@dataclass
class ExtractionTraceEvent:
    timestamp: str
    domain: str
    image_hash: str
    image_resolution: str
    fields_extracted: int
    fields_null: int
    confidence_score: float
    latency_ms: int
    status: str
    downstream_action: str = "none"


@dataclass
class EnrichmentTracer:
    service_name: str
    events: List[ExtractionTraceEvent] = field(default_factory=list)

    def trace_extraction(
        self,
        domain: str,
        image_bytes: bytes,
        resolution: str,
        extracted: dict,
        confidence: float,
        latency_ms: int,
    ):
        fields_extracted = sum(1 for v in extracted.values() if v is not None and v != [] and v != "")
        fields_null = len(extracted) - fields_extracted

        image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]

        action = "wrote_to_record"
        if confidence < 0.5:
            action = "routed_to_manual_review"
        elif fields_extracted < 3:
            action = "flagged_low_signal"

        event = ExtractionTraceEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            domain=domain,
            image_hash=image_hash,
            image_resolution=resolution,
            fields_extracted=fields_extracted,
            fields_null=fields_null,
            confidence_score=confidence,
            latency_ms=latency_ms,
            status="success" if confidence >= 0.5 else "low_confidence",
            downstream_action=action,
        )
        self.events.append(event)
        return event

    def compute_health_metrics(self, window: int = 50) -> dict:
        recent = self.events[-window:] if len(self.events) >= window else self.events
        if not recent:
            return {"status": "no_data"}

        avg_confidence = sum(e.confidence_score for e in recent) / len(recent)
        avg_latency = sum(e.latency_ms for e in recent) / len(recent)
        manual_rate = sum(1 for e in recent if "manual" in e.downstream_action) / len(recent)
        unique_hashes = len(set(e.image_hash for e in recent))
        hash_change_rate = 1.0 - (unique_hashes / len(recent)) if len(recent) > 1 else 0.0

        status = "healthy"
        alerts = []

        if avg_confidence < 0.6:
            status = "degraded"
            alerts.append(
                f"Average confidence {avg_confidence:.2f} below 0.60 threshold — "
                f"check for website layout changes or model version drift"
            )
        if manual_rate > 0.3:
            status = "degraded"
            alerts.append(
                f"Manual review rate {manual_rate:.0%} — enrichment pipeline "
                f"producing too many low-confidence extractions"
            )
        if avg_latency > 5000:
            alerts.append(
                f"Average latency {avg_latency:.0f}ms — consider batching or "
                f"reducing image resolution"
            )

        return {
            "status": status,
            "samples": len(recent),
            "avg_confidence": round(avg_confidence, 3),
            "avg_latency_ms": round(avg_latency),
            "manual_review_rate": round(manual_rate, 3),
            "image_change_rate": round(hash_change_rate, 3),
            "alerts": alerts,
        }


tracer = EnrichmentTracer(service_name="multimodal-enrichment-prod")

simulated_extractions = [
    ("stripe.com", b"\x89PNG_fake_stripe_v1", 0.92, 1200),
    ("linear.app", b"\x89PNG_fake_linear_v1", 0.88, 1100),
    ("retool.com", b"\x89PNG_fake_retool_v1", 0.85, 1300),
    ("stripe.com", b"\x89PNG_fake_stripe_v1", 0.91, 1150),
    ("unknown-startup.io", b"\x89PNG_fake_unknown_v1", 0.35, 2100),
    ("linear.app", b"\x89PNG_fake_linear_v1", 0.87, 1050),
    ("stripe.com", b"\x89PNG_fake_stripe_REDESIGNED", 0.52, 2400),
    ("retool.com", b"\x89PNG_fake_retool_v1", 0.84, 1250),
    ("unknown-startup.io", b"\x89PNG_fake_unknown_v2", 0.28, 2300),
    ("stripe.com", b"\x89PNG_fake_stripe_REDESIGNED", 0.48, 2500),
]

for domain, img_bytes, conf, latency in simulated_extractions:
    fake_fields = {
        "company_name": "Example" if conf > 0.5 else None,
        "tagline": "Tagline" if conf > 0.5 else None,
        "primary_brand_color": "#000000" if conf > 0.5 else None,
        "pricing_tiers_visible": conf > 0.5,
        "team_page_detected": conf > 0.7,
    }
    tracer.trace_extraction(
        domain=domain,
        image_bytes=img_bytes,
        resolution="1280x720",
        extracted=fake_fields,
        confidence=conf,
        latency_ms=latency,
    )

health = tracer.compute_health_metrics()
print("=== Enrichment Pipeline Health Report ===\n")
print(json.dumps(health, indent=2))

print("\n=== Recent Trace Events (last 5) ===\n")
for event in tracer.events[-5:]:
    print(json.dumps(asdict(event), indent=2))
    print()

print("=== Alert Interpretation ===")
if health["alerts"]:
    for alert in health["alerts"]:
        print(f"  ⚠ {alert}")
else:
    print("  No alerts — pipeline is within operational thresholds.")

print("\n=== GTM Signal: Connection to Zone 12 ===")
print(
    "In production, wire these trace events to your observability backend "
    "(Datadog, Honeycomb, or CloudWatch). The confidence_score field is "
    "your leading indicator — when it drops, enrichment quality drops, "
    "and within 1-2 weeks you will see reply rate decline in your "
    "sequence analytics. That lag is why real-time tracing matters: "
    "you catch the drift before it reaches the sales team."
)
```

The image change rate metric is specific to multimodal pipelines and has no equivalent in text-only enrichment. When a company redesigns their website, the screenshot hash changes, and the model may need re-evaluation on the new layout. A spike in hash change rate across many domains simultaneously could indicate a broader web design trend (e.g., everyone adopting a new framework with different visual patterns) that degrades your model's extraction accuracy industry-wide. This is the kind of failure mode that text-only observability cannot detect — it requires the visual tracing that native multimodal models make possible.