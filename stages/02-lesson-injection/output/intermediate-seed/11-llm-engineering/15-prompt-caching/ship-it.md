## Ship It

Production caching has three failure modes you need to instrument for. First, TTL expiry: Anthropic's default 5-minute window is unforgiving for bursty workloads. If your enrichment pipeline processes a batch of 500 companies, then waits 7 minutes for a downstream API call before processing the next batch, every cache entry from the first batch has expired. The fix is either the 1-hour TTL variant or structuring your pipeline to maintain cache warmth — process batches in overlapping windows rather than discrete sequential batches.

Second, prefix mutation: any change to the system prompt invalidates the cache from the point of change onward. If you include a timestamp ("Current time: 2025-01-15 10:30:00") in your system prompt, the prefix changes on every call and caching never activates. The same applies to dynamic few-shot selection — if you retrieve different examples per request based on the input, the prefix varies. Keep the system prompt deterministic, or split it into a cached static portion and an uncached dynamic portion using multiple `cache_control` markers.

Third, insufficient prefix length: Anthropic requires a minimum of 1,024 tokens (2,048 on Haiku) before caching activates. Google requires 2,048. If your system prompt is 800 tokens, you get zero caching benefit regardless of how many times you send it. Verify your prefix length by checking the `cache_creation_input_tokens` field on the first call — if it's zero when you expected a cache write, your prefix is below the threshold.

Here's a monitoring utility that tracks cache health across a batch of calls:

```python
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

@dataclass
class CacheMetrics:
    calls: list = field(default_factory=list)

    def record(self, response_usage, label=""):
        self.calls.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "input_tokens": getattr(response_usage, "input_tokens", 0),
            "output_tokens": getattr(response_usage, "output_tokens", 0),
            "cache_creation": getattr(response_usage, "cache_creation_input_tokens", 0),
            "cache_read": getattr(response_usage, "cache_read_input_tokens", 0),
        })

    def summary(self):
        total_input = sum(c["input_tokens"] for c in self.calls)
        total_output = sum(c["output_tokens"] for c in self.calls)
        total_cache_creation = sum(c["cache_creation"] for c in self.calls)
        total_cache_read = sum(c["cache_read"] for c in self.calls)

        calls_with_cache_write = sum(1 for c in self.calls if c["cache_creation"] > 0)
        calls_with_cache_read = sum(1 for c in self.calls if c["cache_read"] > 0)
        calls_with_neither = sum(1 for c in self.calls if c["cache_creation"] == 0 and c["cache_read"] == 0)

        cache_read_ratio = total_cache_read / (total_input + total_cache_read) if (total_input + total_cache_read) > 0 else 0

        uncached_cost = ((total_input + total_cache_read + total_cache_creation) / 1_000_000) * 3.00 + (total_output / 1_000_000) * 15.00

        cached_cost = ((total_cache_creation / 1_000_000) * 3.75
                      + (total_cache_read / 1_000_000) * 0.30
                      + (total_input / 1_000_000) * 3.00
                      + (total_output / 1_000_000) * 15.00)

        print("=== Cache Metrics Summary ===")
        print(f"Total calls:                {len(self.calls)}")
        print(f"Calls with cache write:     {calls_with_cache_write}")
        print(f"Calls with cache read:      {calls_with_cache_read}")
        print(f"Calls with no cache:        {calls_with_neither}")
        print(f"Cache read ratio:           {cache_read_ratio:.1%}")
        print(f"Uncached equivalent cost:   ${uncached_cost:.4f}")
        print(f"Actual cost (cached):       ${cached_cost:.4f}")
        print(f"Savings:                    ${uncached_cost - cached_cost:.4f} ({(1 - cached_cost/uncached_cost):.1%})")

        return {
            "cache_read_ratio": cache_read_ratio,
            "savings": uncached_cost - cached_cost,
            "calls_without_cache": calls_with_neither,
        }

metrics = CacheMetrics()

mock_usages = [
    type("Usage", (), {"input_tokens": 50, "output_tokens": 200, "cache_creation_input_tokens": 6000, "cache_read_input_tokens": 0})(),
    type("Usage", (), {"input_tokens": 55, "output_tokens": 180, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 6000})(),
    type("Usage", (), {"input_tokens": 60, "output_tokens": 220, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 6000})(),
    type("Usage", (), {"input_tokens": 45, "output_tokens": 190, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 6000})(),
]

for i, usage in enumerate(mock_usages):
    metrics.record(usage, f"company_{i+1}")

metrics.summary()
```

For reply classification — a Zone 11 (Revenue Intelligence) application where you categorize inbound email replies as interested, not interested, out of office, or needs follow-up — prompt caching applies when you have a detailed rubric. A reply classification system prompt with category definitions, edge-case handling rules, and few-shot examples can easily hit 1,500–2,000 tokens. Across 10,000 inbound replies per day, caching turns a $30/day input cost into $3/day. The cached prefix encodes your classification methodology; the dynamic suffix is each individual reply text.

For long-running enrichment pipelines that exceed the TTL window, consider a cache-warming loop: a lightweight job that sends a minimal request with the cached system prompt every 4 minutes to keep the cache alive between batches. This costs one cache read per warmup call (~$0.0018 for 6,000 tokens at Sonnet's cache-read rate) but prevents full cache rewrites that cost 12.5× more per token.