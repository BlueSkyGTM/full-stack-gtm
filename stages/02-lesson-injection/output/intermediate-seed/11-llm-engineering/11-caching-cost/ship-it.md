## Ship It

**Production checklist for enrichment pipelines:**

Log cache hit rate continuously. If it drops below 60%, either your TTL is too short (data expires before the next run reads it) or your deduplication key is wrong (you are hashing on a field that varies across runs, like a timestamp). A healthy enrichment cache running weekly on a stable contact list should see 70-85% hit rates.

Expose the rate limiter's refill rate as an environment variable, not a hardcoded constant. Provider RPM limits change when you upgrade tiers or when the provider changes their policy. Hardcoding the rate means a production change requires a code deployment.

Track cost per enriched contact as a derived metric: total API spend divided by successfully enriched contacts. If this metric trends upward over weeks, investigate — it usually means duplicate rates are increasing (stale dedup keys) or cache TTL is too aggressive.

Set budget alerts at the provider level. OpenAI and Anthropic both support spend thresholds that trigger webhooks. Configure the alert at 80% of your monthly budget, not 100%, so you have time to react before the bill compounds.

For the composite pipeline (the Hard exercise below), log four numbers on every run: total input count, cache hits, API calls made, and estimated cost. These four numbers are your enrichment dashboard. If any of them spikes without a corresponding increase in contacts enriched, something is broken.