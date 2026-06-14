## Ship It

Before deploying any of these patterns to production, you need three things: observability, idempotency, and a convergence strategy.

Observability means every agent logs its inputs, outputs, and timing. In a fan-out of 200 concurrent calls, one agent hanging silently for 60 seconds is invisible without per-agent logging. At minimum, log the agent name, start timestamp, end timestamp, token usage, and a success/failure flag. Wrap each agent call in structured logging:

```python
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("agent")

async def observed_call(agent_name: str, prompt: str) -> str:
    start = time.monotonic()
    logger.info(f"[{agent_name}] START")
    try:
        result = await llm(prompt)
        elapsed = time.monotonic() - start
        logger.info(f"[{agent_name}] SUCCESS in {elapsed:.2f}s | output_len={len(result)}")
        return result
    except Exception as e:
        elapsed = time.monotonic() - start
        logger.error(f"[{agent_name}] FAILED in {elapsed:.2f}s | error={e}")
        raise

async def ship_demo():
    tasks = [f"Summarize company {name}" for name in ["Stripe", "Plaid", "Ramp"]]
    results = await asyncio.gather(*[observed_call("research", t) for t in tasks])
    for r in results:
        print(r[:80])

asyncio.run(ship_demo())
```

Idempotency means re-running an agent produces the same result without duplicating side effects. If your research agent writes to a database and the pipeline crashes mid-run, a retry should not create duplicate records. Use a deterministic task ID (e.g., `f"research:{company_name}:{date}"`) as a cache key. Before executing, check if the result already exists. This pattern is critical for enrichment waterfalls: if you enrich 1,000 contacts and the process dies at row 800, you should resume from row 800, not restart from row 1.

For convergence strategy, pick one explicitly. Do not leave it implicit. If two enrichment providers return different email addresses for the same contact, your code should have a documented rule: priority-based (provider 1 wins over provider 2), confidence-based (the provider with higher match score wins), or recency-based (most recently fetched value wins). Write the rule in a comment next to the merge logic. When a sales rep asks why the email bounced, you can trace it back to which provider supplied it and why that provider was chosen.

For the handoff and networked patterns, set a maximum depth or cycle count. A swarm handoff chain of depth 10 means 10 sequential LLM calls—potentially 30+ seconds of latency. A networked pipeline where the review agent rejects and re-triggers the draft agent in a loop will burn tokens indefinitely. Hard-cap both: `max_handoffs = 5` in the swarm, `max_cycles = 3` in the networked pipeline. When the cap is hit, fall back to a human or a default response.