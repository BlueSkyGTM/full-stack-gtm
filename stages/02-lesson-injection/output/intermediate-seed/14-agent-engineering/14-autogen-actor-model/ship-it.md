## Ship It

To deploy an actor-based enrichment pipeline in production, you need to handle three failure modes that the minimal runtime above does not address: actor timeouts, message redelivery, and backpressure. Each maps to a real operational concern in GTM tooling.

**Actor timeouts** occur when an enrichment provider (Clearbit, Apollo, a custom Claygent prompt) takes too long to respond. In the call-stack model, this blocks the entire pipeline. In the actor model, you add a supervisor actor that wraps each enrichment actor with a deadline. The supervisor sends the work message to the enrichment actor and simultaneously starts a timer. If the enrichment actor responds before the timer fires, the supervisor cancels the timer and forwards the result. If the timer fires first, the supervisor sends a failure message downstream and optionally retries with a fallback provider. This is the mechanism behind "try provider A, if it fails, try provider B" — which is exactly how a Clay waterfall with multiple data providers works.

```python
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
import uuid

@dataclass
class Message:
    sender: str
    recipient: str
    topic: str
    payload: Any
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

class Actor:
    def __init__(self, name: str, handler: Callable[[Message], Awaitable[Any]]):
        self.name = name
        self.inbox: asyncio.Queue = asyncio.Queue()
        self._handler = handler

    async def receive(self, msg: Message):
        await self.inbox.put(msg)

    async def run_one(self) -> list[Message] | None:
        msg = await self.inbox.get()
        if msg is None:
            return None
        return await self._handler(msg)

async def slow_provider(msg: Message):
    await asyncio.sleep(2.0)
    return [Message("provider", "sink", "result", {"data": "from slow provider", "cost_credits": 5})]

async def fallback_provider(msg: Message):
    await asyncio.sleep(0.2)
    return [Message("fallback", "sink", "result", {"data": "from fallback", "cost_credits": 2})]

async def sink(msg: Message):
    print(f"[SINK] {msg.payload}")
    return None

async def supervisor_with_timeout(work_msg: Message, primary: Actor, fallback: Actor, timeout_s: float):
    try:
        result = await asyncio.wait_for(primary.run_one(), timeout=timeout_s)
        print(f"[SUPERVISOR] Primary succeeded within {timeout_s}s")
        return result
    except asyncio.TimeoutError:
        print(f"[SUPERVISOR] Primary timed out after {timeout_s}s — switching to fallback")
        await fallback.receive(work_msg)
        return await fallback.run_one()

async def main():
    primary = Actor("provider", slow_provider)
    fallback = Actor("fallback", fallback_provider)
    sink_actor = Actor("sink", sink)

    work = Message("orchestrator", "provider", "enrich", {"domain": "acme.com"})
    await primary.receive(work)

    results = await supervisor_with_timeout(work, primary, fallback, timeout_s=1.0)

    if results:
        for msg in results:
            await sink_actor.receive(msg)
            await sink_actor.run_one()

    print("\n--- Cost Impact ---")
    print("Primary would have cost: 5 credits (never delivered, timed out)")
    print("Fallback cost: 2 credits (delivered result)")
    print("Net: saved 3 credits by using fallback instead of waiting")
    print("In a call-stack model, you would have paid 5 credits AND waited 2s AND gotten nothing")

asyncio.run(main())
```

The supervisor pattern is the mechanism behind multi-provider waterfalls. When Clay tries one enrichment provider and falls back to another on failure or timeout, that is a supervisor actor managing two child actors. The cost impact is direct: every credit you avoid spending on a timed-out provider is a credit you save (Zone 14 — GTM Stack Cost Management).

**Backpressure** matters when you are enriching a batch of 10,000 companies. If every domain resolver actor fires simultaneously, you will hit API rate limits instantly. The actor model handles this naturally: if the firmographic actor's inbox grows faster than it can process, messages queue. You can add a bounded queue that rejects new messages when full, forcing the upstream actor to slow down. This is not a feature you build — it is a property of message-passing architecture. The inbox *is* the backpressure mechanism.

The practical deployment guidance: start with AgentChat (AutoGen v0.4's high-level API) for prototyping. When you need custom routing, per-actor timeouts, or multi-provider fallbacks, drop to Core. When you need cross-machine distribution (enrichment actors running on different machines with different API keys), the same message protocol works — you swap the local `asyncio.Queue` for a Redis stream or RabbitMQ queue, and the actors do not change.

Note that AutoGen v0.4 is now in maintenance mode. Microsoft Agent Framework (public preview October 2025) is the successor and implements the same actor-model patterns with a different API surface. The concepts here — private state, message passing, supervisor patterns, fault isolation — transfer directly. What changes is import paths and class names.