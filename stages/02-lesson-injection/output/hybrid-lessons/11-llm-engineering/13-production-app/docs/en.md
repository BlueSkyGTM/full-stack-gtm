# Building a Production LLM Application

## Learning Objectives

- Implement a production LLM client with exponential backoff retry, token-level cost tracking, and JSON schema validation against the live Anthropic API
- Classify LLM API failures as transient or semantic and apply the correct recovery strategy for each
- Trace the full request-response lifecycle of an LLM call through logging output that captures latency, token counts, retry events, and accumulated cost
- Build a health check and graceful degradation layer that keeps a GTM enrichment pipeline running when the model backend is unreachable

## The Problem

A working notebook demo and a system that runs 10,000 calls a day without incident are separated by about six weeks of engineering that nobody warns you about. The demo calls the API, gets a response, prints it, and you move on. Then production arrives and three failure modes eat your pipeline alive.

The first is **silent degradation**. The API returns a 429, your code catches the exception, swallows it, and returns an empty string. Downstream, a personalization step interpolates that empty string into an outbound email. The prospect receives a message that says "Hi , I noticed that ." Nobody flags it because the system never errored — it just stopped working correctly. Silent degradation is worse than a crash because a crash is visible.

The second is **uncontrolled spend**. You deploy an enrichment feature that calls the model once per account. Your CRM has 40,000 accounts. At $0.015 per call, that is $600 — manageable. But the feature runs on a cron that fires every four hours, the prompt includes a 4,000-token context block you forgot to cache, and three weeks later your API bill is $4,200 with no breakdown of which feature generated it. You have no cost ledger, no per-feature attribution, and no threshold alert. You just have a bill.

The third is **unhandled malformation**. You ask the model for JSON. It returns JSON wrapped in markdown fences. Or it returns valid JSON with a missing field. Or it returns a refusal — "I cannot help with that" — formatted as valid JSON with an empty object. Your parser does not check for any of these cases. The downstream system receives garbage and propagates it. By the time someone notices, 600 records in your CRM contain `{"error": "I cannot assist with this request"}` in the `ai_summary` field.

Every LLM application in production hits these three walls. The walls are not architectural — they are operational. You do not need a new framework. You need a client that retries the right errors, logs what it spends, validates what it receives, and fails loudly when it cannot recover.

## The Concept

Every LLM API call passes through the same lifecycle, and production hardening means instrumenting each stage. The lifecycle is: prompt assembly, token budget check, API call, response status classification, completion parsing, output validation, usage logging, and cost accumulation. At each stage, something can go wrong, and the failure mode determines your response.

Failures split into two families. **Transient failures** are conditions the provider tells you will resolve if you wait: rate limits (429), server errors (500, 502, 503), and timeouts. These are safe to retry because the underlying request was valid — the provider just could not handle it right now. **Semantic failures** are conditions where retrying will not help: authentication errors (401), bad requests (400 — usually a malformed prompt or context overflow), and responses where the model returns something structurally valid but semantically wrong (a refusal where you expected data, a prose paragraph where you expected JSON). Retrying a 400 error five times wastes time and money. Retrying a refusal five times is worse — it returns the same refusal five times and you pay for each one.

```mermaid
flowchart TD
    A[Prompt Assembly] --> B[Token Budget Check]
    B --> C[API Call]
    C --> D{HTTP Status}
    D -->|200| E[Parse Completion]
    D -->|429 500 502 503| F[Classify: Transient]
    D -->|400 401| G[Classify: Fatal]
    F --> H{Attempts Remaining?}
    H -->|Yes| I[Exponential Backoff]
    I --> C
    H -->|No| J[Raise to Caller]
    G --> J
    E --> K{Schema Valid?}
    K -->|Yes| L[Log Tokens and Cost]
    K -->|No| M[Classify: Semantic Failure]
    M --> J
    L --> N[Return Parsed Result]
```

The **retry-with-backoff** pattern handles transient failures. When the API returns a 429, you wait, then try again. The wait is not fixed — it grows exponentially. Attempt 1 waits 1 second, attempt 2 waits 2, attempt 3 waits 4, attempt 4 waits 8. This is exponential backoff: `delay = base * 2^attempt`. A small random jitter (typically 10–25% of the delay) prevents thundering herd problems where multiple retrying clients all hit the API at the same instant. Without jitter, if your system has 50 concurrent calls that all get rate-limited simultaneously, they all retry at the exact same moment and get rate-limited again. Jitter spreads them out.

The **circuit breaker** pattern is the next layer up from retry. If retries keep failing — say, 5 consecutive calls all hit 429s — continuing to retry is wasteful. The circuit breaker "trips" and blocks all outbound calls for a cooldown period (typically 30–60 seconds). During the cooldown, the client returns immediately with a fallback or error instead of making another doomed API call. After the cooldown, the breaker allows one test call through. If it succeeds, the circuit closes and traffic resumes. If it fails, the cooldown restarts. This pattern exists because a drowning provider does not benefit from you retrying harder.

The Anthropic Python SDK (`anthropic` package) implements retry-with-backoff internally for transient HTTP errors. It does not implement circuit breaking, semantic failure classification, cost tracking, or output validation. Those are your job. The SDK gives you the transport layer. Production hardening is everything you build on top of it.

## Build It

The following class wraps the Anthropic SDK with the four production concerns: retry classification (so semantic errors do not waste retry budget), token usage logging (so you can see what each call costs in real time), response schema validation (so malformed output is caught before it reaches downstream systems), and cost accumulation (so you have a running ledger per session). Every call prints structured output to the console — latency, token counts, retry events, and accumulated cost — so you can observe the system behaving correctly.

```python
import anthropic
import time
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("llm_prod")

PRICING_PER_MTOK = {
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}


class CostTracker:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.calls = 0

    def add(self, in_tok, out_tok, model):
        rates = PRICING_PER_MTOK.get(model, {"input": 3.00, "output": 15.00})
        cost = (in_tok / 1_000_000 * rates["input"]) + (out_tok / 1_000_000 * rates["output"])
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.cost_usd += cost
        self.calls += 1
        return cost

    def summary(self):
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


class ProductionLLMClient:
    TRANSIENT = (
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.InternalServerError,
    )

    def __init__(self, model="claude-sonnet-4-20250514", max_retries=4, base_delay=1.0):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.tracker = CostTracker()

    def _backoff_seconds(self, attempt):
        delay = self.base_delay * (2 ** attempt)
        return delay + (delay * 0.1)

    def _classify(self, err):
        if isinstance(err, self.TRANSIENT):
            return "transient"
        if isinstance(err, anthropic.BadRequestError):
            return "bad_request"
        if isinstance(err, anthropic.AuthenticationError):
            return "auth_error"
        return "unknown"

    def complete(self, system, messages, max_tokens=1024, temperature=0.0):
        for attempt in range(self.max_retries + 1):
            start = time.monotonic()
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=messages,
                )
                latency = time.monotonic() - start
                cost = self.tracker.add(
                    resp.usage.input_tokens, resp.usage.output_tokens, self.model
                )
                logger.info(
                    "OK latency=%.3fs in=%d out=%d cost=$%.6f total=$%.6f",
                    latency, resp.usage.input_tokens, resp.usage.output_tokens,
                    cost, self.tracker.cost_usd,
                )
                return resp

            except Exception as err:
                kind = self._classify(err)
                latency = time.monotonic() - start
                logger.warning(
                    "ERR type=%s attempt=%d/%d latency=%.3fs %s: %s",
                    kind, attempt + 1, self.max_retries + 1,
                    latency, type(err).__name__, str(err)[:200],
                )
                if kind != "transient" or attempt == self.max_retries:
                    raise
                wait = self._backoff_seconds(attempt)
                logger.info("RETRY backoff=%.2fs", wait)
                time.sleep(wait)

    def complete_json(self, system, messages, max_tokens=1024, temperature=0.0):
        resp = self.complete(system, messages, max_tokens, temperature)
        raw = resp.content[0].text
        stripped = raw.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            stripped = "\n".join(lines[1:-1])
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            logger.error("SCHEMA_VIOLATION non-JSON output: %s", stripped[:200])
            raise ValueError(f"Expected JSON, got: {stripped[:200]}")
        if not isinstance(parsed, dict):
            logger.error("SCHEMA_VIOLATION JSON is %s, not dict", type(parsed).__name__)
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
        return parsed


if __name__ == "__main__":
    client = ProductionLLMClient()

    system = "You are a data extraction assistant. Respond ONLY with valid JSON. No prose, no markdown."
    messages = [
        {"role": "user", "content": "Extract: company name is Acme Corp, industry is manufacturing, employee count is 450. Return JSON with keys: company, industry, employees."}
    ]

    result = client.complete_json(system, messages, max_tokens=256)
    print("\nParsed result:")
    print(json.dumps(result, indent=2))
    print(f"\nCost summary: {client.tracker.summary()}")
```

Run this and you will see structured log lines for the successful call: latency in milliseconds, input and output token counts, per-call cost, and accumulated total. The `complete_json` method strips markdown fences (because models frequently wrap JSON in ` ```json ` blocks despite instructions not to), attempts a parse, and raises a `ValueError` with a truncated preview of the raw output if parsing fails. That `ValueError` is the loud failure that prevents silent degradation — downstream code either gets a valid Python dict or an exception it must handle.

The retry loop classifies every exception before deciding what to do. A `RateLimitError` is transient — back off and retry. A `BadRequestError` is fatal — retrying will not fix a malformed prompt. An unknown error type is treated as fatal by default, which is the conservative choice: better to raise and let a human investigate than to retry an error you do not understand.

## Use It

Retry classification and structured cost tracking become non-negotiable when an LLM sits inside a Clay enrichment waterfall processing thousands of accounts. A Clay waterfall tries providers in sequence — if ZoomInfo returns nothing, try Apollo; if Apollo returns nothing, try an LLM-generated research summary. When the LLM is the last step, its output becomes the enrichment of record. A silent failure — an empty string, a truncated response, a refusal formatted as valid JSON — propagates into every downstream personalization field. The prospect receives an email with a blank where their GTM motion should be. This is Cluster 1.2, TAM Refinement & ICP Scoring — the exact layer where a malformed `{"error": "..."}` in the `ai_summary` field corrupts the ICP signal for an entire segment.

The production client prevents this three ways. The retry layer ensures a transient 429 does not produce an empty enrichment record. The JSON validation catches structurally malformed output before it reaches Clay's field mapping. The cost tracker gives per-session attribution so you know what the run cost before the monthly invoice arrives.

```python
import anthropic

client = ProductionLLMClient(model="claude-sonnet-4-20250514")

SYSTEM = """You are a B2B account research analyst. Given raw firmographic data,
produce a JSON object with these exact keys:
- icp_fit: one of "strong", "moderate", "weak"
- signal: a one-sentence reason for the fit rating
- suggested_channel: one of "email", "linkedin", "phone"
Respond ONLY with the JSON object."""

accounts = [
    {"name": "Stripe", "industry": "fintech", "employees": 8000, "signal": "hired 20 SDRs"},
    {"name": "Notion", "industry": "productivity", "employees": 400, "signal": "new enterprise tier"},
    {"name": "Midwest Manufacturing Co", "industry": "manufacturing", "employees": 120, "signal": "legacy ERP"},
]

for account in accounts:
    user_msg = f"Company: {account['name']}, Industry: {account['industry']}, Employees: {account['employees']}, Recent signal: {account['signal']}"
    try:
        enrichment = client.complete_json(SYSTEM, [{"role": "user", "content": user_msg}], max_tokens=200)
        print(f"{account['name']}: {enrichment['icp_fit']} — {enrichment['signal']}")
    except ValueError as e:
        print(f"{account['name']}: ENRICHMENT FAILED — {e}")
    except anthropic.APIStatusError as e:
        print(f"{account['name']}: API ERROR — {e}")

print(f"\nBatch cost: {client.tracker.summary()}")
```

Each account either gets a validated enrichment dict or a logged failure that a human can review. No silent empty strings. No unvalidated model output flowing into your CRM. The cost summary at the end tells you exactly what this batch cost — extrapolate to 3,000 accounts and you know the budget before you commit. [CITATION NEEDED — concept: Clay waterfall enrichment pricing models per-record]

The same pattern applies to reply classification in a Gong-style revenue intelligence workflow. When an SDR forwards a prospect reply for classification (interested, not interested, out of office, objection), the LLM call needs the same guarantees: retry on transient failure, validate the classification label against an allowed set, track cost per classification so you know what automated triage costs per month.

## Exercises

### Exercise 1 — Semantic Field Validation (Medium)

The `complete_json` method confirms the model returned a parseable JSON dict. It does not confirm the dict's *contents* are correct. A refusal formatted as `{"error": "I cannot assist"}` passes validation today and would propagate into your CRM. Fix this.

Add two optional parameters to `complete_json`: `required_keys` (a set of key names the response must contain) and `enum_fields` (a dict mapping field names to their allowed values, e.g., `{"icp_fit": {"strong", "moderate", "weak"}}`). After parsing, before returning, validate both conditions. Raise `ValueError` with the specific field name and observed value on failure. Then test with the enrichment script above: modify the system prompt to provoke a refusal and confirm the client rejects the output rather than passing garbage downstream.

### Exercise 2 — Circuit Breaker Integration (Hard)

The retry loop handles per-call transient failures. But if the Anthropic API goes down for 10 minutes, every call in your batch burns through its full retry budget — 5 attempts, 15 seconds of backoff, then an exception. With 3,000 accounts, that is 3,000 doomed retry sequences hitting a provider that is already down. Implement a `CircuitBreaker` class that prevents this.

The breaker tracks consecutive failures across all calls. After `failure_threshold` consecutive failures (default 5), it trips to the **open** state: `complete` raises immediately without making an API call. After `cooldown_seconds` (default 30) it transitions to **half-open**: one trial call is allowed. A success closes the circuit. A failure restarts the cooldown. Add state-transition logging (`CIRCUIT_OPEN`, `CIRCUIT_HALF_OPEN`, `CIRCUIT_CLOSED`). Integrate it into `ProductionLLMClient` so the breaker wraps the API call, not the retry loop — the breaker sits above retry. Test by setting `ANTHROPIC_API_KEY` to an invalid value and watching the breaker trip after 5 auth errors, then block subsequent calls instantly instead of retrying.

## Key Terms

- **Transient failure** — An API error condition that resolves with time: rate limits (429), server errors (500/502/503), timeouts. The request was valid; the provider could not handle it at that moment. Safe to retry with backoff.

- **Semantic failure** — A condition where retrying will not help: authentication errors (401), malformed prompts (400), or responses that are structurally valid JSON but semantically wrong (a refusal where data was expected, a missing required field). Retrying burns money for the same result.

- **Exponential backoff** — Retry delay that doubles with each attempt: `delay = base * 2^attempt`. Prevents a retrying client from hammering a provider that is already overloaded. Combined with jitter (random 10–25% variation) to avoid thundering herd effects when many clients retry simultaneously.

- **Circuit breaker** — A protective layer above retry that tracks consecutive failures across calls. After a threshold of failures, it blocks all outbound calls for a cooldown period rather than continuing to retry against an unavailable backend. The canonical pattern from distributed systems engineering.

- **Cost ledger** — Per-session or per-feature tracking of input tokens, output tokens, and accumulated dollar cost. Turns a monthly invoice into per-feature spend visibility. Without it, you cannot attribute spend to the enrichment run that caused it.

- **Markdown fence stripping** — Removing the `` ```json ... ``` `` wrapper that models frequently add to JSON output despite instructions not to. A preprocessing step before `json.loads()` — without it, the first JSON request often fails with a `JSONDecodeError`.

## Sources

- Anthropic Python SDK — retry behavior, exception hierarchy, and usage object structure: https://github.com/anthropics/anthropic-sdk-python
- Anthropic API pricing (input/output per million tokens by model): https://www.anthropic.com/pricing
- Martin Fowler, "CircuitBreaker" — canonical pattern reference for the breaker state machine (closed / open / half-open): https://martinfowler.com/bliki/CircuitBreaker.html
- [CITATION NEEDED — concept: Clay waterfall enrichment pricing models per-record]