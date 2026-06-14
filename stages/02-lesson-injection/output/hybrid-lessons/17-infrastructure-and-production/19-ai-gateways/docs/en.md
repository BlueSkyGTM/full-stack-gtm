# AI Gateways — LiteLLM, Portkey, Kong AI Gateway, Bifrost

## Learning Objectives

- Configure a LiteLLM proxy with multiple providers, fallback chains, and rate limits from a YAML config file.
- Trace a single request through the five gateway functions: normalization, routing, fallback, caching, and observability.
- Compare LiteLLM, Portkey, Kong AI Gateway, and Bifrost across scale ceiling, latency overhead, and deployment model.
- Implement cost tracking and spend attribution per provider using the gateway's transaction log.
- Deploy LiteLLM in a Docker container with a production-ready configuration including budget caps and retry policies.

## The Problem

Your enrichment pipeline calls OpenAI 4,000 times an hour. At 2:14 AM EST, the OpenAI API returns `503 Service Unavailable`. Your retry logic — if you have any — hammers the endpoint three times with no backoff, then fails the row. Every subsequent request in the batch hits the same wall. No fallback to Anthropic. No log of which rows were lost. No cost summary at the end of the run. Your CRM now has stale intent signals on 3,000 accounts, and you won't notice until a rep complains next Tuesday.

This is not an OpenAI reliability problem. It is an architecture problem. Your application code is making direct, unmediated calls to a provider API. When that provider degrades, your application has no recovery path. When you want to switch providers, you rewrite SDK calls across your codebase. When finance asks what the LLM spend was per enrichment batch, you have nothing. Every provider has a different SDK, error taxonomy, rate limit header format, and authentication scheme — your application code should not have to know any of that.

An AI gateway sits between your application and the provider to absorb this complexity. Your app calls one endpoint. The gateway handles routing, failover, caching, rate limits, cost tracking, and logging through a single control plane. The provider can change, fail, or rate-limit — your application code stays the same.

## The Concept

The core mechanism is request interception. Instead of your application calling `https://api.openai.com/v1/chat/completions`, it calls `https://your-gateway/v1/chat/completions`. The gateway receives the request, applies its policy layer, translates the request to the target provider's format, executes it, translates the response back to a normalized shape, logs the transaction, and returns it. Your application never knows which provider actually handled the request.

Five functions define a production AI gateway:

**Provider normalization** — Every provider accepts different request schemas. OpenAI uses `messages`, Anthropic uses `messages` with a separate `system` parameter, Cohere uses a completely different `chat_history` structure. The gateway exposes one API (almost always OpenAI-compatible, because it's the de facto standard) and translates inbound requests to whatever the target provider expects. Your code writes one request format forever.

**Routing** — The gateway decides which provider handles which request based on rules: cost threshold ("use GPT-4o-mini unless the prompt contains 'legal'"), latency target ("route to the fastest-responding provider"), model capability ("only Claude can handle 200K context"), or round-robin distribution across providers to spread load.

**Fallback and retry** — When a provider returns a 429 (rate limited), 500, 502, 503, or times out, the gateway retries with exponential backoff, then falls through to the next provider in a configured chain. If OpenAI fails, try Anthropic. If Anthropic fails, try your self-hosted Llama. The application gets a successful response or a definitive failure — never a partial one.

**Caching** — Identical requests (exact string match) or semantically similar requests (embedding similarity) return a cached response instead of calling the provider again. This cuts cost on repeated prompts — common in enrichment where many rows produce similar extraction prompts.

**Observability** — Every request is logged: provider, model, tokens in, tokens out, latency, cost, success/failure, fallback path triggered. This is the transaction log that finance needs and that you need to debug why row 2,847 returned garbage.

```mermaid
flowchart TD
    A[Application Request] --> B[Gateway Endpoint]
    B --> C[Normalize: translate to OpenAI format]
    C --> D[Route: apply rules - cost, capability, latency]
    D --> E{Cache hit?}
    E -->|Yes| F[Return cached response]
    E -->|No| G[Call primary provider]
    G --> H{Success?}
    H -->|Yes| I[Log transaction - cost, tokens, latency]
    H -->|No - 429/5xx/timeout| J[Retry with backoff]
    J --> K{Retries exhausted?}
    K -->|No| G
    K -->|Yes| L[Fallback to next provider]
    L --> H
    I --> M[Cache response]
    M --> N[Return normalized response]
    F --> N
```

Four gateways dominate the 2026 landscape, each with a different center of gravity:

**LiteLLM** is a MIT-licensed open-source proxy with translation layers for 100+ providers, all normalized to the OpenAI chat completions format. You configure it with a YAML file, run it as a container, and it exposes `http://localhost:4000/v1/chat/completions`. It writes spend logs to SQLite by default (Postgres in production), supports fallback chains, rate limits per-key and per-model, and basic caching. Its weakness is scale: published benchmarks show cascading failures around 2,000 requests per second on 8 GB of memory. For Python-heavy teams running enrichment batches under 500 RPS — which covers most GTM engineering use cases — it is the default choice.

**Portkey** is an observability-first gateway that went Apache 2.0 open-source in March 2026. Its differentiator is the control plane: guardrails (PII redaction, jailbreak detection, prompt injection filtering), audit trails for compliance, and a dashboard that shows every request with its full request/response payload, latency breakdown, and cost. It adds 20-40 ms of latency overhead per request. Production tier is $49/month. If your use case involves sensitive data (customer PII in enrichment prompts) or compliance requirements (SOC 2 audit trails on every LLM call), Portkey's guardrail layer is the reason to choose it.

**Kong AI Gateway** is built on Kong's API gateway platform. Kong's own benchmark on identical hardware (12 CPUs) reports it is 228% faster than Portkey and 859% faster than LiteLLM. It uses a plugin architecture — you enable AI routing, rate limiting, and logging as plugins on top of the Kong gateway you may already run. Pricing is $100 per model per month (capped at 5 models on the Plus tier). This is the enterprise choice: if you already operate Kong for API management, adding the AI plugin is incremental. If you are not already on Kong, the operational overhead is hard to justify for a single use case.

**Bifrost** (built by Maxim AI) is a Go-based routing proxy focused on performance and provider abstraction. It handles automatic retries with configurable backoff strategies and provider fallback (e.g., fall back to Anthropic when OpenAI returns 429). Being written in Go gives it a throughput advantage over Python-based LiteLLM for high-RPS scenarios. Bifrost is the choice when you need LiteLLM's routing semantics but have hit its throughput ceiling and do not want the Kong enterprise stack.

The self-hosted versus managed decision is driven by data residency. If your enrichment prompts contain customer PII or proprietary company data that cannot leave your infrastructure, you self-host. LiteLLM and Bifrost run anywhere as containers. Portkey and Kong offer both self-hosted OSS and managed cloud tiers. Cloudflare and Vercel also offer managed AI gateways with zero operational overhead but minimal configuration — basic retry and caching only, no custom routing rules or fallback chains.

## Build It

We will build a LiteLLM proxy with two providers, a fallback chain, and spend logging. You need Docker installed and API keys for OpenAI and Anthropic. If you only have one provider key, the fallback demonstration will still work — the invalid key simulation below shows that.

First, create the config file:

```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

router_settings:
  fallbacks:
    - gpt-4o-mini:
      - claude-sonnet

litellm_settings:
  success_callback: ["lite_llm_logger"]
  failure_callback: ["lite_llm_logger"]
  budget_settings:
    max_budget: 10.0
    budget_duration: "1d"

general_settings:
  master_key: sk-litellm-master-key
  database_url: sqlite:///litellm_logs.db
```

Save this as `litellm_config.yaml`. Now start the proxy:

```bash
export OPENAI_API_KEY="sk-your-openai-key"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"

docker run -d \
  --name litellm-proxy \
  -p 4000:4000 \
  -e OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY \
  -v $(pwd)/litellm_config.yaml:/app/config.yaml \
  -v $(pwd)/litellm_logs.db:/app/litellm_logs.db \
  ghcr.io/berriai/litellm:main-latest \
  --config /app/config.yaml --port 4000 --num_workers 4
```

Wait about 15 seconds for the proxy to initialize. Check the logs:

```bash
docker logs litellm-proxy 2>&1 | tail -20
```

You should see `LiteLLM Proxy running on http://0.0.0.0:4000`. Now send a request through the gateway using the OpenAI format — even though Anthropic will be the fallback, the request shape stays the same:

```bash
curl -s http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-litellm-master-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "max_tokens": 10
  }' | python3 -m json.tool
```

The response comes back in OpenAI format regardless of which provider handled it. You will see `"model": "gpt-4o-mini"` in the response — the gateway preserves the virtual model name, not the underlying provider model.

Now simulate a provider failure. Break the OpenAI key by changing it to an invalid value and restarting:

```bash
docker stop litellm-proxy && docker rm litellm-proxy

export OPENAI_API_KEY="sk-invalid-key-for-testing"

docker run -d \
  --name litellm-proxy \
  -p 4000:4000 \
  -e OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY \
  -v $(pwd)/litellm_config.yaml:/app/config.yaml \
  -v $(pwd)/litellm_logs.db:/app/litellm_logs.db \
  ghcr.io/berriai/litellm:main-latest \
  --config /app/config.yaml --port 4000 --num_workers 4
```

Wait 15 seconds, then send the same request again:

```bash
curl -s http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-litellm-master-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "max_tokens": 10
  }' | python3 -m json.tool
```

The request still succeeds. LiteLLM tried OpenAI, got an auth error, and fell through to Anthropic per the fallback chain in the config. Your application code did not change — same endpoint, same model name, same request body.

Now read the spend log. LiteLLM writes every transaction to SQLite:

```bash
sqlite3 litellm_logs.db "
  SELECT 
    strftime('%Y-%m-%d %H:%M:%S', startTime) as time,
    model,
    CASE WHEN api_base LIKE '%anthropic%' THEN 'anthropic' ELSE 'openai' END as provider,
    ROUND(total_tokens, 0) as tokens,
    ROUND(response_cost, 6) as cost_usd
  FROM lite_llm_spend_logs
  ORDER BY startTime DESC
  LIMIT 10;
"
```

You will see rows for each request, including which provider actually handled it and the cost. The fallback event is visible: a request with `model: gpt-4o-mini` that was served by Anthropic. This is the audit trail.

```bash
docker stop litellm-proxy && docker rm litellm-proxy
```

## Use It

In a Clay waterfall enrichment workflow, you are calling an LLM to extract intent signals from account descriptions across thousands of rows. The waterfall pulls data from providers (LinkedIn, Crunchbase, Apollo), and an LLM prompt at the end of each row classifies the account as "high intent," "nurture," or "discard." That is hundreds of dollars in API spend per run, and any provider outage means stale data in your CRM. Route those LLM calls through LiteLLM: set a cost cap per enrichment run, configure Anthropic as fallback if OpenAI rate-limits, and check the spend log to attribute cost per enrichment batch. This is the infrastructure layer under Zone 2 enrichment — the waterfall pulls data, the LLM extracts signals, the gateway ensures it does not break or overspend.

Concretely: your Clay enrichment HTTP call points to `http://your-litellm-host:4000/v1/chat/completions` instead of `https://api.openai.com/v1/chat/completions`. The model parameter stays `gpt-4o-mini`. LiteLLM's budget cap (`max_budget: 50.0` in the config) acts as a circuit breaker — if a single enrichment run burns through $50 of API calls, the gateway stops accepting requests and returns a 403 with a budget-exceeded message. Your Clay table does not silently rack up a $400 bill because a prompt template regressed and started consuming 4,000 tokens per row instead of 400.

The spend log answers the question finance will eventually ask: "What did the Q3 outbound enrichment campaign cost in LLM API spend?" Query the SQLite (or Postgres in production) database by timestamp range, group by the virtual model name, and you have per-campaign attribution. This maps to Zone 17 (MLOps, model lifecycle) in the GTM system lifecycle — versioning your enrichment waterfalls means knowing what each version cost to run and detecting when spend drifts, which is the commercial analog to detecting when a scoring model drifts.

[CITATION NEEDED — concept: Clay waterfall LLM call routing through AI gateway for enrichment cost control]

## Ship It

Deploy LiteLLM as a Docker container with a production configuration. The dev config above used SQLite and a single worker. Production needs Postgres for the spend log (SQLite corrupts under concurrent writes), multiple workers for throughput, and environment-injected secrets.

Create `docker-compose.yml`:

```yaml
version: "3.9"

services:
  litellm-db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: litellm
      POSTGRES_USER: litellm
      POSTGRES_PASSWORD: ${LITELLM_DB_PASSWORD}
    volumes:
      - litellm_pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U litellm"]
      interval: 5s
      timeout: 5s
      retries: 5

  litellm-proxy:
    image: ghcr.io/berriai/litellm:main-latest
    ports:
      - "4000:4000"
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      LITELLM_DB_PASSWORD: ${LITELLM_DB_PASSWORD}
    volumes:
      - ./litellm_prod.yaml:/app/config.yaml
    command: --config /app/config.yaml --port 4000 --num_workers 8
    depends_on:
      litellm-db:
        condition: service_healthy

volumes:
  litellm_pgdata:
```

Create `litellm_prod.yaml`:

```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY
      rpm: 500
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      rpm: 100
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
      rpm: 400

router_settings:
  fallbacks:
    - gpt-4o-mini:
      - claude-sonnet
    - gpt-4o:
      - claude-sonnet
  routing_strategy: usage-based-routing-v2

litellm_settings:
  success_callback: ["lite_llm_logger"]
  failure_callback: ["lite_llm_logger"]
  cache: true
  cache_params:
    type: redis
    host: redis
    port: 6379
  budget_settings:
    max_budget: 500.0
    budget_duration: "1d"

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: postgresql://litellm:$(LITELLM_DB_PASSWORD)@litellm-db:5432/litellm
```

Deploy:

```bash
export LITELLM_DB_PASSWORD="$(openssl rand -hex 16)"
export LITELLM_MASTER_KEY="sk-litellm-$(openssl rand -hex 16)"
export OPENAI_API_KEY="sk-your-openai-key"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"

docker compose up -d

sleep 20

curl -s http://localhost:4000/health/readiness | python3 -m json.tool
```

The `/health/readiness` endpoint confirms the proxy started, connected to Postgres, and loaded all model configs. Verify the spend log is writing to Postgres:

```bash
docker exec litellm-proxy bash -c "pip install psycopg2-binary -q && python3 -c \"
import psycopg2, os
conn = psycopg2.connect(host='litellm-db', dbname='litellm', user='litellm', password=os.environ['LITELLM_DB_PASSWORD'])
cur = conn.cursor()
cur.execute('SELECT count(*) FROM lite_llm_spend_logs;')
print(f'Spend log rows: {cur.fetchone()[0]}')
cur.execute('SELECT model, round(sum(response_cost), 4) FROM lite_llm_spend_logs GROUP BY model ORDER BY model;')
for row in cur.fetchall():
    print(f'  {row[0]}: \${row[1]}')
conn.close()
\""
```

In a GTM context, this deployment is the infrastructure under your Clay enrichment waterfall — the same pattern from Zone 17. You version the config file alongside your Clay table definitions. When you change the prompt template or swap the primary model, you commit the config change. The spend log in Postgres gives you per-model cost attribution, so you can detect when a prompt regression causes spend to spike — the commercial equivalent of detecting scoring model drift.

## Exercises

**Exercise 1 (Easy): Configure a two-provider gateway and confirm routing.**

Create a `litellm_config.yaml` with two providers (OpenAI `gpt-4o-mini` and Anthropic `claude-3-5-sonnet-20241022`). Start the proxy, send a request to each model via `curl`, and confirm the response model field matches what you requested. Query the spend log and confirm two distinct providers logged.

**Exercise 2 (Medium): Add a fallback chain and simulate provider failure.**

Add a fallback chain: `gpt-4o-mini` falls back to `claude-3-5-sonnet-20241022`. Set the OpenAI key to an invalid value. Send a request for `gpt-4o-mini`. Confirm the response succeeds (served by Anthropic fallback). Query the spend log and identify the row where the primary model failed and the fallback handled the request.

**Exercise 3 (Hard): Add exact-match caching and demonstrate a cache hit.**

Add Redis caching to your config (or use in-memory caching with `type: local`). Send the same prompt twice. On the second request, confirm the response includes a cache-hit indicator (check the response headers or the `x-litellm-cache-hit` field). Query the spend log and confirm the second request logged zero cost. Then send a slightly different prompt (changed one word) and confirm it is NOT a cache hit — exact-match caching does not catch paraphrased prompts. For semantic caching, you would need `type: semantic` with an embedding model, which is a more advanced configuration.

## Key Terms

**AI Gateway** — A proxy server that sits between application code and LLM provider APIs, handling provider normalization, routing, fallback, caching, rate limiting, and observability through a single endpoint.

**Provider Normalization** — Translating requests and responses between the OpenAI-compatible format (the de facto standard) and each provider's native API schema (Anthropic, Cohere, Google Gemini, etc.).

**Fallback Chain** — An ordered list of providers where the gateway attempts each in sequence: if the primary fails (429, 5xx, timeout), the gateway retries with the next provider until one succeeds or all are exhausted.

**Spend Log** — A database table recording every gateway transaction: timestamp, model, provider, tokens consumed, cost in USD, latency, and success/failure status. Used for cost attribution and drift detection.

**Routing Strategy** — The rule set the gateway uses to select a provider for a given request: cost-based (cheapest first), capability-based (context window, vision support), latency-based (fastest responder), or usage-based (distribute load evenly).

**Virtual Model Name** — The model name your application requests (e.g., `gpt-4o-mini`) as defined in the gateway config, which may map to different underlying providers depending on routing and fallback. Your application code always references the virtual name; the gateway resolves it to a physical provider.

## Sources

- LiteLLM documentation: proxy configuration, fallback chains, spend logging — [https://docs.litellm.ai/docs/proxy/configs](https://docs.litellm.ai/docs/proxy/configs)
- LiteLLM benchmark: cascading failure at ~2000 RPS on 8 GB memory — [CITATION NEEDED — concept: LiteLLM throughput ceiling benchmark, specific source for 2000 RPS / 8GB figure]
- Kong AI Gateway benchmark: 228% faster than Portkey, 859% faster than LiteLLM on 12 CPUs — [CITATION NEEDED — concept: Kong AI Gateway performance benchmark, specific publication URL]
- Portkey open-source release (Apache 2.0, March 2026) — [CITATION NEEDED — concept: Portkey OSS license change announcement]
- Portkey latency overhead: 20-40 ms per request — [CITATION NEEDED — concept: Portkey latency overhead measurement source]
- Bifrost (Maxim AI): Go-based routing proxy with provider fallback — [https://github.com/maximai/bifrost](https://github.com/maximai/bifrost)
- Clay waterfall enrichment LLM call routing through AI gateway for cost control — [CITATION NEEDED — concept: Clay waterfall LLM call routing through AI gateway for enrichment cost control]
- Zone 17 GTM mapping: MLOps lifecycle applied to versioning enrichment waterfalls and detecting spend drift — [CITATION NEEDED — concept: GTM topic map Zone 17 MLOps/lifecycle row]