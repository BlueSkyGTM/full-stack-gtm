# AI Gateways — LiteLLM, Portkey, Kong AI Gateway, Bifrost

## Hook

Your enrichment pipeline calls OpenAI 4,000 times an hour. At 2:14 AM EST the OpenAI API returns `503 Service Unavailable`. Every request fails. No fallback. No retry. No log of what was lost. An AI gateway sits between your application code and the provider API to handle exactly this class of problem — routing, failover, caching, rate limits, and cost tracking through a single control plane.

## Concept

The mechanism: your application calls one endpoint (the gateway), which translates the request to whatever provider format is needed, applies routing rules (cost threshold, latency target, model capability), executes fallback chains on failure, caches identical or semantically similar requests, logs every transaction, and returns a normalized response. Cover the five core functions — provider normalization, routing, fallback/retry, caching, and observability — then map each to LiteLLM (open-source proxy, 100+ provider translation to OpenAI format), Portkey (observability-first gateway with caching and guardrails), Kong AI Gateway (plugin-based AI traffic management on Kong's API gateway), and Bifrost (Go-based routing proxy with provider abstraction).

## Demonstration

Start a LiteLLM proxy from a YAML config with two providers (OpenAI + Anthropic). Send a request via `curl`. Kill the OpenAI API key. Send the same request — observe the fallback to Anthropic. Print the spend log from the SQLite database LiteLLM writes to. Show the same normalized response shape regardless of which provider handled it.

## Guided Practice

Practitioner writes a `litellm_config.yaml` with three model entries, a fallback chain, and a rate limit. Starts the proxy. Sends requests that trigger each path: primary success, primary failure with fallback, rate limit exceeded. Reads the spend log and confirms cost per provider.

**Exercise hooks:**
- Easy: Configure a two-provider config and confirm routing via `curl`.
- Medium: Add a fallback chain and simulate provider failure by providing an invalid key.
- Hard: Add semantic caching and demonstrate a cache hit on a paraphrased prompt.

## Use It

In a Clay waterfall enrichment workflow, you're calling an LLM to extract intent signals from account descriptions across thousands of rows. That's hundreds of dollars in API spend per run, and any provider outage means stale data in your CRM. Route those LLM calls through LiteLLM: set a cost cap per enrichment run, configure Anthropic as fallback if OpenAI rate-limits, and check the spend log to attribute cost per enrichment batch. This is the infrastructure layer under Zone 2 enrichment — the waterfall pulls data, the LLM extracts signals, the gateway ensures it doesn't break or overspend.

[CITATION NEEDED — concept: Clay waterfall LLM call routing through AI gateway for enrichment cost control]

## Ship It

Deploy LiteLLM as a Docker container with a production config: three providers, fallback chains per model tier, per-team budget limits, Redis-backed caching, and a spend log exported to your observability stack. Write a Python client that calls the gateway for a batch enrichment job, handles partial failures, and logs total cost on completion. This is the same proxy configuration that keeps a GTM enrichment pipeline running when any single provider goes down.

**Exercise hooks:**
- Easy: `docker compose up` a local LiteLLM proxy with your config and confirm it responds.
- Medium: Write a Python script that processes a CSV of company descriptions through the gateway with error handling and cost reporting.
- Hard: Add a Redis-backed semantic cache and run the same enrichment batch twice — print the cache hit rate and cost delta.

---

**Learning Objectives (3–5, action verbs only):**

1. Configure an AI gateway proxy with multiple LLM providers, fallback chains, and rate limits.
2. Compare LiteLLM, Portkey, Kong AI Gateway, and Bifrost on provider normalization, caching, and observability mechanisms.
3. Implement fallback routing that automatically switches providers on failure.
4. Evaluate API spend across providers using gateway cost tracking logs.
5. Deploy a containerized AI gateway with production routing rules for a batch LLM workload.