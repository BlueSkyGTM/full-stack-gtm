# LLM Routing Layer — LiteLLM, OpenRouter, Portkey

## Hook
Your enrichment pipeline calls GPT-4 to classify 10,000 accounts. OpenAI returns a 429 at 2,000. Your pipeline is now broken and you're in Slack explaining why SDRs have no new leads. A routing layer intercepts that failure, falls back to Claude, and the pipeline finishes.

## Concept
**Mechanism: Provider abstraction + fallback waterfall.** Applications hardcode provider SDKs. When that provider fails, the application fails. A routing layer normalizes the OpenAI chat completions schema across providers, then implements a waterfall: try provider A, on failure try provider B, on failure try provider C. Three routing patterns emerge: cost routing (cheap model for classification, expensive for copywriting), latency routing (fastest responder wins), and fallback routing (primary → secondary chain). LiteLLM implements this as a Python library and proxy that translates OpenAI-format calls to 100+ provider APIs with config-driven fallback. OpenRouter implements this as a hosted gateway with a single endpoint and provider-negotiated pricing. Portkey implements this as an AI gateway adding caching, retries, and observability on top of routing.

## Demo
Working Python script that sends the same prompt through LiteLLM with a fallback config (GPT-4o → Claude → Gemini), intentionally triggers a failure on the first provider using an invalid model name, and prints which provider actually responded plus latency. Observable output confirms the waterfall executed. Second demo shows cost routing: same script sends a classification task to a cheap model and a generation task to an expensive model, printing token costs for each.

## Use It
**GTM redirect: AI-enriched data pipelines (Zone 1).** When Clay or a custom enrichment webhook calls an LLM to score leads or generate personalization, a routing layer sits between the webhook and the provider. Configure LiteLLM's fallback waterfall so that a provider outage does not break your enrichment flow. Specific mechanism: your enrichment function calls LiteLLM's proxy endpoint instead of OpenAI directly. If OpenAI 429s, the proxy falls back to Anthropic. Your Clay table never sees the failure.

## Ship It
Deploy LiteLLM as a proxy container alongside your enrichment worker. Set fallback configs as environment variables, not hardcoded. Log which provider handled each request and track fallback frequency — if fallback rate exceeds 20%, your primary provider is unreliable and you should promote your secondary. Add cost accumulation logging per GTM workflow (enrichment vs. personalization vs. scoring) so you can attribute AI spend to pipeline stages. Monitor latency per provider to catch degradation before it slows your enrichment batch jobs.

## Quiz
Five questions: identify which routing pattern (cost/latency/fallback) applies to a given scenario, predict waterfall behavior given a config and a specific failure type, explain what happens to in-flight requests when the primary provider returns a 429 and no fallback is configured, compare LiteLLM vs. OpenRouter on deployment model (self-hosted vs. hosted gateway), and identify the observability gap when using raw provider SDKs vs. a routing layer. All questions grounded in the mechanism, not tool trivia.

---

**Exercise hooks:**
- **Easy:** Configure LiteLLM with two providers and print which one responds.
- **Medium:** Implement cost routing that sends classification to GPT-3.5 and generation to GPT-4o, printing cost per call.
- **Hard:** Build a fallback chain that logs every attempt, latency per attempt, and total waterfall time to completion, then trigger it with an intentionally invalid first provider.