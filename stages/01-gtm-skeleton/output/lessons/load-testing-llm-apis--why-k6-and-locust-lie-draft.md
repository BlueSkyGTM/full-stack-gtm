# Load Testing LLM APIs — Why k6 and Locust Lie

## Hook
Your k6 script says the endpoint handles 500 req/s. Your enrichment pipeline dies at 30. The gap isn't a bug — it's a category error. Standard load testers measure HTTP semantics. LLM APIs bill on tokens, throttle on tokens, and stream over persistent connections. The numbers you've been reading are fiction.

## Concept
LLM APIs expose three load surfaces that HTTP benchmarks miss entirely: **token-rate ceilings** (TPM limits, not RPM), **streaming connection duration** (SSE holds sockets open 10–100× longer than a JSON response), and **time-to-first-token** (a latency profile that makes p99 meaningless). k6 and Locust count requests and measure response time to final byte. They conflate a 2-token completion and a 2000-token completion into the same metric. This section breaks down the four rate-limit dimensions (RPM, TPM, concurrent connections, requests in flight), explains why streaming invalidates standard concurrency math, and shows the correct throughput unit: **tokens per second per dollar**, not requests per second.

## Demonstration
Run a k6 script against an OpenAI-compatible endpoint using request-rate targeting. Show the lie: k6 reports 200 req/s with acceptable p95. Then run the same load with a token-aware script that tracks input tokens, output tokens, TTFT, and stream duration. The second run reveals the real bottleneck at ~40 effective req/s when payload sizes vary. Print both result sets side-by-side to terminal.

## Guided Exercise
Three tiers:
- **Easy:** Instrument a single streaming call to print TTFT, tokens received, total wall time, and calculated tokens/second.
- **Medium:** Extend the k6 script to weight virtual users by token cost — send short, medium, and long prompts in a ratio matching production traffic; report throughput in tokens/second, not requests/second.
- **Hard:** Build a token-bucket model that predicts when you'll hit the TPM ceiling given your prompt distribution. Compare predicted saturation point to actual 429 errors from a sustained test run.

## Use It
Any Clay waterfall that calls an LLM enrichment step has a hidden throughput ceiling defined by TPM, not by the number of Clay credits you hold. If you're routing 10,000 accounts through a GPT-4 classify step, the bottleneck isn't Clay's queue — it's the token rate limit on your API key. This section shows how to calculate actual pipeline throughput by multiplying your average tokens-per-call by your Clay row count, then dividing by your TPM limit. That number is your true minimum pipeline runtime. Map this to Zone 1 infrastructure planning — specifically, sizing API key tiers and concurrent job counts for batch enrichment workflows. Foundational for GTM systems that rely on LLM-backed scoring or enrichment at scale.

## Ship It
Production deliverable: a `loadtest/` directory containing a token-weighted k6 scenario file, a Python script that models your TPM ceiling from a sample of real prompts, and a short markdown runbook that converts load-test output into a capacity estimate a GTM engineer can use to answer "how long will this Clay table take to run?" Print the capacity estimate as structured JSON to stdout for piping into dashboards or Slack alerts.

---

**Learning Objectives (draft):**
1. Compare request-rate and token-rate throughput using output from instrumented API calls.
2. Explain why streaming SSE connections invalidate standard concurrency calculations in k6 and Locust.
3. Implement a token-weighted load test that reports throughput in tokens/second.
4. Predict TPM saturation point given a prompt-size distribution and a sustained concurrency level.
5. Calculate minimum pipeline runtime for a batch enrichment job using TPM limits and average token cost per row.

**Citations:**
- OpenAI rate limit documentation (RPM/TPM tiers): [CITATION NEEDED — concept: OpenAI rate limit tiers by model and key tier]
- k6 streaming support and its limitations with SSE: [CITATION NEEDED — concept: k6 handling of Server-Sent Events and streaming responses]
- Locust event-based streaming measurement gaps: [CITATION NEEDED — concept: Locust SSE/async response measurement behavior]