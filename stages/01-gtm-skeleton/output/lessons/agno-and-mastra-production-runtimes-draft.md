# Agno and Mastra: Production Runtimes

## Hook It

You've built agents in notebooks and REPLs. Now a rep clicks "research this account" in Salesforce and expects a result in 12 seconds — not a traceback. Production runtimes solve the gap between "works on my machine" and "runs when a CRM webhook fires at 3 AM." This lesson covers two runtimes that take different architectural bets on how agents should execute in production.

## Map It

Define what a "production runtime" actually means: durable execution, state persistence, failure recovery, observability, and concurrency management. Compare the architectural models — Agno's Python-native agent-as-function approach versus Mastra's TypeScript-native workflow-graph approach. Explain where each runtime sits on the spectrum: lightweight orchestration layer vs. full workflow engine. Cover the mechanism of agent lifecycle management (initialization → tool execution → memory write → response) and how each runtime implements it differently.

## Build It

Exercise hook (easy): Instantiate a minimal Agno agent with one tool, pass a input, print the structured response and token usage to stdout.

Exercise hook (medium): Build a two-step Mastra workflow — step one fetches a company domain, step two enriches it — and print the intermediate state between steps to confirm the graph executes sequentially.

Exercise hook (hard): Implement the same agent in both runtimes (research a company given a domain), then print a side-by-side comparison of: latency, token count, retry behavior on a forced failure, and output schema validation result.

## Use It

Redirect to **Zone 02 — Enrichment** and **Zone 03 — Outbound**. The mechanism here is agent-driven research at scale: when a Clay waterfall can't resolve a data point, a production agent runtime can execute multi-step reasoning (search → scrape → extract → write back) as a durable job. Specifically: Agno or Mastra agents can be wired behind a Clay webhook to handle edge-case enrichment that static data providers miss. [CITATION NEEDED — concept: Clay webhook integration with external agent runtimes]

## Ship It

Cover what "done" looks like: containerized agent, health endpoint, structured logging (JSON to stdout), retry configuration, and a single curl command that exercises the agent end-to-end. Show deployment as a Docker container (Agno/FastAPI) and as a Mastra standalone server. Address the operational questions: what happens when the LLM provider rate-limits mid-execution, how to surface agent failures to the GTM team without exposing stack traces, and how to version agent prompts without redeploying. Exercise hook (medium): Wrap an Agno agent in a FastAPI endpoint, curl it, and confirm the response matches a JSON schema.

## Prove It

Five quiz questions grounded in observable behavior: identify which runtime preserves state between workflow steps, what the default retry behavior is for a tool call failure, how to access token usage metadata, what happens to a running Mastra workflow when the process restarts, and how to configure concurrent agent execution limits. Each question maps to a specific learning objective and can be answered by someone who ran the exercises.