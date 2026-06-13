# Building an MCP Server — Python + TypeScript SDKs

## Hook

MCP is the wire protocol that lets an LLM call your code like a function. If you've ever wanted Claude to query your CRM, score a lead, or enrich a record mid-conversation, MCP is how that happens. This lesson covers building a server that exposes those tools, in both SDKs.

## Concept

MCP defines three primitives: **tools** (functions the model calls), **resources** (data the model reads), and **prompts** (reusable prompt templates). A server implements one or more of these. The SDK handles JSON-RPC transport, schema validation, and capability negotiation. The mechanism is request-response over stdio or HTTP — the model sends a structured request, your server returns a structured response. No magic, just a protocol.

## Walkthrough

Build a minimal MCP server in TypeScript that exposes one tool (e.g., a lead scorer that takes company domain and returns a score). Then build the same server in Python. Run each locally, connect via Claude Code Desktop, and invoke the tool from a prompt. Print the request/response payloads to confirm the protocol is working.

## Use It

MCP servers are the infrastructure layer for agentic GTM workflows. A server that exposes a `lookup_company` tool is what an agent calls mid-conversation to pull firmographic data — same pattern as the Clay waterfall enrichment, but you own the endpoint. Map this to **Zone 1: ICP & Scoring** — expose your scoring model as a tool so any MCP-compatible agent can call it during prospecting workflows. [CITATION NEEDED — concept: Clay MCP integration or MCP adoption in GTM tooling]

## Ship It

Covers: packaging your server for distribution (npx/uvx), configuring Claude Code Desktop to discover it, handling authentication for external APIs your tools call, error handling when the model sends malformed arguments, and logging for debugging tool calls in production.

## Drill

- **Easy**: Add a second tool to the existing server that returns a static JSON payload. Confirm it appears in `list_tools` output.
- **Medium**: Build a resource endpoint that returns a markdown document. Read it from Claude Code Desktop using a resource URI.
- **Hard**: Build a server with a tool that calls an external API (e.g., a mock enrichment endpoint), handles rate limiting, and returns structured errors back through MCP's error format.