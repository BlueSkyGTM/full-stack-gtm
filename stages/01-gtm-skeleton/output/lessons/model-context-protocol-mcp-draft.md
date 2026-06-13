# Lesson: Model Context Protocol (MCP)

## Hook

Every LLM integration re-implements the same glue: read a file, call an API, fetch a schema. MCP standardizes that glue into a single JSON-RPC contract so any model client can talk to any tool server without custom adapters.

---

## Concept

**What MCP is.** A JSON-RPC–based protocol where a *host* (e.g., Claude Desktop) spawns or connects to *servers* that expose three primitives: resources (read-only data), tools (callable functions), and prompts (reusable templates). The host and server negotiate capabilities through an initialization handshake before any work happens.

**Transport layer.** Two transport mechanisms exist: `stdio` (local subprocess — the server runs as a child process and communicates over stdin/stdout) and Streamable HTTP (remote — the server runs as an HTTP endpoint with SSE for server-to-client messages). Local tooling uses stdio; shared infrastructure uses HTTP.

**The lifecycle.** Client sends `initialize` with its capabilities → server responds with its capabilities → client sends `initialized` → session is active. Every subsequent message follows JSON-RPC 2.0: requests have `id`, `method`, `params`; responses have `id`, `result` or `error`; notifications have `method` with no `id`.

**Three primitives in detail:**
- **Resources:** static or semi-static data the model can *read*. Think file contents, database schemas, configuration blobs. Servers emit `resources/list` and `resources/read`.
- **Tools:** functions the model can *invoke* with structured input. Servers emit `tools/list` (schemas) and the host sends `tools/call` with arguments. This is the extension point for any API integration.
- **Prompts:** parameterized prompt templates the server publishes. The host can discover them via `prompts/list` and render them via `prompts/get`.

**Why this matters over raw function calling.** Without MCP, every tool integration is a one-off adapter. MCP decouples the *shape* of the tool (its schema) from the *transport* used to reach it. One server can serve multiple hosts. One host can consume multiple servers. The protocol handles capability negotiation and error propagation; you focus on the tool logic.

---

## Demo

Build a minimal MCP server using the official Python SDK (`mcp` package) that exposes one tool — a function that takes a company domain and returns a hardcoded enrichment payload. Run it via `stdio` transport, and inspect the JSON-RPC messages by sending manual `initialize` → `initialized` → `tools/list` → `tools/call` frames through stdin. Print the raw JSON-RPC responses to confirm the protocol is working end-to-end.

Exercise hooks:
- **Easy:** Add a second tool that returns a different hardcoded payload, and verify it appears in `tools/list`.
- **Medium:** Add a `resources/list` handler that exposes a static markdown file, then read it with `resources/read`.
- **Hard:** Replace the hardcoded tool payload with a live HTTP call to a public API, and handle errors per the JSON-RPC error spec.

---

## Use It

MCP is the integration layer for connecting any LLM host to your GTM data stack. In Zone 02 (Targeting & Enrichment), a Clay waterfall is a fixed sequence of enrichment lookups — Apollo → Hunter → Clearbit. MCP generalizes that pattern: instead of hardcoding each vendor integration inside Clay, each vendor (or your wrapper around it) becomes an MCP tool server. The Clay agent becomes an MCP host that discovers available enrichment tools dynamically, calls them with structured input, and handles the response without custom glue code for each provider.

The mechanism: your enrichment wrapper exposes `tools/list` with a JSON Schema for each vendor's parameters. The host calls `tools/call` with the domain or company ID. Your server makes the API call and returns structured data. If a vendor goes down, the server returns a JSON-RPC error — the host can fall back to the next tool in its list. This is the same waterfall logic, but the contract is standardized.

Exercise hooks:
- **Easy:** Write an MCP server that wraps a single enrichment API (e.g., Hunter's domain-search endpoint) and returns structured results via `tools/call`.
- **Medium:** Extend the server to expose three tools (one per enrichment vendor) and include rate-limit metadata in each tool's response so the host can prioritize.
- **Hard:** Implement a `prompts/get` template that generates a personalized outreach email using the enrichment data from a previous `tools/call` — chain tool output into prompt rendering within the same MCP session.

---

## Ship It

Package a working MCP server as a CLI entry point with proper `stdio` transport, a `pyproject.toml` or `package.json` that declares the entry point, and a configuration snippet that a host (Claude Desktop, or any MCP client) would use to register it. The deliverable is an installable server that exposes at least one tool and one resource, and can be verified by running it standalone and sending JSON-RPC messages through a pipe.

Exercise hooks:
- **Easy:** Package the demo server as an installable Python package with a `console_scripts` entry point. Run it from the command line and confirm it starts listening on stdio.
- **Medium:** Add a `mcp.json` configuration file that a host would use to register your server. Document the exact JSON the host sends during the `initialize` handshake and the expected response.
- **Hard:** Add a Streamable HTTP transport mode to your server so it can run as a standalone HTTP service. Test it by sending raw JSON-RPC over HTTP (using `curl` or `httpie`) and confirming the SSE response stream.

---

## Evaluate

Assessment hooks targeting the protocol mechanics, not the SDK conveniences:

1. **JSON-RPC structure:** Given a raw JSON-RPC request frame for `tools/call`, identify whether it is valid (has `jsonrpc`, `id`, `method`, `params`) and predict the shape of the response.
2. **Capability negotiation:** Describe what happens if a client sends `tools/call` before the `initialize`/`initialized` handshake completes. Predict the error.
3. **Transport differences:** Compare stdio vs. Streamable HTTP transport — which scenarios require which, and what breaks if you use the wrong one.
4. **Primitive selection:** Given a use case (e.g., "expose a live CRM lookup" vs. "expose a static pricing schema"), determine whether to implement it as a resource or a tool, and justify.
5. **Error handling:** Write the JSON-RPC error response your server should return when an upstream enrichment API returns a 429 status code. Include the correct error code structure per the MCP spec.