# MCP Transports — stdio vs Streamable HTTP vs SSE Migration

---

## Beat 1: Hook

The transport layer determines where your MCP server can run and who can reach it. Pick wrong and your agent sits in a subprocess it can't talk to, or listens on a port nothing connects to. This lesson covers the three transport mechanisms MCP supports, why SSE is being deprecated in favor of Streamable HTTP, and what breaks during migration.

---

## Beat 2: Concept

**stdio transport**: The client spawns the server as a child process. Messages are JSON-RPC over stdin/stdout. Lifecycle is tied to the parent process — when the client dies, the server dies. No network stack involved.

**SSE transport (legacy)**: The client opens two HTTP connections — one GET for the server-to-client stream (Server-Sent Events), one POST for client-to-server messages. Stateful by design; the server must maintain connection state. Defined in older MCP spec revisions and now deprecated. [CITATION NEEDED — concept: MCP spec version where SSE was deprecated]

**Streamable HTTP transport (current)**: Single-endpoint HTTP. Client POSTs JSON-RPC messages; server responds synchronously or streams via chunked transfer encoding. Stateless option available — server can operate without session affinity. Supports optional SSE streaming for multi-message responses, but the connection model is request/response, not long-lived bidirectional.

**Migration path**: SSE → Streamable HTTP. The server drops the dedicated GET endpoint. The client stops subscribing to an event stream and instead reads streaming responses from POST. Session management moves from implicit (connection = session) to explicit (`Mcp-Session-Id` header).

---

## Beat 3: Demonstration

Three minimal MCP servers, each using a different transport, all exposing the same tool. Each prints observable output confirming message receipt and transport type.

- **stdio demo**: Node script reads from stdin, logs each JSON-RPC message, writes responses to stdout. Run with `echo '{"jsonrpc":"2.0",...}' | node server.js`.
- **Streamable HTTP demo**: HTTP server listens on a port, accepts POST to `/mcp`, parses JSON-RPC, streams back a response. Curl command triggers it.
- **SSE demo (for comparison)**: HTTP server with GET `/sse` endpoint and POST `/messages` endpoint. Shows the two-connection model. Curl commands open both.

Each example includes a minimal client snippet that connects, sends an `initialize` request, and prints the response — confirming the transport works end-to-end.

---

## Beat 4: Use It

**GTM redirect**: MCP transports are foundational for **Zone 2 — Tooling & Integration**. Any GTM workflow where an AI agent calls external tools (enrichment APIs, CRM lookups, intent data providers) depends on the transport layer being correct for the deployment environment.

**Scenario**: You're deploying an MCP server that wraps a company enrichment API. In local development, stdio is fine — your agent runs in the same process tree. In production, the agent runs in a container and the enrichment server runs in another — you need Streamable HTTP so the agent can reach the server over the network. The exercise configures the same MCP server to accept both transports based on an environment variable.

**Exercise hooks**:
- *Easy*: Start an existing stdio-based MCP server and confirm it responds to an `initialize` request via pipe.
- *Medium*: Add a Streamable HTTP transport endpoint to a given MCP server and test with curl.
- *Hard*: Configure a single MCP server binary that selects transport based on `MCP_TRANSPORT` env var (stdio if unset, Streamable HTTP on port from `PORT` env var). Deploy locally and hit both modes.

---

## Beat 5: Ship It

Build a transport-agnostic MCP server that exposes one tool and can run in any of the three transport modes. Then migrate a provided SSE-based client to Streamable HTTP.

**Exercise hooks**:
- *Easy*: Run the provided server in stdio mode. Send an `initialize` and a `tools/list` request. Print both responses.
- *Medium*: Run the same server in Streamable HTTP mode. Write a script that sends the same two requests over HTTP and prints responses. Include the `Mcp-Session-Id` header in the second request.
- *Hard*: Given a legacy SSE client implementation, rewrite it to use Streamable HTTP. The client must handle both immediate JSON responses and streaming responses (chunked). Confirm by running against the server and printing each message as it arrives.

---

## Beat 6: Assess

**Learning objectives** (testable, action-verb):

1. **Compare** stdio, SSE, and Streamable HTTP transports by connection model, statefulness, and deployment constraints.
2. **Configure** an MCP server to accept connections over multiple transports.
3. **Implement** a Streamable HTTP client that handles both immediate and streaming responses.
4. **Migrate** an SSE-based MCP client to Streamable HTTP, preserving session semantics.
5. **Diagnose** transport mismatch failures from error output.

**Assessment hooks**:
- MC: Given a deployment description (agent in container A, server in container B), select the correct transport.
- Short answer: Name two differences between SSE transport and Streamable HTTP transport at the protocol level.
- Code trace: Given a failing SSE client log, identify which line assumes a GET-based event stream and explain why that breaks under Streamable HTTP.
- Debugging: Provided a server log showing connection refused on stdio, explain why and identify the correct transport for a remote deployment.

---

## GTM Redirect Rules (summary)

- **"Use It"**: Foundational for Zone 2 — transport choice determines deployment topology for AI-powered GTM toolchains (enrichment agents, CRM integrations, intent signal processors).
- **"Ship It"**: Same redirect. The exercise simulates the exact deployment decision a practitioner makes when moving an MCP-backed GTM tool from local dev to shared infrastructure.
- No forced connection: MCP transports are infrastructure. The redirect names the zone honestly.