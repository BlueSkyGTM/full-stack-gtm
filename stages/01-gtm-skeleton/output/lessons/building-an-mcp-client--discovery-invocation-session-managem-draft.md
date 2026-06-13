# Building an MCP Client — Discovery, Invocation, Session Management

## Hook

You have a stack of APIs — Clearbit, Hunter, Apollo, Linkedin, your own internal services. Every integration is a bespoke auth flow, a bespoke request shape, a bespoke error handler. MCP (Model Context Protocol) replaces that chaos with a single JSON-RPC contract: discover what a server can do, call it, tear down the session. This lesson builds a client that does all three from scratch.

## Concept

Three mechanisms, one protocol:

**Discovery.** The client sends a `tools/list` or `resources/list` JSON-RPC request to a server. The server responds with a schema — name, description, input schema (JSON Schema). No guesswork, no scraping docs.

**Invocation.** The client sends `tools/call` with a tool name and arguments conforming to the discovered schema. The server returns content blocks (text, image, embedded resource). Errors come back as `isError: true` with diagnostic text — not HTTP status codes you have to decode.

**Session management.** MCP runs over two transports: `stdio` (local processes, pipes) and `Streamable HTTP` (remote servers, HTTP with SSE). Every session starts with an `initialize` handshake that negotiates protocol version and capabilities, and ends with a clean shutdown. The client must handle connection lifecycle, ping/pong keep-alives, and response correlation via JSON-RPC request IDs.

The protocol is specified at [spec.modelcontextprotocol.io](https://spec.modelcontextprotocol.io). Current specification version as of this writing is `2025-03-26`. If the spec has moved since then, defer to the canonical spec — this lesson describes mechanisms, not a snapshot.

## Demo

Build a complete MCP client that connects to any stdio-based MCP server, discovers its tools, invokes one, and shuts down cleanly. Uses Anthropic's official `@modelcontextprotocol/sdk` TypeScript package — the reference implementation.

```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const transport = new StdioClientTransport({
  command: "npx",
  args: ["-y", "@modelcontextprotocol/server-everything"],
});

const client = new Client(
  { name: "gtm-mcp-client", version: "1.0.0" },
  { capabilities: {} }
);

await client.connect(transport);

console.log("=== SESSION INITIALIZED ===");

const tools = await client.listTools();
console.log("=== DISCOVERED TOOLS ===");
for (const tool of tools.tools) {
  console.log(`  ${tool.name}: ${tool.description}`);
}

const result = await client.callTool({
  name: "echo",
  arguments: { message: "GTM pipeline test" },
});
console.log("=== INVOCATION RESULT ===");
console.log(JSON.stringify(result, null, 2));

await client.close();
console.log("=== SESSION CLOSED ===");
```

Output confirms each phase: the handshake succeeds, the discovered tools list prints, the echo tool returns the test message, and the session closes without error.

Second demo: the same pattern over Streamable HTTP transport against a remote server, demonstrating that the session layer is transport-agnostic — only the transport constructor changes.

## Use It

**GTM redirect — Zone 2, Data Engine: enrichment tooling.**

Every enrichment waterfall in Clay or a custom pipeline follows the same shape: discover what a provider can return (person data, company data, email validation), call it with what you know, fold the result into your record. MCP makes that shape explicit and portable. Instead of hard-coding the Clearbit API, the Apollo API, and the Hunter API as three separate integrations, you wrap each as an MCP server and your client discovers + invokes them uniformly.

This is the same mechanism as the Clay waterfall — sequential provider calls with fallback — but with a standardized contract between the orchestration layer and each provider. The discovery step is what lets you add or remove providers without changing orchestration code.

If you are building enrichment tooling, the exercise for this section has you wrap a real data source as an MCP server and call it from the client above.

[CITATION NEEDED — concept: Clay waterfall enrichment pattern and MCP integration]

## Ship It

**Production concerns for an MCP client in a GTM pipeline:**

- **Timeout management.** Every `callTool` needs a deadline. Enrichment calls that hang kill pipeline throughput. Wrap invocations in a `Promise.race` with a timeout.
- **Reconnection logic.** stdio transports die when the child process exits. Streamable HTTP transports drop on network blips. Implement exponential backoff reconnection with a max retry count.
- **Capability negotiation.** The `initialize` response tells you what the server supports (tools, resources, prompts, sampling). Gate your calls on what was negotiated — don't assume every server has tools.
- **Schema validation before invocation.** Validate your arguments against the discovered `inputSchema` before sending. Catches errors locally instead of round-tripping to the server.
- **Concurrent sessions.** If you're calling multiple enrichment providers, run sessions concurrently, not sequentially. Each session is independent — no shared state.

Exercise hook: instrument the demo client with timeout logic, then intentionally stall a server call to observe the timeout fire.

## Assess

**Easy.** What JSON-RPC method does a client send to discover available tools on an MCP server? What does the response contain?

**Medium.** You connect to an MCP server over stdio and the child process crashes mid-session. Describe what happens to pending `callTool` requests and what the client must do to recover.

**Hard.** Design an enrichment pipeline that calls three MCP servers concurrently — one for person data, one for company data, one for email validation. The pipeline should: (a) discover tools from all three on startup, (b) validate that the expected tools exist before running, (c) invoke all three concurrently with a 5-second timeout per call, and (d) aggregate results, skipping any that timed out or errored. Write the client code.