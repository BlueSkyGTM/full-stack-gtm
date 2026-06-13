# MCP Fundamentals — Primitives, Lifecycle, JSON-RPC Base

---

## Beat 1: Hook

**Why you're here and what problem this solves.** You've seen "MCP" in every tool's changelog. This beat strips the hype: MCP is a JSON-RPC protocol that lets a host (e.g., Claude Desktop) discover and invoke capabilities on a server over stdio or HTTP. No magic, no vendor claims — just a wire format with three primitives and a handshake. You'll build a minimal server from scratch before touching any SDK.

---

## Beat 2: Concept

**The mechanism, broken into three pieces.**

1. **JSON-RPC 2.0 base:** Every MCP message is a JSON-RPC `request`, `response`, or `notification`. Cover the required fields (`jsonrpc`, `id`, `method`, `params`, `result`, `error`) and why `id` correlation matters for async transports. Show the difference between a request (expects response) and a notification (fire-and-forget).

2. **Three primitives:** Tools (executable functions the model can call), Resources (read-only data the model can pull), Prompts (reusable prompt templates). Define each by what it does to the context window — tools *act*, resources *inform*, prompts *shape*.

3. **Lifecycle:** `initialize` → `initialized` → (operation) → `shutdown`. The capability negotiation that happens during init and why a server that skips the handshake will be rejected by spec-compliant hosts.

---

## Beat 3: Demonstration

**Working code that proves the concept.** Build a minimal MCP server using only Python's `json` and `sys` modules — no SDK, no framework. The server will:

- Respond to `initialize` with declared capabilities (one tool, one resource)
- Handle `tools/list` and `resources/list` requests
- Execute a `tools/call` for a trivial `echo` tool and return a result
- Print every incoming request and outgoing response to stderr for observability

All messages over stdio. Output confirmed by piping a hardcoded JSON-RPC request through stdin and capturing stdout.

---

## Beat 4: Guided Exercise

**Practitioner builds with support.**

- **Easy:** Modify the demo server to add a second tool (`add` — takes two numbers, returns sum). Send a hand-crafted JSON-RPC request and confirm the result.
- **Medium:** Add a `resource` that returns the contents of a local file. Subscribe to it via the resource protocol and print the content.
- **Hard:** Implement a notification handler — when the client sends a `notifications/cancelled`, the server aborts the in-progress tool call and logs the cancellation to stderr.

---

## Beat 5: Use It

**GTM redirect — where this connects to go-to-market engineering.** MCP is the transport layer behind tool-calling agents that power GTM automation. Specific cluster: **Zone 3, enrichment and action workflows** — an MCP server wrapping a CRM API (e.g., HubSpot contact search as a `tool`, account 10-K as a `resource`) is the mechanism that lets an agent autonomously research and score a prospect. The redirect: "this is how a Clay waterfall or Salesforce agent actually invokes its steps — each step is a tool on an MCP server." If the GTM fit feels forced for a given team, the fallback redirect is "foundational for Zone XX — you cannot debug agent failures in GTM tool-calling pipelines without reading the JSON-RPC messages."

---

## Beat 6: Ship It

**Deliverable that proves mastery.** Practitioner ships a self-contained MCP server (single Python file, stdio transport) that exposes at least one tool, one resource, and completes the full lifecycle (init handshake → operation → clean shutdown). The server must log all JSON-RPC traffic to a file for audit. Include a test script that sends the full lifecycle sequence and exits with code 0 only if every response matches expected shape. No SDK — raw JSON-RPC only, proving the practitioner understands the protocol beneath the abstractions.

---

### Learning Objectives (testable)

1. **Distinguish** a JSON-RPC request from a notification by identifying the presence or absence of the `id` field.
2. **Implement** a minimal MCP server that completes the `initialize` → `initialized` handshake and responds to `tools/list`.
3. **Compare** the three MCP primitives (tools, resources, prompts) by their effect on model context: action, information, instruction.
4. **Diagnose** a failed lifecycle negotiation by reading the JSON-RPC error response and identifying the missing or malformed field.
5. **Build** a stdio-based MCP server from raw JSON-RPC (no SDK) that exposes at least one tool and completes a clean shutdown.