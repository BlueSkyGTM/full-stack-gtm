# MCP Gateways and Registries — Enterprise Control Planes

---

## Beat 1: Hook — The N+1 Server Problem

You have three AI agents, each needing five tools. That's fifteen MCP server processes if every agent runs its own. Now add ten more agents. The combinatorics break down — port exhaustion, credential sprawl, no visibility into what's calling what. This beat establishes why point-to-point MCP connections collapse at enterprise scale and introduces the control plane pattern as the architectural response.

---

## Beat 2: Concept — Registry, Gateway, Router

Three distinct mechanisms, often conflated. A **registry** is a service catalog: it stores server metadata, capability schemas, and health status — it answers "what exists?" A **gateway** is a proxy layer: it sits between MCP clients and servers, handling authentication, rate limiting, and request routing — it answers "who can reach what?" A **router** composes multiple servers behind a single endpoint, presenting one unified tool list to the client. This beat separates these concerns, diagrams the request flow (client → gateway → router → server), and explains where policy gets enforced at each hop.

---

## Beat 3: Demo — Stand Up a Local Registry and Gateway

Build a minimal registry that tracks MCP server definitions in a SQLite-backed catalog, then wire a gateway proxy that intercepts `tools/list` and `tools/call` requests, logs them, and forwards to registered servers. Observable output: a running proxy that accepts a tool call, prints the routing decision, and returns the result. Uses the MCP SDK transport layer directly — no browser, no UI, just terminal output confirming the gateway intercepted and routed the request.

---

## Beat 4: Use It — Centralized Tool Governance for GTM Stacks

GTM teams running Clay waterfalls, enrichment pipelines, and multi-agent outreach need a single control point for tool access. A registry lets you track which MCP servers expose which enrichment APIs (People Data Labs, Hunter, Clearbit). A gateway lets you rotate API keys without touching agent configs. This maps to **Zone 03 — Enrichment** and **Zone 04 — Activation**: the gateway pattern is how you govern which agents can trigger which enrichment or outreach actions, enforce rate limits across shared API quotas, and audit tool usage per campaign. [CITATION NEEDED — concept: enterprise MCP gateway production deployments in GTM stacks]

---

## Beat 5: Drill

**Easy:** Query the registry catalog to list all registered servers and their tool counts — print structured output.

**Medium:** Add a middleware layer to the gateway that rejects `tools/call` requests for servers marked `disabled` in the registry — log the rejection reason.

**Hard:** Implement request-level routing rules: direct `tools/call` for `enrichment`-tagged tools to Server A and `outreach`-tagged tools to Server B, with a fallback to a default server. Print the routing table and a trace for each routed call.

---

## Beat 6: Ship It — From Local Proxy to Shared Infrastructure

Checklist for graduating a local gateway to a shared team resource: add TLS termination, externalize the registry to a persistent store, implement server health checking with auto-deregistration, and add request logging that outputs structured JSON for downstream observability. The takeaway: a gateway is not a server — it's a policy enforcement point. Every MCP connection that passes through it becomes auditable, governable, and composable. Ship when you can answer "who called what, when, and did it succeed?" for every tool invocation.

---

## Learning Objectives (for `docs/en.md`)

1. **Compare** registry, gateway, and router roles in an MCP control plane architecture.
2. **Implement** a local MCP gateway that proxies and logs `tools/list` and `tools/call` requests.
3. **Configure** a registry catalog that tracks server metadata and health status.
4. **Evaluate** routing policies that restrict tool access based on server tags and client identity.
5. **Diagnose** request failures by tracing the gateway → router → server hop chain.