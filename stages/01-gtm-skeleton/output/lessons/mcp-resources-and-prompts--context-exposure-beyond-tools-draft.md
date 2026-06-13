# MCP Resources and Prompts — Context Exposure Beyond Tools

## Hook

MCP servers expose three primitives. Tools get all the attention because they *do* things. But resources and prompts handle the read-path — structured data exposure and reusable prompt templates. Without them, every tool call hardcodes its own context. This lesson covers the two primitives that make MCP servers composable instead of one-off.

## Concept

**Resources** are server-exposed, client-consumed data items identified by URIs. They are read-only. A client (like Claude Desktop) can list available resources, read their contents, and subscribe to updates. Think of them as GET endpoints — the LLM asks for context, the server returns it. No execution, no side effects. URIs can be static (`config://app`) or templated (`user://{user_id}/profile`).

**Prompts** are named, parameterized prompt templates that servers expose. A client discovers them, fills in arguments, and receives a structured set of messages (user/assistant turns with content). Prompts standardize interactions across sessions. They are not executed server-side — the server returns the prompt text, and the LLM interprets it.

Mechanism distinction: Tools = action (POST). Resources = data (GET). Prompts = instruction (TEMPLATE).

[CITATION NEEDED — concept: MCP specification canonical reference for resources and prompts primitives]

## Use It

**GTM Cluster: Enrichment (Zone 3) — Account Intelligence Exposure**

Resources map directly to enrichment read-paths. An MCP server backed by a data warehouse exposes `account://{domain}/firmographic` as a resource. The LLM reads it to personalize outreach. No tool invocation needed — the context is just *there*.

Prompts map to reusable outreach templates. A server exposes `prompt://icp-research` with parameters `{company}`, `{title}`. The LLM receives a structured research prompt instead of the user re-typing it each time.

This is the enrichment-then-act pattern: resources provide the data, prompts frame the task, tools execute the action. Three primitives, three responsibilities.

[CITATION NEEDED — concept: GTM enrichment-to-outreach pipeline mapping to MCP resource→prompt→tool flow]

## Build It

**Easy:** Configure an MCP server with one static resource and one parameterized prompt. Read the resource and render the prompt from Claude Code CLI. Print the output.

**Medium:** Build a server with a templated resource (`account://{domain}/tech-stack`) that returns mock firmographic data. Create a prompt `research-account` that accepts `{domain}` and produces a multi-turn research brief. Wire it in `claude_desktop_config.json`. Call both from a session and print results.

**Hard:** Implement resource subscription (real-time updates via SSE). Expose a resource that changes state every 5 seconds. Have the client subscribe and log each update. Compare to polling a tool — print timing data for both approaches.

## Ship It

Deploy the enrichment MCP server from "Build It" as a local stdio server. Add it to your Claude Desktop config. Verify: (1) `list_resources` returns your firmographic URIs, (2) `list_prompts` returns your research template, (3) reading a resource and invoking a prompt in the same session produces context-aware output without hardcoded data in the prompt itself.

This is foundational for Zone 3 (Enrichment). The pattern — resources for read-only context, prompts for standardized task framing — replaces ad-hoc copy-paste enrichment workflows with composable server primitives.

## Evaluate It

- Compare three MCP primitives by side-effect profile and invocation pattern (tool vs resource vs prompt).
- Implement a static resource and read it from a client. Print the URI and content.
- Implement a parameterized prompt and render it with arguments. Print the resulting messages.
- Diagnose a failing resource URI — identify whether the issue is in URI template syntax, server registration, or client config.
- Decide when to use a resource vs a tool for data retrieval: defend the choice with the read-only/executable distinction.

---

**GTM Redirect Summary:**
- **Use It / Ship It:** Enrichment (Zone 3) — resource reads replace manual data lookup; prompt templates replace copy-paste research briefs. The three-primitive model (resource→prompt→tool) maps to enrichment→frame→act.