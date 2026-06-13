# A2A — Agent-to-Agent Protocol

## Hook

You have one agent that scrapes a prospect's website and another that drafts outreach. Right now you glue them together with a Python script and a prayer. A2A defines a standard protocol for agents to discover each other's capabilities, negotiate task handoffs, and exchange structured messages — without custom integration code for every pair.

## Concept

A2A is an open protocol published by Google that specifies how autonomous agents advertise what they can do (Agent Cards), how a client agent asks a remote agent to execute a task (Task Lifecycle), and how those agents exchange artifacts and status updates over HTTP+JSON. It sits one level above function calling: instead of a model invoking a tool it owns, it delegates to a separately deployed agent it does not control.

## Mechanism

Covers the three core primitives — Agent Cards (capability discovery via `.well-known/agent.json`), the Task lifecycle (submitted → working → completed/failed/canceled), and Part/Artifact message structure. Then covers the transport: standard HTTP POST with JSON-RPC–style payloads, streaming via Server-Sent Events, and push notifications via webhook callbacks. Compares A2A to MCP (MCP connects an agent to tools/data; A2A connects an agent to another agent) and to plain REST APIs (A2A standardizes the negotiation layer, not just the data layer).

## Use It

Walk through building two minimal A2A-compatible agents: one "research" agent that exposes an Agent Card and accepts a task to look up company data, and one "orchestrator" client that discovers the research agent, submits a task, polls for completion, and retrieves the artifact. Both agents run locally and communicate over HTTP. The practitioner sees the full request/response cycle with printed output at each stage.

**GTM redirect:** This is the mechanism behind multi-step GTM workflows where a research agent feeds a personalization agent — it maps to the enrichment-and-write pattern in Zone 02/03 enrichment pipelines, specifically the pattern where Clay's waterfall delegates to multiple data providers in sequence. [CITATION NEEDED — concept: Clay waterfall agent delegation pattern]

**Exercise hooks:**
- Easy: Run two provided agents locally, submit a task from the client, print the returned artifact.
- Medium: Add a second capability to the research agent's Agent Card and invoke it from the client.
- Hard: Chain three agents — research → scoring → draft — where each agent's output artifact becomes the next agent's task input.

## Ship It

Deploy one or both agents to a public endpoint (Render, Railway, or any HTTP host), verify the Agent Card is reachable at `/.well-known/agent.json`, and confirm a remote client can submit and complete a task. Covers HTTPS requirements, authentication via API keys in headers, and error handling for task failures.

**GTM redirect:** This is how a shared "enrichment agent" can serve multiple GTM workflows across a team — one deployed agent, many client orchestrators — mapping to Zone 03 enrichment infrastructure.

**Exercise hooks:**
- Easy: Deploy a single A2A agent and curl its Agent Card from your local machine.
- Medium: Deploy both agents, run the full task lifecycle remotely, print the final artifact.
- Hard: Add push-notification support so the client receives a webhook POST when the task completes instead of polling.

## Review

Checks whether the practitioner can distinguish A2A from MCP, explain the Task lifecycle states, read an Agent Card and identify capabilities, and trace a full request/response cycle from task submission to artifact retrieval. Confirms the practitioner can articulate where A2A fits in a GTM toolchain versus where a simpler API call suffices.