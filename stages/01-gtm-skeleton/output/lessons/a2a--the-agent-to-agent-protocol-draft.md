# A2A — The Agent-to-Agent Protocol

## GTM Redirect Rules

GTM cluster: **Multi-Agent Orchestration** — maps to Zone 2 (Enrichment) and Zone 3 (Outreach) in the GTM topic map. The redirect: "This is how you wire an enrichment agent to hand off to an outreach agent without a human-in-the-loop orchestrator." Where the concept is foundational infrastructure, the redirect says so plainly.

---

## Beat 1: Hook — Why This Matters Now

Single-agent systems hit a ceiling when one model must evaluate prospects, enrich records, draft outreach, and score replies. A2A is an open protocol (Google, April 2025) that defines how agents advertise capabilities, negotiate handoffs, and execute tasks for each other over HTTP+JSON. If you run multi-step GTM workflows, this protocol replaces bespoke webhook glue with a standard inter-agent contract.

---

## Beat 2: Concept — What A2A Actually Is

A2A defines three roles: **Client Agent** (initiates a task), **Remote Agent** (executes it), and the **Agent Card** (a JSON manifest advertising capabilities, auth, and input schemas). Communication flows over a single HTTP endpoint using JSON-RPC 2.0. No message queue, no websocket session — a request/response loop where the client sends a `Task` and gets back a stream of `TaskStatusUpdateEvent` or `TaskArtifactUpdateEvent` objects. The mechanism is closer to a structured API contract than a conversation.

---

## Beat 3: Mechanism — How the Protocol Works End-to-End

Walk through the four protocol operations: `SendTask`, `GetTask`, `CancelTask`, and the streaming variants. Show how the `Task` object tracks lifecycle states (`submitted → working → completed | failed | cancelled`). Demonstrate how an Agent Card is fetched via well-known URI (`/.well-known/agent.json`), parsed for supported `skills`, and matched against the client's intent. Cover push notification config for long-running tasks.

**Exercise hooks:**
- Easy: Fetch and parse a static Agent Card JSON, print the skills list.
- Medium: Implement a minimal Remote Agent that accepts a `SendTask` request and returns a completed Task with an artifact.
- Hard: Wire two agents — one generates a prospect summary, the second consumes that summary as a Task artifact and returns a scored output. Print the full task lifecycle.

---

## Beat 4: Use It — GTM Application

Wire an enrichment agent to an outreach-drafting agent. The enrichment agent acts as Client: it packages enriched prospect data into a Task, sends it to the outreach agent's A2A endpoint. The outreach agent returns a `TaskArtifactUpdateEvent` containing the drafted email. This replaces the manual handoff between enrichment and copy-generation steps in a Clay-style waterfall — the agents negotiate the schema themselves via the Agent Card.

**Exercise hooks:**
- Easy: Define an Agent Card for an enrichment agent that exposes a single `enrich_prospect` skill.
- Medium: Send a mock prospect record as a `SendTask` payload to a local outreach agent; print the returned artifact.
- Hard: Build both agents — enrichment fetches company data from a public API, outreach consumes it — and run the full round-trip with observable output at each state transition.

---

## Beat 5: Ship It — Production Considerations

Cover auth (the Agent Card declares supported auth schemes), error handling (`TaskState` failures with error messages), idempotency (task IDs must be unique, retries must not duplicate work), and observability (logging task state transitions). Address the current spec gaps: no formal registry for Agent Card discovery yet, no standardized rate-limit negotiation. [CITATION NEEDED — concept: A2A production deployment patterns and registry discovery].

**Exercise hooks:**
- Easy: Add authentication metadata to an Agent Card and validate it on the receiving end.
- Medium: Implement retry logic for a `SendTask` call that simulates a transient failure; confirm the task ID remains constant.
- Hard: Instrument both agents with structured logging of every task state transition; run a full workflow and print the log trace.

---

## Beat 6: Extend — Where This Goes Next

A2A is one layer in a stack: it handles agent-to-agent transport, while MCP (Model Context Protocol) handles agent-to-tool access. The two protocols compose — an agent can use MCP to call a tool, then use A2A to hand the result to another agent. The open questions: agent discovery at scale (registry vs. peer-to-peer), multi-hop task chains with rollback, and whether A2A adoption consolidates or fragments against alternative inter-agent protocols.

---

## Learning Objectives (Draft)

1. **Implement** a minimal A2A-compliant Remote Agent that responds to `SendTask` with a completed Task artifact.
2. **Parse** an Agent Card JSON manifest and extract declared skills and authentication requirements.
3. **Compare** A2A's request/response task lifecycle to MCP's tool-invocation pattern — identify when each is appropriate.
4. **Configure** a two-agent handoff where one agent's output artifact becomes the input to a second agent's task.
5. **Evaluate** the current A2A specification's limitations for production GTM workflows and identify where custom orchestration is still required.