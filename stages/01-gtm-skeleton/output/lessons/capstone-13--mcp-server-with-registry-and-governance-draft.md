# Capstone 13 — MCP Server with Registry and Governance

## Hook (Beat 1)

You've built MCP tools individually. Now the real problem: when an agent has access to 30 tools across enrichment, scoring, and outreach, how do you control which agent calls which tool, log every invocation, and prevent a runaway loop from burning through your Apollo API credits? This capstone builds the infrastructure that makes MCP safe to run in production GTM stacks.

## Concept (Beat 2)

Three mechanisms, layered:

**Tool Registry.** A central store that holds tool metadata—name, input schema, owning team, cost tier, required permissions. The registry is not the tool itself; it's the catalog the governance layer queries before allowing invocation. Without it, governance has nothing to reason about.

**Governance Middleware.** A function that sits between the agent's tool call and the actual tool execution. It checks: (1) does the caller have permission? (2) is the call within rate limits? (3) does the input pass validation beyond what JSON Schema provides? If any check fails, the call returns an error, not a tool result. This is the same pattern as API gateway middleware—intercept, evaluate, forward or reject.

**Audit Log.** Every invocation—permitted or rejected—writes to an append-only log with timestamp, caller identity, tool name, input hash, and outcome. This is not optional debugging; it's the record compliance and cost attribution require.

The architecture: `Agent → MCP Client → Governance Middleware → Registry Lookup → Tool Execution → Audit Write → Response`. The middleware is the gate. The registry is the reference data. The audit log is the evidence.

## Demo (Beat 3)

Build a working MCP server using the `@modelcontextprotocol/sdk` TypeScript package that:

1. Registers 3 tools (mock enrichment, scoring, and outreach tools) in an in-memory registry with metadata (permission level, cost units, rate limit)
2. Implements governance middleware that checks permissions and rate limits before each call
3. Logs every invocation (permitted and rejected) to a JSONL audit file
4. Returns governance-aware error messages when calls are rejected

The demo runs as a stdio MCP server. A test script invokes tools directly to show permitted calls, permission-denied calls, and rate-limited calls, then reads the audit log to confirm all were recorded.

Key observable output: the test script prints each tool call result and then dumps the audit log entries showing the full sequence including rejections.

## Use It (Beat 4)

**GTM Redirect: Zone 3 — Tool Orchestration, specifically the Clay waterfall and multi-tool enrichment chains.**

In production GTM, an agent might chain Clay's waterfall (enrich person → score → route to Slack). Each step calls a different API with different cost profiles. Governance prevents a single bad prompt from triggering 10,000 enrichment calls. The registry tracks which tools are GTM-critical versus experimental. The audit log lets you attribute API spend to specific campaigns or reps.

**Exercise hook (easy):** Register your existing enrichment tool from an earlier lesson in the registry with a cost tier and permission level. Confirm it appears in a registry listing endpoint.

**Exercise hook (medium):** Add a governance rule that blocks any tool call where the input contains a competitor's domain. Write a test that confirms the block works and the rejection appears in the audit log.

**Exercise hook (hard):** Implement per-caller rate limiting keyed by a caller ID passed in the tool metadata. Two callers share the same rate limit pool. Confirm that when caller A exhausts the limit, caller B is also blocked—and that the audit log distinguishes who triggered the exhaustion.

## Ship It (Beat 5)

**GTM Redirect: Zone 3 — Tool Orchestration with cost governance.**

Build a governed MCP server that wraps at least 2 real API tools (e.g., a person enrichment call and a Slack notification). The server must:

- Register both tools with different permission levels and cost units
- Enforce that "notification" tools require a higher permission level than "read" tools
- Rate-limit by caller identity
- Write audit logs that include enough context to reconstruct what happened without logging sensitive PII

Deliverable: the server code, a test script showing permitted and rejected calls, and the resulting audit log.

**Exercise hook (hard):** Extend the governance layer with a "dry run" mode. When enabled, the middleware evaluates all rules and logs the decision but does not execute the tool. The caller gets back what *would* have happened. This is the mechanism you need before letting agents loose on production CRM writes.

## Evaluate (Beat 6)

Quiz hooks grounded in observable behavior from the demo code:

1. **Registry mechanism:** What data structure does the registry expose to the governance middleware, and what happens if a tool is invoked that is not in the registry? (Testable: run the demo with an unregistered tool name and observe the error.)

2. **Governance ordering:** The middleware checks permissions before rate limits. What observable difference does reordering these checks make when a caller is both unauthorized *and* over rate limit? (Testable: swap the check order and compare error messages and audit entries.)

3. **Audit completeness:** How many audit log entries does a single successful tool invocation produce, and what fields distinguish it from a rejected invocation? (Testable: count entries after one permitted call versus one rejected call.)

4. **Rate limit shared state:** When two callers share a rate limit pool, what mechanism ensures the counter is shared? Predict what happens if the counter is per-caller instead, then modify the code and confirm. (Testable: change the keying and observe the behavioral difference.)

---

**GTM Cluster Mapping:** Zone 3 — Tool Orchestration. The registry + governance pattern is what separates demo MCP servers from production GTM infrastructure where cost control and audit trails are non-negotiable. [CITATION NEEDED — concept: MCP governance patterns in production GTM toolchains]