# MCP Apps — Interactive UI Resources via `ui://`

## Hook

MCP servers can expose static data (resources) and callable functions (tools), but neither handles the case where you need a human to interact with structured input — forms, config panels, review screens. `ui://` resources close that gap by letting servers serve HTML that renders in the MCP client.

## Concept

The `ui://` URI scheme is a resource protocol where the MCP server returns HTML content instead of raw text or JSON. The client renders it in a sandboxed webview. The HTML can include forms, scripts, and styles — turning a resource read into a full interactive session. Server-sent events or postMessage bridges allow the UI to communicate back to the server.

## Mechanism

A server registers a resource handler for URIs matching `ui://<name>/<path>`. When the client requests that resource, the handler returns a `ReadResourceResult` with `mime_type: "text/html"`. The client detects the MIME type and renders it as a live document rather than displaying raw source. Communication back to the server flows through tool calls or a paired transport — the `ui://` resource itself is read-only in the MCP protocol, but the rendered HTML can invoke tools via the client's tool-calling interface.

**Exercise (Easy):** Serve a static HTML page from an `ui://` resource and confirm it renders in Claude Desktop.

## Use It

GTM redirect: **Zone 1 — Infrastructure / Data Enrichment Workflows.** An `ui://` resource can serve as a configuration panel for enrichment pipelines — a human reviews suggested company matches, corrects false positives, and submits. The same MCP server that runs the enrichment waterfall exposes the review UI as a resource. This is the review step in [CITATION NEEDED — concept: human-in-the-loop enrichment review] — the server provides both the automation (tools) and the review interface (`ui://`).

**Exercise (Medium):** Build an MCP server with one tool (`enrich_company`) and one `ui://` resource (`ui://review/company`) that displays the enrichment result as an editable form. The form submits corrections back via a tool call.

## Ship It

Deployment considerations: `ui://` HTML runs in a sandbox with no external network access in most clients. All data must be embedded at render time or fetched through MCP tool calls. CORS does not apply — the content is local to the client. The HTML must be self-contained: inline CSS, inline JS, no CDN imports.

**Exercise (Hard):** Ship a complete MCP server with two `ui://` resources (a list view and a detail/edit view), one tool for persisting edits, and confirm end-to-end flow: list → select → edit → save → list refresh.

## Extend It

`ui://` resources are read-only in the protocol — the interactivity comes from the rendered HTML calling back into MCP tools. This means the pattern generalizes: any workflow that needs human judgment at a decision point can expose a `ui://` review screen paired with a confirmation tool. Next step: multi-step wizards where each step is a separate `ui://` resource, and tool calls advance the state machine.

---
**GTM Cluster:** Zone 1 — Infrastructure, enrichment review workflows, human-in-the-loop configuration panels.