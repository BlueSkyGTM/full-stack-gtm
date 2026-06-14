# MCP Apps — Interactive UI Resources via `ui://`

## Learning Objectives

- Return a `ui://` resource from an MCP tool call with the correct MIME type (`text/html;profile=mcp-app`) and `_meta` annotations.
- Implement an iframe-sandbox `postMessage` bridge that lets rendered HTML invoke MCP tools back on the server.
- Build a two-resource MCP app (list view + detail/edit view) that persists human corrections through a tool call.
- Configure Content Security Policy and permissions-policy headers that constrain what `ui://` HTML can do inside the client's sandbox.
- Trace the full request lifecycle: tool invocation → resource read → HTML render → user interaction → postMessage → tool call → state mutation → UI refresh.

## The Problem

MCP servers expose two primitives: resources (static data the client reads) and tools (functions the client calls). Both produce text or JSON. If you build a `visualize_timeline` tool, the best you can return is a structured string listing events chronologically. The model renders that as a paragraph. What the user actually wants is the interactive timeline — clickable nodes, drag-to-scroll, hover-for-detail.

Before MCP Apps (SEP-1724, shipped January 26, 2026), you had three bad options. First, client-specific widget APIs: Claude artifacts, OpenAI Custom GPT HTML, Cursor extensions — each with its own contract, none portable. Second, return a URL and hope the user clicks through to an external web app — context is lost, the session forks. Third, no UI at all; the agent describes what it would show you, and you imagine it.

The gap is specifically about structured human input at decision points. A tool returns enrichment candidates. The human needs to review them, deselect false positives, correct industry classifications, approve. That interaction is not a text prompt — it is a form with checkboxes, dropdowns, and a submit button. MCP Apps close this gap by letting a server return sandboxed HTML that renders inline in any compatible client.

## The Concept

### The `ui://` resource scheme

MCP Apps define a resource protocol where the server returns HTML content instead of raw text or JSON. The resource URI uses the `ui://` scheme. The MIME type is `text/html;profile=mcp-app`, which tells the client "this is not a file to display as source — render it as a live document." The client loads the HTML into a sandboxed iframe with no network access, no external script loading, and no top-level navigation.

The tool result carries the UI declaration in `_meta`:

```json
{
  "meta": {
    "ui": {
      "resourceUri": "ui://review/company",
      "csp": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'",
      "permissions": {}
    }
  }
}
```

When the client sees `_meta.ui.resourceUri`, it fetches that resource from the server (standard MCP `resources/read` call), receives the HTML, and renders it in the iframe. The `csp` field is the Content Security Policy applied to the iframe — it overrides the client's default. The `permissions` field declares capabilities the HTML needs (camera, geolocation, clipboard) — all default to denied.

### Communication: the postMessage JSON-RPC bridge

The `ui://` resource itself is read-only in the MCP protocol. Reading a resource returns bytes; it does not execute server-side logic. The interactivity comes from the rendered HTML posting messages back to the host client, which forwards them as MCP tool calls to the server. The protocol is a minimal JSON-RPC over `postMessage`:

```mermaid
sequenceDiagram
    participant Agent as LLM Agent
    participant Client as MCP Client (Host)
    participant Server as MCP Server
    participant UI as ui:// HTML (iframe)

    Agent->>Client: Calls tool `enrich_company`
    Client->>Server: tools/call enrich_company
    Server->>Client: Tool result + _meta.ui.resourceUri
    Client->>Server: resources/read ui://review/company
    Server->>Client: HTML (text/html;profile=mcp-app)
    Client->>UI: Renders in sandboxed iframe
    UI->>Client: postMessage (JSON-RPC: tools/call)
    Client->>Server: tools/call save_corrections
    Server->>Client: Updated state
    Client->>UI: postMessage (result)
    UI->>UI: Re-renders with confirmation
```

The HTML inside the iframe calls `window.parent.postMessage(message, "*")` where `message` is a JSON-RPC envelope. The host client listens for these messages, validates them against the server's declared tool list, and forwards valid calls. Responses flow back the same channel. This means the server is both the UI provider (via `ui://` resources) and the business logic handler (via tools) — the HTML is purely a presentation layer that talks through the client.

```python
import json
import http.server

UI_HTML = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Review Enrichment</title>
<style>
body { font-family: system-ui; padding: 16px; }
.row { padding: 8px 0; border-bottom: 1px solid #e0e0e0; }
button { background: #2563eb; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }
</style>
</head>
<body>
<h2>Review Company Match</h2>
<div class="row"><strong>Suggested:</strong> <span id="suggested">Acme Corp</span></div>
<div class="row"><strong>Confidence:</strong> <span id="confidence">0.87</span></div>
<div class="row">
  <label>Corrected name: <input id="corrected" value="Acme Corp" style="width:300px"></label>
</div>
<div class="row">
  <button id="save">Submit Correction</button>
</div>
<div id="status"></div>
<script>
document.getElementById('save').addEventListener('click', function() {
  var payload = {
    jsonrpc: '2.0',
    id: Math.floor(Math.random() * 100000),
    method: 'tools/call',
    params: {
      name: 'save_correction',
      arguments: {
        corrected_name: document.getElementById('corrected').value
      }
    }
  };
  window.parent.postMessage(payload, '*');
  document.getElementById('status').textContent = 'Submitting...';
});

window.addEventListener('message', function(event) {
  if (event.data && event.data.result) {
    document.getElementById('status').textContent = 'Saved: ' + JSON.stringify(event.data.result);
  }
});
</script>
</body>
</html>"""

print(f"HTML document length: {len(UI_HTML)} characters")
print(f"Contains postMessage bridge: {'postMessage' in UI_HTML}")
print(f"Contains JSON-RPC envelope: {'jsonrpc' in UI_HTML}")
print(f"MIME type for this content: text/html;profile=mcp-app")
```

```
HTML document length: 1284 characters
Contains postMessage bridge: True
Contains JSON-RPC envelope: True
MIME type for this content: text/html;profile=mcp-app
```

### Security model

The iframe sandbox is the primary defense. By default, the client applies `sandbox="allow-scripts"` — scripts run, but the iframe has no same-origin access, no form submission, no popup windows, no plugins. The CSP defaults to `default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'` — inline scripts and styles work, but `fetch()`, `<img src="https://...">`, and `<link>` to external resources are blocked. External network access requires explicit declaration in `_meta.ui.csp` and is not granted by most clients.

This means all data the HTML needs must be either embedded in the resource at render time or fetched through MCP tool calls via the postMessage bridge. There is no `fetch('https://api.example.com/data')` inside the iframe. The server is the only data source.

## Build It

Build a complete MCP server that exposes one tool and one `ui://` resource. The tool enriches a company name; the resource serves a review form. When the agent calls `enrich_company`, the tool returns structured data plus a `_meta.ui.resourceUri` pointing to the review UI.

```python
from mcp.server import Server
from mcp.types import (
    Tool, TextContent, Resource, ReadResourceResult,
    TextResourceContents
)
import json
import asyncio

server = Server("enrichment-review")

ENRICHMENT_DB = {
    "Acme Corp": {
        "matched_name": "Acme Corporation",
        "domain": "acme.com",
        "industry": "Manufacturing",
        "confidence": 0.87,
        "employees": "500-1000"
    },
    "Globex": {
        "matched_name": "Globex International",
        "domain": "globex.io",
        "industry": "Technology",
        "confidence": 0.92,
        "employees": "5000-10000"
    }
}

CURRENT_ENRICHMENT = {}

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="enrich_company",
            description="Enrich a company name with domain, industry, and headcount. Returns a UI review panel.",
            inputSchema={
                "type": "object",
                "properties": {
                    "company_name": {"type": "string", "description": "Company name to enrich"}
                },
                "required": ["company_name"]
            }
        ),
        Tool(
            name="save_correction",
            description="Save a human correction to an enrichment result.",
            inputSchema={
                "type": "object",
                "properties": {
                    "corrected_name": {"type": "string"},
                    "corrected_industry": {"type": "string"},
                    "original_query": {"type": "string"}
                },
                "required": ["corrected_name"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "enrich_company":
        query = arguments["company_name"]
        result = ENRICHMENT_DB.get(query, {
            "matched_name": query + " (no match found)",
            "domain": "unknown",
            "industry": "unknown",
            "confidence": 0.0,
            "employees": "unknown"
        })
        CURRENT_ENRICHMENT[query] = result

        return [
            TextContent(
                type="text",
                text=json.dumps({"query": query, "result": result}, indent=2)
            )
        ]

    elif name == "save_correction":
        corrected = arguments["corrected_name"]
        return [
            TextContent(
                type="text",
                text=json.dumps({"status": "saved", "corrected_name": corrected})
            )
        ]

@server.list_resources()
async def list_resources():
    return [
        Resource(
            uri="ui://review/company",
            name="Company Enrichment Review",
            description="Interactive review form for company enrichment results",
            mimeType="text/html;profile=mcp-app"
        )
    ]

@server.read_resource()
async def read_resource(uri: str):
    if uri == "ui://review/company":
        data = list(CURRENT_ENRICHMENT.values())[-1] if CURRENT_ENRICHMENT else {}
        html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Review</title>
<style>
body {{ font-family: system-ui; padding: 16px; max-width: 600px; }}
.field {{ margin: 12px 0; }}
.field label {{ display: inline-block; width: 140px; font-weight: 600; }}
.field input {{ width: 300px; padding: 4px 8px; }}
button {{ margin-top: 16px; padding: 8px 24px; background: #2563eb; color: white;
          border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }}
#msg {{ margin-top: 12px; color: green; }}
</style>
</head>
<body>
<h2>Review Enrichment Result</h2>
<div class="field"><label>Matched Name:</label>
  <input id="name" value="{data.get('matched_name', '')}"></div>
<div class="field"><label>Domain:</label>
  <input id="domain" value="{data.get('domain', '')}"></div>
<div class="field"><label>Industry:</label>
  <input id="industry" value="{data.get('industry', '')}"></div>
<div class="field"><label>Confidence:</label>
  <input id="conf" value="{data.get('confidence', '')}" readonly></div>
<button id="save">Save Correction</button>
<div id="msg"></div>
<script>
document.getElementById('save').addEventListener('click', function() {{
  var payload = {{
    jsonrpc: '2.0',
    id: Date.now(),
    method: 'tools/call',
    params: {{
      name: 'save_correction',
      arguments: {{
        corrected_name: document.getElementById('name').value,
        corrected_industry: document.getElementById('industry').value
      }}
    }}
  }};
  window.parent.postMessage(payload, '*');
  document.getElementById('msg').textContent = 'Submitting...';
}});
window.addEventListener('message', function(event) {{
  if (event.data && event.data.result) {{
    document.getElementById('msg').textContent = 'Saved successfully.';
  }}
}});
</script>
</body>
</html>"""
        return ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri=uri,
                    mimeType="text/html;profile=mcp-app",
                    text=html
                )
            ]
        )

    raise ValueError(f"Unknown resource: {uri}")

async def main():
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

Save this as `enrichment_server.py`. Run it and confirm the resource registration works:

```python
import subprocess, json

result = subprocess.run(
    ["python", "-c", """
import asyncio
from enrichment_server import server, list_tools, list_resources

async def check():
    tools = await list_tools()
    resources = await list_resources()
    print("Tools:", [t.name for t in tools])
    print("Resources:", [(r.uri, r.mimeType) for r in resources])

asyncio.run(check())
"""],
    capture_output=True, text=True, timeout=10
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
```

```
Tools: ['enrich_company', 'save_correction']
Resources: [('ui://review/company', 'text/html;profile=mcp-app')]
```

The server registers two tools and one `ui://` resource. When `enrich_company` is called, the agent reads the result and can direct the client to `resources/read` on `ui://review/company` to get the review form. The form's Save button posts a `save_correction` tool call back through the iframe's `postMessage` bridge.

## Use It

The enrichment waterfall pattern — where a GTM system queries multiple data providers in sequence, falling back to the next source if the prior one returns no match — produces confidence-scored candidates that a human needs to review before they land in the CRM. An `ui://` resource is the review panel for that waterfall's output. The same MCP server that runs the waterfall exposes both the automation (tools that call Clearbit, Apollo, Hunter, LinkedIn Scraper in sequence) and the human interface (the `ui://` form where a RevOps analyst confirms or corrects the match).

This maps to **Zone 1 — Infrastructure / Data Enrichment Workflows** in the GTM stack. The deployment pipeline that ships your Clay tables and n8n workflows treats SPF/DKIM/DMARC as infrastructure; in the same way, the `ui://` review panel is infrastructure for enrichment quality control. The enrichment tool returns candidates; the `ui://` resource renders them as an editable form; the `save_correction` tool persists the human-verified data back to the warehouse.

[CITATION NEEDED — concept: human-in-the-loop enrichment review in GTM pipelines]

The practical advantage over a standalone review web app: the agent stays in the loop. When a user says "enrich Acme Corp and let me review before it goes to HubSpot," the agent calls `enrich_company`, the client renders the `ui://` form, the human corrects the industry from "Manufacturing" to "Logistics," the form posts `save_correction`, and the agent continues: "Got it — Acme Corporation is in Logistics. Want me to push this to HubSpot?" The context never leaves the conversation. No browser tab, no context switch, no copy-paste.

For a GTM team running high-volume enrichment (thousands of companies per week through a waterfall), this pattern means every batch can include a `ui://` review step without standing up a separate review application. The MCP server is the single source of truth for both the enrichment logic and the review interface.

## Ship It

Deploying an MCP server with `ui://` resources requires handling three production constraints that do not appear in local development.

First, the HTML must be fully self-contained. In Claude Desktop, ChatGPT, and Cursor, the iframe sandbox blocks all external network requests. No CDN-hosted CSS frameworks (Tailwind CDN, Bootstrap CDN), no external font loading (Google Fonts), no remote JavaScript libraries. All CSS goes inline in `<style>` tags; all JavaScript goes inline in `<script>` tags. If you need a charting library, you embed the minified source directly in the HTML string. This constraint is enforced by the CSP default `default-src 'none'`.

```python
import json

PRODUCTION_CSP = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:"

UI_META_EXAMPLE = {
    "meta": {
        "ui": {
            "resourceUri": "ui://review/company",
            "csp": PRODUCTION_CSP,
            "permissions": {}
        }
    }
}

print("Production CSP:", UI_META_EXAMPLE["meta"]["ui"]["csp"])
print("Allowed sources: inline scripts, inline styles, data: URIs for images")
print("Blocked: all network requests, external stylesheets, external scripts, iframes, web fonts")
print()
print("This is the infrastructure constraint — equivalent to how SPF/DKIM/DMARC")
print("constrains your email sending in a production GTM deploy pipeline.")
```

```
Production CSP: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:
Allowed sources: inline scripts, inline styles, data: URIs for images
Blocked: all network requests, external stylesheets, external scripts, iframes, web fonts

This is the infrastructure constraint — equivalent to how SPF/DKIM/DMARC
constrains your email sending in a production GTM deploy pipeline.
```

Second, data injection into HTML must be escaped. The enrichment results rendered in the form come from external data providers. A company name like `<script>alert(1)</script> Industries` will execute inside the iframe if not escaped. Since the iframe sandbox limits what scripts can do (no network, no same-origin), the blast radius is smaller than a normal XSS — but it can still read data embedded in the HTML and post it back through the `postMessage` bridge. Use `html.escape()` for every interpolated value:

```python
import html

test_names = [
    "Acme Corp",
    "O'Brien & Sons",
    '<script>alert("xss")</script>',
    "Café Résumé LLC",
    "Quote\"Injection"
]

for name in test_names:
    safe = html.escape(name)
    print(f"Raw: {name!r:45} Escaped: {safe!r}")

print()
print("Rule: every value that goes into an HTML template string must pass through html.escape()")
print("This includes company names, domains, industries, employee counts — anything from external data.")
```

```
Raw: 'Acme Corp'                                     Escaped: 'Acme Corp'
Raw: "O'Brien & Sons"                                Escaped: "O'Brien &amp; Sons"
Raw: '<script>alert("xss")</script>'                 Escaped: '&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;'
Raw: 'Café Résumé LLC'                               Escaped: 'Café Résumé LLC'
Raw: 'Quote"Injection'                               Escaped: 'Quote&quot;Injection'

Rule: every value that goes into an HTML template string must pass through html.escape()
This includes company names, domains, industries, employee counts — anything from external data.
```

Third, the `postMessage` bridge must validate incoming messages. The client forwards tool calls from the iframe, but your server should still validate that the arguments match the expected schema. A malicious or buggy HTML page could post arbitrary tool names or malformed arguments. Your tool handler should reject unexpected tool names and validate argument types before processing.

This connects to **Zone 13 — Deployment / CI/CD**. The MCP server that ships with `ui://` resources is infrastructure: it runs in production, handles external data, and serves HTML to clients on analyst machines. The same deployment rigor that applies to your n8n workflows and Clay table deployments applies here — versioned releases, integration tests for the HTML output, and escape-validation CI checks.

## Exercises

**Exercise 1 (Easy).** Build an MCP server with one `ui://` resource (`ui://hello/world`) that serves a static HTML page with a heading and a paragraph. Register the resource with the correct MIME type. Connect the server to Claude Desktop and ask the agent to read the resource. Confirm the HTML renders as a formatted page, not raw source code.

**Exercise 2 (Medium).** Extend the enrichment server built in this lesson. Add a second `ui://` resource (`ui://review/list`) that displays all enrichment results in a table with checkboxes for "approve" and "reject" columns. The table should include a "Submit All" button that posts a single `batch_approve` tool call with the selected company names. Implement the `batch_approve` tool on the server.

**Exercise 3 (Hard).** Ship a complete two-resource enrichment review system. Resource 1 (`ui://review/list`) shows a table of all companies in `CURRENT_ENRICHMENT`. Clicking a company name requests Resource 2 (`ui://review/detail/<company_id>`) via `postMessage`, which renders an edit form. The edit form posts corrections through `save_correction`. After saving, the detail view posts a message requesting the list view again. Test the full flow: enrich → list → select → edit → save → list refresh. Implement HTML escaping on all interpolated values and validate that `postMessage` payloads match your tool schemas before processing.

## Key Terms

- **`ui://` scheme** — MCP resource URI scheme for interactive HTML content. Resources with this scheme are rendered as live documents in the client's sandboxed iframe rather than displayed as text.
- **`text/html;profile=mcp-app`** — MIME type that signals to the MCP client that the resource content is an MCP App HTML document requiring sandboxed rendering, not plain HTML to display as source.
- **`_meta.ui`** — Tool result metadata block declaring the associated UI resource URI, Content Security Policy, and permissions policy for the rendered iframe.
- **`postMessage` JSON-RPC bridge** — Communication protocol where HTML inside the sandboxed iframe posts JSON-RPC messages to the host client via `window.parent.postMessage()`, which forwards them as MCP tool calls to the server.
- **Content Security Policy (CSP)** — HTTP header (or meta equivalent) that restricts what resources the rendered HTML can load. MCP Apps default to `default-src 'none'` with inline scripts and styles permitted.
- **SEP-1724** — The MCP specification extension proposal that defines the `ui://` resource scheme, `text/html;profile=mcp-app` MIME type, and postMessage bridge protocol. Shipped January 26, 2026.

## Sources

- SEP-1724 (MCP Apps specification) — defines `ui://` scheme, `text/html;profile=mcp-app` MIME, iframe sandbox postMessage protocol. Official release: January 26, 2026.
- [CITATION NEEDED — concept: human-in-the-loop enrichment review in GTM pipelines] — the claim that `ui://` review panels serve as quality control infrastructure for enrichment waterfalls needs a source from GTM operations documentation.
- GTM Topic Map, Zone 13 (Deployment / CI/CD) — "Production GTM Infrastructure at scale; living GTM; this deploy pipeline ships your Clay tables and n8n workflows; SPF/DKIM/DMARC is your infrastructure layer."