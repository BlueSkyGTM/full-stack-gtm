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