## Ship It

Deploying an A2A agent to a public endpoint follows the same path as any HTTP service — with two protocol-specific requirements. First, the Agent Card must be reachable at `/.well-known/agent.json` on the deployed domain. Second, the task endpoint must accept POST requests with JSON-RPC payloads from external clients.

Deploy to Render or Railway by wrapping the HTTP handler in a WSGI-compatible entry point. The only code change is binding to `0.0.0.0` on the port the platform assigns:

```python
import os
from http.server import HTTPServer

port = int(os.environ.get("PORT", 8001))
host = os.environ.get("HOST", "0.0.0.0")

server = HTTPServer((host, port), ResearchAgentHandler)
print(f"[research-agent] listening on {host}:{port}")
server.serve_forever()
```

After deploy, verify the Agent Card is publicly reachable:

```python
import urllib.request
import json

deployed_url = "https://your-agent.onrender.com"
req = urllib.request.Request(f"{deployed_url}/.well-known/agent.json")
resp = urllib.request.urlopen(req)
card = json.loads(resp.read())

assert card["name"] == "research-agent", f"Unexpected agent: {card['name']}"
assert card["url"].startswith("https://"), f"Endpoint not HTTPS: {card['url']}"
print(f"[deploy] Agent Card verified at {deployed_url}")
print(f"[deploy] Endpoint: {card['url']}")
print(f"[deploy] Skills: {[s['id'] for s in card['skills']]}")
```

Output:

```
[deploy] Agent Card verified at https://your-agent.onrender.com
[deploy] Endpoint: https://your-agent.onrender.com/
[deploy] Skills: ['company_lookup']
```

Add authentication by checking for an API key in the `Authorization` header on every request. The Agent Card declares this requirement in its `securitySchemes` and `security` fields so clients know to send credentials:

```python
SECURE_AGENT_CARD = {
    "schemaVersion": "1.0",
    "name": "research-agent",
    "description": "Looks up company firmographics and tech stack data.",
    "url": "https://your-agent.onrender.com/",
    "version": "1.0.0",
    "capabilities": {"streaming": False, "pushNotifications": False, "stateTransitionHistory": True},
    "skills": [{
        "id": "company_lookup",
        "name": "Company Lookup",
        "description": "Given a company domain, returns employee count, industry, and tech stack.",
        "tags": ["research", "firmographics"]
    }],
    "securitySchemes": {
        "apiKey": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    },
    "security": [{"apiKey": []}],
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"]
}

def authenticate(handler):
    api_key = handler.headers.get("X-API-Key")
    expected = os.environ.get("AGENT_API_KEY", "dev-key-12345")
    if api_key != expected:
        handler._send_json(401, {"error": {"code": -32001, "message": "Invalid or missing API key"}})
        return False
    return True
```

Handle task failures gracefully. If the agent cannot process a task — bad input, internal error, timeout — it returns a task with `state: "failed"` and a message explaining the error. This lets the client agent decide whether to retry, fall back to a different agent, or surface the error to a human:

```python
def handle_task_error(self, task_id, error_msg):
    task = {
        "id": task_id,
        "state": "failed",
        "messages": [{
            "role": "agent",
            "parts": [{"kind": "text", "text": f"Task failed: {error_msg}"}]
        }]
    }
    tasks_store[task_id] = task
    return task

def _handle_send(self, params):
    task_id = params.get("taskId") or str(uuid.uuid4())
    message = params.get("message", {})
    text_parts = [p["text"] for p in message.get("parts", []) if p.get("kind") == "text"]

    if not text_parts:
        return self.handle_task_error(task_id, "No text input provided")

    domain = text_parts[0]
    if "." not in domain:
        return self.handle_task_error(task_id, f"Invalid domain format: {domain}")

    task = {"id": task_id, "state": "working", "messages": [message]}
    tasks_store[task_id] = task

    time.sleep(1)

    company_data = {
        "domain": domain,
        "employees": 450,
        "industry": "B2B SaaS",
        "tech_stack": ["Salesforce", "HubSpot", "Stripe"],
        "funding_round": "Series B",
        "last_raise": "$25M"
    }

    task["state"] = "completed"
    task["artifacts"] = [{
        "name": "company_profile",
        "parts": [{"kind": "text", "text": json.dumps(company_data)}]
    }]
    tasks_store[task_id] = task
    return task
```

This is how a shared enrichment agent can serve multiple GTM workflows across a team. One deployed agent with a public Agent Card becomes a reusable service — your outbound campaign tool calls it for lead enrichment, your CRM sync calls it for account updates, your ABM platform calls it for ICP scoring. Each consumer discovers the same Agent Card, authenticates with its own API key, and submits tasks independently. The agent's internal model, framework, and data sources stay opaque to all callers. This is the GTM infrastructure layer — the same pattern where SPF/DKIM/DMARC is your email infrastructure, the Agent Card is your enrichment infrastructure: a single endpoint with a published contract that any system can integrate against. [CITATION NEEDED — concept: A2A Agent Card as shared GTM enrichment infrastructure]