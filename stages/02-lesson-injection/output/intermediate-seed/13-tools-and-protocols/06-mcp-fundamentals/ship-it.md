## Ship It

Production MCP servers need more than the minimal handshake. The deployment pattern for a GTM engineering team running MCP servers at scale involves three concerns: transport choice, authentication, and observability.

**Transport**: stdio is fine for local development (Claude Desktop spawning a child process). For production, HTTP with SSE (Server-Sent Events) is the transport that supports remote agents and multi-tenant deployments. The HTTP transport requires handling the `initialize` request as an HTTP POST and streaming responses back via SSE. The same JSON-RPC messages flow over the wire — only the transport framing changes.

**Authentication**: the 2025-11-25 spec added OAuth 2.1 as the standard auth mechanism for HTTP transports. A production MCP server wrapping a CRM API needs to validate the host's bearer token before processing any request. Without auth, the server is an open proxy to your HubSpot/Salesforce instance.

**Observability**: the stderr logging pattern from the build exercise is the baseline. In production, pipe those logs to structured JSON output and ship to your monitoring stack. The `id` correlation field in JSON-RPC is your trace ID — log it with each request/response pair so you can reconstruct multi-step agent calls.

From a GTM infrastructure perspective (Zone 13 — Deployment, CI/CD), an MCP server is deployed like any other internal service: containerized, behind a load balancer, with health checks that verify the `initialize` handshake completes. The deploy pipeline ships your MCP servers alongside your Clay tables and n8n workflows — they're all components of the same GTM automation stack. SPF/DKIM/DMARC is your email infrastructure layer; MCP is your agent-tool infrastructure layer.

```python
import json
import sys

HEALTH_CHECK = {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}}

def health_check(server_process):
    server_process.stdin.write(json.dumps(HEALTH_CHECK) + "\n")
    server_process.stdin.flush()
    response = server_process.stdout.readline()
    try:
        msg = json.loads(response)
        healthy = (
            msg.get("jsonrpc") == "2.0"
            and msg.get("id") == 0
            and "result" in msg
            and "protocolVersion" in msg["result"]
        )
        print(f"health: {'PASS' if healthy else 'FAIL'}")
        if healthy:
            print(f"  protocol: {msg['result']['protocolVersion']}")
            print(f"  server: {msg['result'].get('serverInfo', {}).get('name', 'unknown')}")
            caps = list(msg['result'].get('capabilities', {}).keys())
            print(f"  capabilities: {caps}")
        return healthy
    except (json.JSONDecodeError, KeyError):
        print("health: FAIL (invalid response)")
        return False

if __name__ == "__main__":
    import subprocess
    proc = subprocess.Popen(
        ["python3", "mcp_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    health_check(proc)
    proc.terminate()
```

```
health: PASS
  protocol: 2025-11-25
  server: minimal-mcp
  capabilities: ['tools', 'resources']
```

This health check sends a bare `initialize` request and verifies the server returns a valid JSON-RPC response with the expected fields. Wire it into your container's healthcheck endpoint or CI pipeline.