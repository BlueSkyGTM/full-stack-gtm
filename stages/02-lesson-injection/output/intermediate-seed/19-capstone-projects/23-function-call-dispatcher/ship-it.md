## Ship It

Production dispatchers need three things the toy version above does not have: timeouts, idempotency, and permission boundaries. Timeouts prevent a slow handler from hanging the entire agent loop — if your `lookup_company` handler calls a third-party enrichment API that takes 30 seconds, you need `asyncio.wait_for` or a thread timeout to kill it and return an error. Idempotency prevents duplicate side effects when the LLM retries a tool call that actually succeeded but whose response was slow to arrive. Permission boundaries prevent an agent from calling tools it should not have access to in the current session.

Here is a production dispatcher with permission checks and call tracing:

```python
import json
import time

REGISTRY = {}

def tool(name, required_role="viewer"):
    def decorator(fn):
        REGISTRY[name] = {"handler": fn, "required_role": required_role}
        return fn
    return decorator

@tool("lookup_company", required_role="viewer")
def lookup_company(domain: str) -> str:
    companies = {
        "acme.com": {"name": "Acme Corp", "employees": 250},
        "techflow.io": {"name": "TechFlow", "employees": 45},
    }
    return json.dumps(companies.get(domain, {"error": "not found"}))

@tool("score_lead", required_role="editor")
def score_lead(company: str, signal: str) -> str:
    return json.dumps({"company": company, "score": 85, "grade": "A"})

@tool("delete_lead", required_role="admin")
def delete_lead(lead_id: str) -> str:
    return json.dumps({"deleted": lead_id})

CALL_LOG = []

def dispatch(tool_call, session_role="viewer"):
    name = tool_call["function"]["name"]
    args = json.loads(tool_call["function"]["arguments"])
    t0 = time.time()

    if name not in REGISTRY:
        result = {"error": f"Unknown tool: {name}"}
        CALL_LOG.append({"tool": name, "args": args, "result": result, "latency_ms": 0, "rejected": "unknown_tool"})
        return json.dumps(result)

    entry = REGISTRY[name]
    if session_role != entry["required_role"] and not (
        session_role == "admin" or
        (session_role == "editor" and entry["required_role"] == "viewer")
    ):
        result = {"error": f"Permission denied: requires role '{entry['required_role']}', session has '{session_role}'"}
        CALL_LOG.append({"tool": name, "args": args, "result": result, "latency_ms": 0, "rejected": "permission"})
        return json.dumps(result)

    try:
        raw = entry["handler"](**args)
        latency = round((time.time() - t0) * 1000, 1)
        CALL_LOG.append({"tool": name, "args": args, "result": json.loads(raw), "latency_ms": latency, "rejected": None})
        return raw
    except Exception as e:
        latency = round((time.time() - t0) * 1000, 1)
        result = {"error": f"{type(e).__name__}: {str(e)}"}
        CALL_LOG.append({"tool": name, "args": args, "result": result, "latency_ms": latency, "rejected": "exception"})
        return json.dumps(result)

viewer_session = "viewer"
calls = [
    {"function": {"name": "lookup_company", "arguments": '{"domain": "techflow.io"}'}},
    {"function": {"name": "score_lead", "arguments": '{"company": "TechFlow", "signal": "demo"}'}},
    {"function": {"name": "delete_lead", "arguments": '{"lead_id": "123"}'}},
]

print("=== Viewer session ===")
for tc in calls:
    result = dispatch(tc, session_role=viewer_session)
    print(f"  {tc['function']['name']:20s} -> {result}")

print("\n=== Call log ===")
for entry in CALL_LOG:
    status = "OK" if entry["rejected"] is None else f"REJECTED({entry['rejected']})"
    print(f"  {entry['tool']:20s} {entry['latency_ms']:6.1f}ms  {status}")
```

```
=== Viewer session ===
  lookup_company      -> {"name": "TechFlow", "employees": 45}
  score_lead          -> {"error": "Permission denied: requires role 'editor', session has 'viewer'"}
  delete_lead         -> {"error": "Permission denied: requires role 'admin', session has 'viewer'"}

=== Call log ===
  lookup_company         0.0ms  OK
  score_lead             0.0ms  REJECTED(permission)
  delete_lead            0.0ms  REJECTED(permission)
```

The viewer session can look up companies but cannot score or delete leads. The call log records every dispatch — tool name, arguments, latency, rejection reason — so you can audit what the agent tried to do. In a GTM context, this matters when an agent has write access to a CRM and you need to prove it only modified records it was authorized to touch.

The hand-rolled dispatcher you built here maps cleanly onto OpenAI's tool-calling API surface. OpenAI gives you `tools` (your registry schemas), `tool_calls` in the response (what your dispatch function parses), and `tool` role messages (how you feed results back). The difference is that OpenAI's API defines the wire format; your dispatcher defines the execution policy. The API does not enforce timeouts, does not check permissions, does not log calls. That is your code, on your machine, at the seam where the LLM's decision becomes your system's action.