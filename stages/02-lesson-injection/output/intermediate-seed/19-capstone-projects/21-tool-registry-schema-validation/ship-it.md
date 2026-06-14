## Ship It

In a production GTM system, the registry has two jobs beyond validation: it generates the tool definitions you expose to the model, and it logs every call for observability. The tool definitions are derived directly from the registered input schemas — this is the same format OpenAI's function calling API expects, so the registry doubles as the source of truth for both validation and model-facing declarations.

The logging layer is where the registry earns its keep in production. Because every call passes through `call()`, you have a single chokepoint to record what tool was called, with what arguments, whether it passed validation, and what it returned. This is how you debug enrichment pipelines that silently produce bad data — you replay the call log through the validator and find which provider returned a shape that violated its output schema.

Here is the registry extended with tool-definition generation and call logging. The tool definitions are exportable in the format OpenAI's API expects, and every call is appended to a log you can replay:

```python
import json
from jsonschema import validate, ValidationError
from typing import Callable, Any
from datetime import datetime, timezone

class ProductionRegistry:
    def __init__(self):
        self._tools = {}
        self._call_log = []

    def register(self, name, description, handler, input_schema, output_schema, override=False):
        if name in self._tools and not override:
            raise ValueError(f"Tool '{name}' already registered")
        self._tools[name] = {
            "name": name,
            "description": description,
            "handler": handler,
            "input_schema": input_schema,
            "output_schema": output_schema,
        }

    def export_tool_definitions(self):
        defs = []
        for tool in self._tools.values():
            defs.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            })
        return defs

    def call(self, name, arguments):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": name,
            "input": arguments,
            "ok": None,
            "error": None,
            "output": None,
        }

        if name not in self._tools:
            entry["ok"] = False
            entry["error"] = f"Unknown tool: {name}"
            self._call_log.append(entry)
            return {"ok": False, "error": entry["error"]}

        tool = self._tools[name]

        try:
            validate(instance=arguments, schema=tool["input_schema"])
        except ValidationError as e:
            path = "/".join(str(p) for p in e.absolute_path) or "(root)"
            entry["ok"] = False
            entry["error"] = f"Input validation failed at '{path}': {e.message}"
            self._call_log.append(entry)
            return {"ok": False, "error": entry["error"]}

        try:
            result = tool["handler"](**arguments)
        except Exception as e:
            entry["ok"] = False
            entry["error"] = f"Handler error: {e}"
            self._call_log.append(entry)
            return {"ok": False, "error": entry["error"]}

        try:
            validate(instance=result, schema=tool["output_schema"])
        except ValidationError as e:
            path = "/".join(str(p) for p in e.absolute_path) or "(root)"
            entry["ok"] = False
            entry["error"] = f"Output validation failed at '{path}': {e.message}"
            self._call_log.append(entry)
            return {"ok": False, "error": entry["error"]}

        entry["ok"] = True
        entry["output"] = result
        self._call_log.append(entry)
        return {"ok": True, "result": result}

    def get_call_log(self):
        return self._call_log

    def replay_log(self, log=None):
        log = log or self._call_log
        results = []
        for entry in log:
            replayed = self.call(entry["tool"], entry["input"])
            results.append({
                "original_ok": entry["ok"],
                "replay_ok": replayed["ok"],
                "tool": entry["tool"],
                "match": entry["ok"] == replayed["ok"],
            })
        return results


def company_lookup(domain: str) -> dict:
    return {"name": "TechFlow", "employees": 120, "industry": "Fintech"}

def contact_search(name: str, company: str = "") -> dict:
    return {"email": f"{name.lower().replace(' ', '.')}@{company.lower() or 'example.com'}", "name": name}

prod = ProductionRegistry()

prod.register(
    name="lookup_company",
    description="Look up company data by domain.",
    handler=company_lookup,
    input_schema={
        "type": "object",
        "properties": {"domain": {"type": "string", "format": "uri"}},
        "required": ["domain"],
        "additionalProperties": False,
    },
    output_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "employees": {"type": "integer"},
            "industry": {"type": "string"},
        },
        "required": ["name", "employees", "industry"],
    },
)

prod.register(
    name="search_contact",
    description="Search for a contact by name.",
    handler=contact_search,
    input_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "company": {"type": "string"},
        },
        "required": ["name"],
        "additionalProperties": False,
    },
    output_schema={
        "type": "object",
        "properties": {"email": {"type": "string"}, "name": {"type": "string"}},
        "required": ["email", "name"],
    },
)

print("EXPORTED TOOL DEFINITIONS (for OpenAI API):")
print(json.dumps(prod.export_tool_definitions(), indent=2))

calls = [
    ("lookup_company", {"domain": "https://techflow.com"}),
    ("lookup_company", {"domain": "not-a-url"}),
    ("search_contact", {"name": "Sarah Chen", "company": "TechFlow"}),
    ("search_contact", {}),
]

print(f"\n{'='*60}")
print("EXECUTING CALLS:")
for tool_name, args in calls:
    result = prod.call(tool_name, args)
    status = "PASS" if result["ok"] else "FAIL"
    print(f"  [{status}] {tool_name}({json.dumps(args)})")
    if not result["ok"]:
        print(f"         {result['error']}")

print(f"\n{'='*60}")
print("CALL LOG:")
for entry in prod.get_call_log():
    status = "OK" if entry["ok"] else "ERR"
    print(f"  [{status}] {entry['timestamp'][:19]} | {entry['tool']} | {entry['error'] or 'succeeded'}")

print(f"\n{'='*60}")
print("REPLAY (re-validates all logged calls):")
for r in prod.replay_log():
    match = "MATCH" if r["match"] else "DRIFT"
    print(f"  [{match}] {r['tool']}: original={r['original_ok']}, replay={r['replay_ok']}")
```

The tool definitions section shows exactly what you would pass to the OpenAI API's `tools` parameter. The call log gives you a replayable audit trail. The replay feature re-runs every logged call through the validator — if a handler started returning different shapes after a deploy, the replay catches the drift. In a Clay-style enrichment pipeline, this replay capability is how you detect that a provider changed their API response format without updating their documentation.