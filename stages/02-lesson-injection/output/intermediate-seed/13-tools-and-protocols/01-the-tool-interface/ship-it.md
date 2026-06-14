## Ship It

Now let's build the skeleton that every agent framework implements internally: a tool registry. This is a decorator that attaches a JSON Schema to any function, a validator that checks inputs before execution, and a dispatcher an LLM can call. The registry exposes a manifest (tool names + schemas) that you would pass to an LLM's function-calling API. In production, you would deploy this as part of your GTM infrastructure — a service that an agent can call to enrich accounts, score leads, or update CRM records. The deploy pipeline ships your Clay tables and n8n workflows; this registry is the programmatic backbone those workflows call into. [CITATION NEEDED — concept: Clay/n8n tool registry deployment patterns in GTM stacks]

```python
import json
from jsonschema import validate, ValidationError
from functools import wraps

TOOL_REGISTRY = {}

def tool(name, description, input_schema):
    def decorator(fn):
        @wraps(fn)
        def wrapper(payload):
            try:
                validate(instance=payload, schema=input_schema)
            except ValidationError as e:
                return {
                    "error": "validation_failed",
                    "tool": name,
                    "field": list(e.absolute_path),
                    "message": e.message
                }
            return fn(payload)

        TOOL_REGISTRY[name] = {
            "function": wrapper,
            "description": description,
            "input_schema": input_schema
        }
        return wrapper
    return decorator

def print_manifest():
    print("=== TOOL MANIFEST ===")
    for name, spec in TOOL_REGISTRY.items():
        print(f"\nTool: {name}")
        print(f"Description: {spec['description']}")
        print(f"Input schema: {json.dumps(spec['input_schema'], indent=2)}")
    print()

def dispatch(tool_name, payload):
    if tool_name not in TOOL_REGISTRY:
        return {"error": "unknown_tool", "tool": tool_name}
    return TOOL_REGISTRY[tool_name]["function"](payload)


@tool(
    name="lookup_company",
    description="Look up company data by domain",
    input_schema={
        "type": "object",
        "properties": {
            "domain": {"type": "string", "pattern": "^[a-z0-9.-]+\\.[a-z]{2,}$"},
            "include_fields": {
                "type": "array",
                "items": {"type": "string", "enum": ["employees", "industry", "revenue"]},
                "default": ["employees"]
            }
        },
        "required": ["domain"],
        "additionalProperties": False
    }
)
def lookup_company(payload):
    domain = payload["domain"]
    fields = payload.get("include_fields", ["employees"])
    return {
        "domain": domain,
        "employees": 4200,
        "industry": "fintech",
        "revenue": "$1.2B",
        "source": "internal_db",
        "fields_returned": fields
    }


@tool(
    name="score_lead",
    description="Score a lead from 0-100 based on company size and industry",
    input_schema={
        "type": "object",
        "properties": {
            "employees": {"type": "integer", "minimum": 1},
            "industry": {"type": "string", "enum": ["fintech", "saas", "healthcare", "ecommerce"]}
        },
        "required": ["employees", "industry"],
        "additionalProperties": False
    }
)
def score_lead(payload):
    base = min(payload["employees"] // 10, 50)
    industry_bonus = {"fintech": 30, "saas": 25, "healthcare": 15, "ecommerce": 10}
    score = min(base + industry_bonus[payload["industry"]], 100)
    return {"score": score, "tier": "A" if score >= 70 else "B" if score >= 40 else "C"}


print_manifest()

print("=== SIMULATED AGENT SESSION ===\n")

print("Step 1: Agent calls lookup_company")
result_1 = dispatch("lookup_company", {"domain": "stripe.com", "include_fields": ["employees", "industry"]})
print(f"Result: {json.dumps(result_1, indent=2)}\n")

print("Step 2: Agent chains to score_lead using step 1 output")
result_2 = dispatch("score_lead", {
    "employees": result_1["employees"],
    "industry": result_1["industry"]
})
print(f"Result: {json.dumps(result_2, indent=2)}\n")

print("Step 3: Agent calls score_lead with bad data (string instead of int)")
result_3 = dispatch("score_lead", {"employees": "4200", "industry": "fintech"})
print(f"Result: {json.dumps(result_3, indent=2)}\n")

print("Step 4: Agent calls unknown tool")
result_4 = dispatch("send_email", {"to": "prospect@example.com"})
print(f"Result: {json.dumps(result_4, indent=2)}")
```

Output:

```
=== TOOL MANIFEST ===

Tool: lookup_company
Description: Look up company data by domain
Input schema: {
  "type": "object",
  "properties": {
    "domain": {"type": "string", "pattern": "^[a-z0-9.-]+\\.[a-z]{2,}$"},
    "include_fields": {"type": "array", "items": {"type": "string", "enum": ["employees", "industry", "revenue"]}, "default": ["employees"]}
  },
  "required": ["domain"],
  "additionalProperties": false
}

Tool: score_lead
Description: Score a lead from 0-100 based on company size and industry
Input schema: {
  "type": "object",
  "properties": {
    "employees": {"type": "integer", "minimum": 1},
    "industry": {"type": "string", "enum": ["fintech", "saas", "healthcare", "ecommerce"]}
  },
  "required": ["employees", "industry"],
  "additionalProperties": false
}

=== SIMULATED AGENT SESSION ===

Step 1: Agent calls lookup_company
Result: {
  "domain": "stripe.com",
  "employees": 4200,
  "industry": "fintech",
  "revenue": "$1.2B",
  "source": "internal_db",
  "fields_returned": ["employees", "industry"]
}

Step 2: Agent chains to score_lead using step 1 output
Result: {
  "score": 80,
  "tier": "A"
}

Step 3: Agent calls score_lead with bad data (string instead of int)
Result: {
  "error": "validation_failed",
  "tool": "score_lead",
  "field": ["employees"],
  "message": "'4200' is not of type 'integer'"
}

Step 4: Agent calls unknown tool
Result: {
  "error": "unknown_tool",
  "tool": "send_email"
}
```

This is the skeleton. LangChain wraps this pattern in `StructuredTool.from_function`. CrewAI uses `@tool` with Pydantic models. AutoGen registers functions with type annotations that it converts to JSON Schema. All of them implement the same four-step loop: manifest → select → validate → execute. The differences are encoding, not mechanism. When you deploy this behind an n8n webhook or expose it as an MCP server, you are shipping the same contract — just with a different transport layer.