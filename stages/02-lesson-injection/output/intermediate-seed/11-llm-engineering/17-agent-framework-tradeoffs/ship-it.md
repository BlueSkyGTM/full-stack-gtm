## Ship It

Production agent systems fail in ways that demos do not show. The four frameworks handle these failures differently, and the differences are architectural — you cannot patch over them with a wrapper.

**Retry granularity** is the first production concern. When the scoring LLM call times out, what do you retry? LangGraph retries from the failed node because the state graph knows the exact node and its input state. CrewAI retries the task, but the task may have implicit dependencies on prior task outputs that the framework passes as context — you are trusting the framework's context assembly. AutoGen retries by re-sending the last message, but the agent that receives it may produce a different reply because LLMs are non-deterministic. Agno retries the last tool call, which is clean for tool failures but ambiguous for reasoning failures (the model thought wrong, not the tool).

**Cost attribution** is the second. In a GTM pipeline processing thousands of prospects, you need to know which step consumes the most tokens. LangGraph nodes are named functions — you wrap each with a token counter and get per-node attribution. CrewAI tasks are named but share context implicitly, so token counts include accumulated context from prior tasks. AutoGen messages accumulate, and each agent reply includes the full conversation history — the last agent in a five-agent chat pays for all prior messages in its context window. Agno tool calls are individually attributable, but the reasoning between tool calls (the model's internal chain of thought) is a single block.

```python
print("=== Production: Retry + Cost Attribution ===")

from functools import wraps

token_usage = {}

def track_tokens(node_name):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            simulated_tokens = len(str(args)) + len(str(kwargs)) + 150
            token_usage[node_name] = token_usage.get(node_name, 0) + simulated_tokens
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if attempt == 1 and node_name == "score":
                        raise TimeoutError("LLM timeout (simulated)")
                    result = fn(*args, **kwargs)
                    return result
                except Exception as e:
                    print(f"  [{node_name}] attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
            return result
        return wrapper
    return decorator

@track_tokens("research")
def prod_research(company):
    return research(company)

@track_tokens("score")
def prod_score(data):
    return score(data)

@track_tokens("summarize")
def prod_summarize(r, s):
    return summarize(r, s)

print("Running pipeline with retry + token tracking...\n")
r = prod_research("Acme Corp")
s = prod_score(r)
m = prod_summarize(r, s)

print(f"\nFinal output: {m}")
print(f"\nToken usage by node:")
for node, tokens in token_usage.items():
    print(f"  {node}: {tokens} simulated tokens")
print(f"  Total: {sum(token_usage.values())} simulated tokens")

print("\n=== Retry Pattern Comparison ===")
print("LangGraph: retry from failed node. State checkpoint survives.")
print("CrewAI: retry task. Context reassembly may differ on retry.")
print("AutoGen: resend message. Non-deterministic reply possible.")
print("Agno: retry tool call. Clean for tool errors, messy for reasoning errors.")