## Ship It

Production agent systems need three things beyond the basic loop: a hard iteration cap, a token budget enforcement layer, and structured trace logging. The iteration cap prevents runaway costs from stuck models. The token budget prevents a single complex query from consuming a disproportionate share of your API spend. Trace logging lets you diagnose failures after the fact — you cannot debug an agent from its final output alone, you need the full thought-action-observation history.

Here is a production-ready wrapper that enforces both caps and emits structured traces:

```python
import json

def run_agent_with_budget(user_query, llm_fn, tools, max_iterations=8, max_tokens=8000):
    history = [{"role": "user", "content": user_query}]
    trace = []
    tokens = 0
    actions = []

    for i in range(max_iterations):
        prompt_tokens = sum(len(json.dumps(m)) for m in history)
        if tokens + prompt_tokens > max_tokens:
            trace.append({"iteration": i+1, "event": "token_budget_exceeded",
                         "tokens_used": tokens, "tokens_needed": prompt_tokens})
            return {"answer": None, "terminated_by": "token_budget",
                    "iterations": i, "tokens": tokens, "trace": trace, "actions": actions}

        response = llm_fn(history)
        tokens += prompt_tokens + len(json.dumps(response))

        tool_name = response["tool_call"]["name"]
        tool_args = response["tool_call"]["args"]
        actions.append(tool_name)

        entry = {
            "iteration": i + 1,
            "thought": response["thought"],
            "tool": tool_name,
            "args": tool_args,
            "tokens_so_far": tokens
        }

        if tool_name == "finish":
            entry["observation"] = tool_args.get("answer", "")
            entry["event"] = "finish"
            trace.append(entry)
            return {"answer": tool_args.get("answer"), "terminated_by": "finish_signal",
                    "iterations": i+1, "tokens": tokens, "trace": trace, "actions": actions}

        handler = tools.get(tool_name)
        if handler is None:
            entry["observation"] = f"ERROR: unknown tool {tool_name}"
            entry["event"] = "unknown_tool"
            trace.append(entry)
            return {"answer": None, "terminated_by": "unknown_tool",
                    "iterations": i+1, "tokens": tokens, "trace": trace, "actions": actions}

        try:
            raw = handler(**tool_args)
            observation = json.dumps(raw) if not isinstance(raw, str) else raw
        except Exception as e:
            observation = f"TOOL_ERROR: {type(e).__name__}: {e}"

        entry["observation"] = observation
        entry["event"] = "tool_executed"
        trace.append(entry)

        history.append({"role": "assistant", "content": response["thought"]})
        history.append({"role": "tool", "content": observation, "name": tool_name})

    return {"answer": None, "terminated_by": "iteration_cap",
            "iterations": max_iterations, "tokens": tokens, "trace": trace, "actions": actions}

def demo_llm(history):
    last_obs = None
    for msg in reversed(history):
        if msg["role"] == "tool":
            last_obs = msg["content"]
            break
    if last_obs is None:
        return {"thought": "Need to look up domain data.",
                "tool_call": {"name": "lookup_domain", "args": {"domain": "acme.com"}}}
    if "employees" in str(last_obs) and "budget" not in str(last_obs):
        return {"thought": "Got employee count, estimating budget.",
                "tool_call": {"name": "estimate_budget", "args": {"employees": 250}}}
    return {"thought": "Budget estimated.",
            "tool_call": {"name": "finish", "args": {"answer": "ACME budget: $300,000/yr"}}}

tools = {
    "lookup_domain": lambda domain: {"company": "ACME Corp", "employees": 250},
    "estimate_budget": lambda employees: {"budget": employees * 1200},
    "finish": lambda answer: answer
}

result = run_agent_with_budget(
    "Estimate ACME Corp's software budget",
    demo_llm,
    tools,
    max_iterations=8,
    max_tokens=8000
)

print("RESULT:")
print(json.dumps(result, indent=2))
print(f"\nTrace shows {len(result['trace'])} steps.")
print(f"Terminated by: {result['terminated_by']}")
print(f"Total tokens: {result['tokens']}")
```

The trace output is what you ship to your observability layer. Each entry records the thought, the tool invoked, the arguments, the observation, and the cumulative token count at that point in the loop. When an agent fails in production — wrong answer, runaway cost, silent timeout — the trace is the only artifact that tells you where the loop went wrong. Log it to whatever datastore your stack uses (Datadog, Langfuse, a simple JSON file). The iteration cap and token budget are configurable per-task: a simple lookup task might get `max_iterations=3`, while a multi-step research task gets `max_iterations=15`. The same budgeting logic applies to Clay enrichment workflows — you set a credit cap per row to prevent a single broken waterfall from draining your monthly allocation.

---