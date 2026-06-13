# Agent Framework Tradeoffs — LangGraph vs CrewAI vs AutoGen vs Agno

## Beat 1: Hook

You will rebuild your agent system at least once. The rebuild happens because the framework you chose encoded assumptions about control flow, state, and agent identity that didn't match your actual workload. This lesson maps those assumptions explicitly so you pick based on mechanism, not README examples.

## Beat 2: Concept

Four frameworks, four architectural bets. LangGraph encodes a finite-state machine where you define every transition. CrewAI encodes roles and tasks where the framework resolves execution order. AutoGen encodes conversation loops between agents. Agno encodes a thin agent abstraction optimized for speed of instantiation. The tradeoff axis: **explicit control vs. declarative convenience vs. conversational flexibility vs. minimal overhead.** Each choice determines what you can observe, what you can retry, and what breaks in production.

**Mechanisms covered:**
- **State management:** Where does state live — in a graph object (LangGraph), in task outputs (CrewAI), in a conversation history (AutoGen), or in agent memory (Agno)?
- **Control flow resolution:** Who decides what runs next — the developer via edges (LangGraph), the framework via task dependency resolution (CrewAI), the agents via reply chains (AutoGen), or a linear tool-call loop (Agno)?
- **Agent identity:** Is an agent a function, a role description, a conversational participant, or a class instance? This affects testing, mocking, and cost attribution.

## Beat 3: Demonstration

Same task in all four frameworks: **research a company, score its ICP fit, draft a one-paragraph summary.** Three steps, one data pipeline. The implementations will show exactly where the frameworks diverge.

```
import json

task_spec = {
    "steps": ["research", "score", "summarize"],
    "input": {"company": "Acme Corp"},
    "expected_output": "str: summary paragraph with ICP score embedded"
}

def run_step(step_name, data):
    simulated = {
        "research": {"company": data["company"], "employees": 450, "industry": "fintech", "funding": "Series B"},
        "score": {"icp_score": 82, "reasons": ["right industry", "right size"]},
        "summarize": f"{data['company']} is a 450-person fintech company at Series B. ICP fit: 82/100."
    }
    return simulated[step_name]

result_research = run_step("research", task_spec["input"])
result_score = run_step("score", result_research)
result_summary = run_step("summarize", {**result_research, **result_score})
print(result_summary)
```

Then the same pipeline shown in LangGraph (explicit graph), CrewAI (agents + tasks), AutoGen (group chat), and Agno (Agent class). Code for each will run locally with mocked LLM calls to show control flow without API keys.

**Exercise hooks:**
- **Easy:** Modify the task spec to add a fourth step and trace how state passes through in each framework.
- **Medium:** Add a conditional branch — if ICP score < 50, skip summarize and output "reject." Implement in LangGraph and CrewAI.
- **Hard:** Add retry logic to the score step (simulated failure on first call). Compare how each framework handles recovery.

## Beat 4: Use It

**GTM redirect:** Multi-agent research pipelines map to **Zone 1B — Account Intelligence** [CITATION NEEDED — concept: multi-agent account research pipeline mapping to GTM zone]. Specifically, the research-score-summarize pattern from Beat 3 is the mechanism behind automated account qualification workflows.

When you build an account research agent that enriches, scores, and drafts an SDR outreach note, you are running the exact pipeline from this lesson. The framework choice determines whether you can: retry just the scoring step without re-running research (LangGraph: yes, CrewAI: partially, AutoGen: no cleanly, Agno: yes with manual wiring), observe intermediate state for debugging, and parallelize independent research calls.

**GTM-specific constraint:** In GTM workflows, cost attribution per account matters. Frameworks that treat agents as conversational participants (AutoGen) make it harder to attribute token spend to a specific account. Frameworks with explicit state (LangGraph) make it trivial.

```
def attribute_cost(account_id, step, tokens_used, cost_per_token=0.00003):
    record = {"account_id": account_id, "step": step, "tokens": tokens_used, "cost_usd": tokens_used * cost_per_token}
    print(json.dumps(record))
    return record

research_cost = attribute_cost("acme-corp", "research", 1200)
score_cost = attribute_cost("acme-corp", "score", 400)
summary_cost = attribute_cost("acme-corp", "summarize", 600)
```

## Beat 5: Ship It

**Production decision matrix:**

| Concern | LangGraph | CrewAI | AutoGen | Agno |
|---|---|---|---|---|
| Observability per step | Native | Partial | Opaque | Native |
| Retry granular step | Yes | Limited | No | Manual |
| Token cost attribution | Explicit | Aggregated | Per-conversation | Per-call |
| Cold start time | Slow | Medium | Slow | Fast |
| Learning curve | Steep | Shallow | Medium | Shallow |
| Breaks under... | Complex nesting | Custom tool logic | Agent disagreement | Multi-agent coordination |

**When to pick what:**
- **LangGraph** when you need audit trails, step-level retries, and deterministic routing — e.g., compliance-sensitive enrichment pipelines.
- **CrewAI** when you want fast prototyping of role-based workflows and can accept less granular control — e.g., content generation crews.
- **AutoGen** when the task is fundamentally conversational between agents — e.g., negotiation or debate patterns. [CITATION NEEDED — concept: production AutoGen deployment patterns]
- **Agno** when you need lightweight agent instantiation with minimal overhead — e.g., high-volume single-agent tool-calling loops.

**Warning:** If you cannot articulate why you need multi-agent orchestration, you do not need multi-agent orchestration. A well-structured function pipeline beats a poorly motivated agent graph every time.

```
def should_use_agents(task_description, requires_multiple_roles=False, requires_conversation=False, requires_dynamic_routing=False):
    reasons = []
    if requires_multiple_roles:
        reasons.append("multiple distinct roles with different tool access")
    if requires_conversation:
        reasons.append("agents must negotiate or debate")
    if requires_dynamic_routing:
        reasons.append("next step depends on LLM judgment, not predefined logic")
    if not reasons:
        print(f"NO — task '{task_description}' should use a function pipeline, not agents.")
        return False
    print(f"MAYBE — reasons: {reasons}. Validate that a simpler approach fails first.")
    return True

should_use_agents("enrich company and draft email")
should_use_agents("research, then let agents debate ICP fit", requires_conversation=True)
```

## Beat 6: Objectives & Assessment

**Learning objectives:**
1. **Compare** the control flow mechanisms of LangGraph, CrewAI, AutoGen, and Agno by identifying where state transitions are resolved in each.
2. **Implement** the same three-step pipeline in two frameworks and identify which supports granular retry of a single step.
3. **Evaluate** a production scenario (given requirements for observability, cost attribution, and retry behavior) and select the appropriate framework with explicit justification.
4. **Detect** when multi-agent orchestration is unnecessary by testing task requirements against the "should use agents" decision criteria.

**Quiz hooks (not full text):**
- Q1: Given a code snippet showing a state transition, identify which framework's control flow model it implements.
- Q2: A GTM pipeline needs per-step token cost attribution and step-level retries. Which two frameworks support this natively? Which does not?
- Q3: A task requires three agents to debate a classification before outputting a result. Which framework's conversational model fits this with the least custom wiring?
- Q4: Read a scenario description. Determine whether agents are justified or if a function pipeline suffices. Justify in one sentence.

---

**GTM cluster reference:** Zone 1B — Account Intelligence [CITATION NEEDED — concept: exact GTM cluster name for multi-agent account research in gtm-topic-map.md]