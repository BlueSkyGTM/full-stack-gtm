## Ship It

Production agent systems have multiple handlers, each with its own scope contract. An orchestrator receives every incoming task, checks it against each agent's contract in order, routes to the first match, or escalates if no contract claims the task. This is the pattern used in Zone 05 (Outbound and Inbound Execution) where you might have a research agent, a copy agent, a delivery scheduling agent, and a compliance review agent — all operating on the same pipeline. Without contracts, agents step on each other: the research agent drafts copy, the copy agent tries to schedule sends, the scheduling agent starts researching accounts. With contracts, each task routes to exactly one handler, and the routing is deterministic.

```python
RESEARCH_AGENT = {
    "agent_id": "research-v1",
    "IN_SCOPE": {"keywords": ["enrich", "lookup", "find", "research", "scrape", "identify", "discover"]},
    "OUT_OF_SCOPE": {"keywords": ["write", "draft", "compose", "schedule", "send", "score"]},
    "ESCALATION": {"keywords": ["acquire", "legal", "partnership"], "route_to": "human-review-queue"}
}

COPY_AGENT = {
    "agent_id": "copy-v1",
    "IN_SCOPE": {"keywords": ["write", "draft", "compose", "rewrite", "edit", "personalize"]},
    "OUT_OF_SCOPE": {"keywords": ["enrich", "lookup", "schedule", "send", "score", "find"]},
    "ESCALATION": {"keywords": ["legal", "compliance", "regulated"], "route_to": "compliance-review-queue"}
}

SCHEDULE_AGENT = {
    "agent_id": "schedule-v1",
    "IN_SCOPE": {"keywords": ["schedule", "calendar", "book", "time slot", "sequence timing"]},
    "OUT_OF_SCOPE": {"keywords": ["write", "enrich", "score", "find"]},
    "ESCALATION": {"keywords": ["holiday", "blackout", "legal hold"], "route_to": "ops-review-queue"}
}

SCORING_AGENT = {
    "agent_id": "scoring-v1",
    "IN_SCOPE": {"keywords": ["score", "rate", "rank", "grade", "tier", "evaluate fit"]},
    "OUT_OF_SCOPE": {"keywords": ["write", "draft", "schedule", "send", "enrich"]},
    "ESCALATION": {"keywords": ["acquire", "strategic", "partnership"], "route_to": "strategy-review-queue"}
}

AGENTS = [RESEARCH_AGENT, COPY_AGENT, SCHEDULE_AGENT, SCORING_AGENT]

def route_task(task_description, agents):
    task_lower = task_description.lower()
    trace = []

    for agent in agents:
        for kw in agent["ESCALATION"]["keywords"]:
            if kw in task_lower:
                trace.append(f"  ESCALATION match in {agent['agent_id']}: '{kw}'")
                return {
                    "task": task_description,
                    "routed_to": agent["ESCALATION"]["route_to"],
                    "decision": "ESCALATE",
                    "trace": trace
                }

    for agent in agents:
        for kw in agent["OUT_OF_SCOPE"]["keywords"]:
            if kw in task_lower:
                trace.append(f"  OUT_OF_SCOPE match in {agent['agent_id']}: '{kw}' (skip)")

    for agent in agents:
        for kw in agent["IN_SCOPE"]["keywords"]:
            if kw in task_lower:
                trace.append(f"  IN_SCOPE match in {agent['agent_id']}: '{kw}'")
                return {
                    "task": task_description,
                    "routed_to": agent["agent_id"],
                    "decision": "EXECUTE",
                    "trace": trace
                }

    trace.append("  No contract matched — defaulting to escalation")
    return {
        "task": task_description,
        "routed_to": "unmatched-task-queue",
        "decision": "ESCALATE",
        "trace": trace
    }

pipeline_tasks = [
    "Enrich this account with technographic data",
    "Write a personalized cold email for the VP of Sales",
    "Schedule the sequence to send Tuesday at 9am",
    "Score this lead against our ICP definition",
    "Evaluate whether we should acquire this company",
    "Research the company and then draft the outreach",
    "Compose a legal compliance review of our email template",
]

print("=" * 70)
print("MULTI-AGENT ORCHESTRATOR — SCOPE CONTRACT ROUTING")
print("=" * 70)
for task in pipeline_tasks:
    result = route_task(task, AGENTS)
    print(f"\n{'─' * 70}")
    print(f"TASK: {task}")
    print(f"DECISION: {result['decision']}")
    print(f"ROUTED TO: {result['routed_to']}")
    print(f"TRACE:")
    for line in result["trace"]:
        print(line)
print(f"\n{'─' * 70}")
print("ROUTING COMPLETE")
```

Note what happens with the compound task "Research the company and then draft the outreach." The orchestrator routes it to the research agent because "research" is checked first and matches. The "draft" component is ignored at routing time. This is intentional — compound tasks should be decomposed upstream, not ambiguously routed. The scope contract makes this ambiguity visible: the trace shows that the task matched one in-scope keyword while also containing an out-of-scope keyword for another agent. That visibility is the point. Without contracts, the compound task would have been silently handled by whichever agent got it first, with no record of the mismatch.