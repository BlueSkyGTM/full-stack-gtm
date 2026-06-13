# CrewAI: Role-Based Crews and Flows

## Learning Objectives

1. Configure a CrewAI Agent with role, goal, backstory, and an LLM backend; print its configuration to confirm assignment.
2. Define a Task with a description, expected output format, and agent binding; execute it and print the result.
3. Assemble a Crew of 2+ agents with a sequential process; run it and print each agent's contribution.
4. Implement a Flow that branches conditionally based on intermediate agent output.
5. Compare sequential vs. hierarchical crew processes and predict which produces lower-latency output for a given task graph.

---

## Beat 1: Hook

Single-prompt LLM calls collapse when your workflow demands distinct expertise—researcher, scorer, copywriter—in sequence. CrewAI implements a role-based orchestration pattern: agents with persistent identities collaborate through defined processes. This is not prompt chaining; each agent holds role context across its full execution cycle.

---

## Beat 2: Concept

**Mechanism: Role-Based Agent Collaboration**

An Agent in CrewAI is defined by four properties: role (job title), goal (what it optimizes for), backstory (context that shapes behavior), and llm (backend model). A Task binds a description and expected output to one agent. A Crew wires agents and tasks together through a process type—sequential (tasks run in order, each receiving prior output as context) or hierarchical (a manager agent delegates). Flows add stateful branching: the output of one crew becomes input to the next, with conditional logic determining the path.

The key distinction from raw prompt chains: agents maintain role identity and can invoke tools during execution. Tasks pass structured context between agents, not raw strings.

---

## Beat 3: Demonstration

Working example: a 3-agent crew that (1) researches a company from a URL, (2) scores it against ICP criteria, (3) drafts a personalized outreach message. Each agent's output prints to terminal. Uses `crewai` package with a local or API-backed LLM. Observable output at every stage.

```python
from crewai import Agent, Task, Crew, Process
import os

os.environ["OPENAI_API_KEY"] = "your-key-here"

researcher = Agent(
    role="Company Researcher",
    goal="Extract key business signals from company data",
    backstory="You are a B2B research analyst who identifies company stage, tech stack, and growth signals.",
    verbose=True
)

scorer = Agent(
    role="ICP Scorer",
    goal="Score companies against ideal customer profile criteria",
    backstory="You evaluate companies on firmographic fit: employee count, industry, funding stage, tech stack.",
    verbose=True
)

research_task = Task(
    description="Research the company at https://example.com and extract: industry, approximate employee count, tech stack signals, funding stage.",
    expected_output="A structured summary with 4 fields: industry, employees, tech_stack, funding_stage.",
    agent=researcher
)

score_task = Task(
    description="Using the research summary, score this company from 0-100 on ICP fit. Criteria: B2B SaaS (30pts), 50-500 employees (25pts), Series A-C (25pts), uses modern stack (20pts).",
    expected_output="A JSON object with fields: score, breakdown, recommendation.",
    agent=scorer
)

crew = Crew(
    agents=[researcher, scorer],
    tasks=[research_task, score_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()
print(result)
```

---

## Beat 4: Use It

**GTM Redirect: Zone 2 — Scout & Enrich (Waterfall Pattern)**

This maps directly to multi-signal enrichment workflows. Instead of a single enrichment call to one provider, a CrewAI crew can: (1) pull firmographic data, (2) pull technographic data, (3) cross-reference with ICP thresholds, (4) emit a scored record. This is the same waterfall pattern Clay implements for sequential data enrichment—CrewAI generalizes it to agent-based orchestration.

Concrete application: build a crew that takes a domain as input, enriches it through multiple data-gathering agents, and outputs a scored lead record ready for CRM insertion. [CITATION NEEDED — concept: CrewAI used in production GTM enrichment pipelines vs. Clay waterfall]

---

## Beat 5: Ship It

**Easy:** Configure a 2-agent crew (researcher + summarizer). Feed it a company domain. Print the final summary.

**Medium:** Build a crew that takes a list of 5 company domains, scores each against ICP criteria, and outputs structured JSON with scores. Sort by score descending. Print the ranked list.

**Hard:** Implement a Flow with conditional branching. Crew A scores a lead. If score ≥ 70, Crew B drafts personalized outreach. If score < 70, log the lead to a "nurture" list with a reason string. Print the routing decision and final output for each lead.

---

## Beat 6: Summary

CrewAI's mechanism: agents with persistent roles collaborate through tasks orchestrated by a process (sequential or hierarchical). Flows add stateful branching across crews. The trade-off is latency and token cost—each agent is a full LLM call cycle with role context maintained throughout. For GTM, this pattern generalizes the waterfall enrichment approach: instead of chaining API calls, you chain role-specialized agents that reason over each step's output.