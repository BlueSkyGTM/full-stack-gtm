# Agent Instructions as Executable Constraints

## Hook (Beat 1)
When you give an agent a instruction like "be helpful," you've given it a wish. When you give it "respond with exactly one JSON object containing keys `company_name` and `confidence_score` between 0.0 and 1.0," you've given it an executable constraint. The difference is the gap between "mostly works" and "pipeline-grade."

## Concept (Beat 2)
Agent instructions operate as soft programs: they define input expectations, behavioral boundaries, output schemas, and failure modes using natural language that the model compiles into behavior at inference time. The mechanism is constraint propagation — each clause in an instruction narrows the sampling distribution of the model's output tokens. This beat introduces the constraint stack: role framing, input contracts, behavioral guardrails, output schemas, and fallback directives.

## Theory (Beat 3)
Why natural language constraints work at all — and why they fail. Covers the attention mechanism's role in instruction following, the recency bias that makes constraint ordering matter, the fidelity cliff (adding constraints improves output until it doesn't), and why structured output formats (JSON schema, function signatures) act as hard constraints versus soft prompt text. Includes observable failure modes: constraint conflicts, prompt drift in long contexts, and the "polite compliance" problem where agents acknowledge constraints they then violate.

## Use It (Beat 4)
In GTM workflows, agent instructions constrain research agents to extract only ICP-relevant signals, enrichment agents to return normalized company data, and outreach agents to stay within brand voice boundaries. [CITATION NEEDED — concept: Clay agent instruction formatting and constraint enforcement in enrichment workflows]. This beat shows how to write constraint stacks that turn a general-purpose agent into a domain-specific GTM tool — without fine-tuning.

## Build It (Beat 5)
Write, test, and iterate on a constraint stack for a company research agent. Exercise hooks:
- **Easy**: Write a constraint set that forces an agent to return company data in a specific JSON schema and reject ambiguous inputs with a structured error.
- **Medium**: Build a constraint stack where the agent classifies companies into ICP tiers (A/B/C) with explicit reasoning traces, then swap two constraint clauses and observe how output changes.
- **Hard**: Implement a multi-agent pipeline where one agent's output constraints become another agent's input contract — chain three agents and detect where constraint propagation breaks.

## Ship It (Beat 6)
Package a tested constraint stack into a reusable agent configuration file (YAML/JSON) with embedded test cases that validate constraint adherence. Covers versioning instruction sets, regression testing when model providers update, and monitoring for constraint drift in production. GTM redirect: this is how you operationalize agent instructions into repeatable enrichment and outreach workflows — the constraint stack becomes a config file your GTM team can modify without touching code.