# Capstone 05 — Autonomous Research Agent (AI-Scientist Class)

## Learning Objectives

1. Implement a best-first tree search over experiments using planner, executor, and evaluator sub-agents that share a structured lab notebook.
2. Build a lab notebook data structure that accumulates evidence across research cycles and feeds prior findings back into the planner's context window.
3. Compare autonomous research loops against single-query ReAct agents and identify the specific conditions where self-directed convergence justifies the added complexity and cost.
4. Add cost limits, wall-clock budgets, and hypothesis-revisit circuit breakers to prevent infinite loops in self-directed agents.
5. Map the hypothesis-experiment loop to GTM account research workflows, converting the lab notebook into an ICP validation brief.

## The Problem

You have built single-purpose tools — classifiers, retrievers, extractors. You have built multi-step agents that chain tool calls to answer a query. Both architectures share a constraint: the human defines the task. The agent executes. This capstone removes that constraint. The system you are about to build defines its own hypotheses, decides which experiments to run, evaluates whether it has answered its own question, and either converges or revises — all without a human in the loop.

This architecture crossed a credibility threshold in 2025–2026. Sakana AI's AI-Scientist-v2 produced generated papers that cleared workshop peer review at machine learning venues [CITATION NEEDED — concept: AI-Scientist-v2, Sakana AI, peer review acceptance]. Agent Laboratory shipped reproducible experiment traces that other teams could rerun and verify [CITATION NEEDED — concept: Agent Laboratory, reproducible autonomous research traces]. ShinkaEvolve extended the pattern to evolving hypotheses that mutate across generations [CITATION NEEDED — concept: ShinkaEvolve, ICLR 2026]. These are not magic systems. They are a plan-execute-verify loop running over a tree of candidate experiments, with cost caps, sandboxed code execution, and an automated reviewer that rejects bad work.

The value of building one yourself is not in discovering something new on the first run. The value is in the infrastructure: the tree-search over experiments, the shared state that accumulates evidence, the evaluator that knows when to stop, and the guardrails that prevent the system from running forever or escaping its sandbox. You build the loop, you watch it think, and you understand exactly why it works and where it fails.

## The Concept

The autonomous research agent is a best-first tree search. Each node in the tree is an experiment specification: a hypothesis, a configuration, the code or queries to test it, and the expected outcome. The agent expands nodes by generating child experiments that probe the hypothesis from different angles. It selects the most promising child using a priority function — typically the evaluator's confidence that this experiment will reduce uncertainty about the research question. The search terminates when confidence exceeds a threshold or when the budget is exhausted.

Three sub-agents cooperate to drive the search. The **planner** reads the research question and accumulated evidence, then decomposes the question into a next testable hypothesis with a concrete experiment plan. The **executor** takes that plan and runs it — executing code, issuing searches, making comparisons — then writes structured observations to a shared lab notebook. The **evaluator** reads the notebook, judges whether the original question has been answered, assigns a confidence score, and either signals completion or feeds back to the planner with guidance on what remains uncertain.

```mermaid
flowchart TB
    RQ[Research Question] --> PL[Planner Agent]
    PL -->|hypothesis + plan| EX[Executor Agent]
    EX -->|structured observations| NB[(Lab Notebook)]
    NB --> EV[Evaluator Agent]
    EV -->{confidence >= threshold?}
    EV -->|No: gap identified| PL
    EV -->|Yes| DONE[Research Report]
    EV -->|budget exhausted| DONE
```

This is not a pipeline. The critical difference between this architecture and a linear chain-of-prompts is feedback. When the evaluator identifies a gap — "we tested sparsity at 50% but not at 70%, and the trend is