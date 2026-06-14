# ReWOO and Plan-and-Execute: Decoupled Planning

## Learning Objectives

- Implement a Plan-and-Execute agent that generates a task list, executes steps sequentially, and conditionally replans based on tool outputs
- Implement a ReWOO agent with planner, worker, and solver roles using dependency-ordered slot variables
- Compare token cost, wall-clock time, and failure handling between interleaved ReAct loops and decoupled planning patterns
- Map decoupled planning patterns to enrichment waterfalls and multi-source account research in GTM workflows
- Diagnose when a static plan (ReWOO) is sufficient versus when replanning (Plan-and-Execute) is required for a given task

## The Problem

ReAct interleaves thinking and acting in a single stream. Each tool call feeds its observation back into the next reasoning step, which means every subsequent thought carries the accumulated context of all previous thoughts and observations. Token usage grows roughly quadratically with chain depth — a 10-step ReAct loop might pass 50K tokens through the model, most of it redundant context the model already reasoned about three steps ago.

The deeper problem is brittleness. When a tool fails mid-loop — returns null, times out, or returns garbage — the model has to re-derive its entire plan from the error observation. It never had a plan to begin with; it was improvising step by step. An error at step 7 means the model might restart reasoning from scratch, or worse, hallucinate a plausible-looking continuation from bad data. Every observation pollutes the reasoning context for every subsequent step, and there is no isolation between "what did I plan to do" and "what did I actually find."

In a GTM context, this cost structure maps directly to enrichment pipelines. Every provider call in a ReAct-style sequential chain pays the full LLM context overhead — the model re-reads every prior enrichment result before deciding the next step. When you are enriching 10,000 companies across three data providers, that overhead compounds into thousands of unnecessary LLM calls. Zone 14 of the GTM stack treats every credit as a cost to optimize, not an expense to accept [CITATION NEEDED — concept: Zone 14 cost optimization framing, 80/20 GTM Engineer Handbook].

## The Concept

Two mechanisms decouple planning from execution. Both answer the same question: what if the planner never sees tool observations until the very end?

**Plan-and-Execute** splits the work into a planner LLM and an executor LLM. The planner produces an ordered list of steps. The executor runs each step, passing results forward. After each execution, a replanner can revise remaining steps based on what the last tool returned. LangGraph ships a reference implementation of this pattern. The replanner is what makes it adaptive — if step 2 returns "email not found," the replanner can add a new step to try an alternative source before