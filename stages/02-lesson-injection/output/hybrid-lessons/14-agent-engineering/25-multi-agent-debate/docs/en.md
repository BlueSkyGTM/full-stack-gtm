# Multi-Agent Debate and Collaboration

## Learning Objectives

- Implement a multi-agent debate loop where N agents independently propose, cross-critique, and converge over R rounds before returning a final answer with confidence.
- Compare full-mesh and sparse communication topologies by token cost, convergence speed, and output agreement.
- Build a task-router agent squad that decomposes a lead event into specialist subtasks (enrichment, research, drafting) with a critic gate that suppresses or escalates.
- Evaluate when multi-agent overhead is justified versus when a single-agent call produces equivalent output at lower cost.

## The Problem

A single LLM answering a question is a monologue. It generates, it commits, and unless you build an explicit self-refinement loop (Lesson 05), nothing pushes back. The model can hallucinate a confident answer because there is no opposing voice in the room. Self-Refine helps, but it is one model critiquing its own output — the same weights, the same biases, the same blind spots. CRITIC helps differently by grounding critique in external tools, but external verification is not always available, especially for judgment calls like "is this lead worth pursuing?"

Multi-agent debate introduces a structural fix: instead of asking one model to be right, you ask several independent instances to disagree with each other. Each agent proposes an answer, sees what the others said, critiques their reasoning, and revises. Over multiple rounds, agents either converge on a shared answer or reveal a persistent disagreement that signals low confidence. This is not a prompt engineering trick — it is a protocol change. You are restructuring the computation from a single forward pass into an iterative search where disagreement drives refinement.

The tradeoff is cost. A 3-agent, 2-round debate makes at least 6 LLM calls instead of 1, and each call's prompt grows as agents incorporate prior-round context. In a GTM context where every enrichment step costs Clay credits or API tokens, that overhead compounds fast across thousands of leads. The engineering question is not "does debate improve accuracy?" — Du et al. (ICML 2024) showed it does. The question is whether the accuracy gain justifies the 6–15x token cost for the specific decision you are automating.

## The Concept

Three mechanisms define multi-agent collaboration: **debate**, **collaboration**, and **suppression**. Understanding the difference between them — and when each applies — determines whether your agent architecture adds value or just burns tokens.

**Debate** runs N model instances that independently propose answers to the same question. Over R rounds, each agent reads the other agents' proposals, generates a critique of their reasoning, and submits a revised answer. After R rounds (or early consensus), a judge function returns the majority or converged answer with a