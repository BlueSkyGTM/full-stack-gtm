# Self-Refine and CRITIC: Iterative Output Improvement

## Learning Objectives

- Implement a Self-Refine loop with separate generate, feedback, and refine prompts that tracks history across iterations
- Compare internal LLM self-critique against external tool-based verification (CRITIC) and identify when each fails
- Build convergence detection into an iterative refinement loop using both score thresholds and delta-based stopping
- Map the evaluator-optimizer pattern to GTM personalization workflows and quantify the token cost per iteration

## The Problem

Your first LLM output is rarely your best. A single generate call gives you a draft — usually decent, sometimes hallucinated, almost always improvable. The question is how to close that gap without a human reviewing every output.

Human-in-the-loop review doesn't scale. If you're generating 500 personalized outreach emails, you cannot manually critique and revise each one. But throwing away the draft and regenerating wastes compute and produces unstructured variation. What you want is a structured loop: generate once, evaluate against criteria, revise specifically, repeat until the output is good enough.

Two papers formalize this. Self-Refine (Madaan et al., 2023) uses one LLM in three roles — generator, critic, refiner — with no external ground truth. CRITIC (Gou et al., 2023) keeps the same loop structure but routes the evaluation step through external tools: search engines, code interpreters, fact-checking APIs. The distinction matters because LLMs are better at discrim