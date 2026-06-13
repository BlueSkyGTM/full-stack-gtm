# Reflexion: Verbal Reinforcement Learning

## Setup
Introduce the core problem: standard LLM agents fail, retry, and fail the same way because they have no memory of *why* they failed. Reflexion replaces scalar reward signals with natural-language self-critique stored as persistent context across attempts. Present the loop: act → evaluate → verbally critique → store critique → retry with critique in context.

## Mechanism
Explain the three components: the actor (generates a response), the evaluator (scores success or flags failure and produces a natural-language explanation), and the self-reflection module (writes a structured critique appended to a persistent memory buffer). Contrast with traditional RL: no gradient updates, no parameter changes — the "learning" is entirely in-context. Note the limitation: context window length caps how many reflection cycles accumulate before older critiques fall off.

## Code
Build a minimal Reflexion loop in Python. A function attempts a task (e.g., answering a multi-hop question), an evaluator checks correctness, and on failure the model generates a textual reflection that gets prepended to the next attempt's prompt. Print the reflection text and the attempt number so output is observable. After N attempts, print success or failure. All code runs in terminal with no browser dependency.

## Use It
GTM redirect: **Zone 2 – Enrichment** research agents. When an agent researches a prospect account and produces a summary that fails a factual-checking step, Reflexion lets the agent critique its own hallucination and retry with that critique in context — producing a corrected enrichment payload without retraining. Map this to enrichment waterfall failure handling in Clay or any multi-step research agent.

## Ship It
Exercise hook (easy): Run the provided Reflexion loop on 3 failing inputs and log which attempt number succeeded for each. Exercise hook (medium): Swap the evaluator from exact-match to a second LLM call that checks factual consistency, then measure how success rate changes. Exercise hook (hard): Add a token-budget limiter that truncates the reflection buffer when it exceeds a threshold, then compare performance with and without truncation across 10 trials.

## Evaluate
Three to five quiz questions grounded in the mechanism: identify which component produces the natural-language critique, predict what happens when the reflection buffer exceeds the context window, distinguish Reflexion from fine-tuning-based RL, explain why scalar rewards alone are insufficient for the self-correction pattern, and trace a two-cycle Reflexion loop given sample outputs. Each question maps directly to a learning objective in `docs/en.md`.