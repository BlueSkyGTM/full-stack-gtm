# Phase 14 — Agent Engineering (quiz factory)

## Focus

Production-grade agentic systems: agent loops, tool use, multi-agent coordination,
failure modes, evaluation, and deployment patterns. Schema repair is the primary job —
all 42 quizzes need the extra `pre` dropped (7q → 6q) and a code-symbol reference added.

## Scrape hints

- `docs/en.md`: "## Build It" steps are the primary code-reference source; name specific
  functions like `AgentLoop`, `ToolRegistry`, `generate_handoff`, `PlanExecuteAgent.run`,
  `WorkbenchSnapshot`, pattern functions like `prompt_chain`, `route`, `evaluator_optimizer`
- `code/main.py`: look for class/function names in the Build section
- Lessons frequently reference prior agent patterns (Lessons 01–15 cover foundations);
  only test what this lesson's doc explicitly defines

## Repair pattern (schema_repair)

Current: `pre, pre, check, check, check, post, post` (7 questions)
Target: `pre, check, check, check, post, post` (6 questions)

1. Keep the stronger `pre`; drop the other (prefer the one harder to guess from the title).
2. Keep all three `check` questions and both `post` questions unchanged unless they fail
   quality criteria (wrong answer, fabricated distractor, or duplicate idea with another Q).
3. Verify at least one check or post names a specific function or class from `code/main.*`;
   rewrite the weakest check or post to add this if it's missing.

## Style anchor

- `phases/14-agent-engineering/26-failure-modes-agentic/quiz.json` (highest raw score in phase)
- `phases/07-transformers-deep-dive/16-speculative-decoding/quiz.json` (gold cross-phase)

## Common distractor patterns

- Confuse agent loop with a single LLM call
- Confuse tool schema with tool implementation
- Confuse handoff packet with a context window dump
- Confuse plan-and-execute with ReAct (both use steps, but plan-and-execute separates planning from execution)
- Confuse cascading failure with a single-step error
- Confuse evaluator-optimizer pattern with pure self-reflection
