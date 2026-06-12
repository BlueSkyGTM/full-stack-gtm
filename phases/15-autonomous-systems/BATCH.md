# Phase 15 — Autonomous Systems (quiz factory)

## Focus

Autonomous AI agents: reasoning loops (ReAct, Reflexion, chain-of-thought), memory architectures (episodic, semantic, working memory), planning and task decomposition, agent evaluation, safety constraints, and production agent frameworks.

## Scrape hints

- `docs/en.md`: loop diagrams (observe → think → act → observe), memory type distinctions, planning algorithm properties (tree search, beam, greedy), failure mode taxonomies
- `code/main.py`: typically implements a mini agent loop, memory store, or planner — quiz what the loop does and its failure modes
- Vocabulary: `glossary/terms.md` for ReAct, scratchpad, working memory, episodic memory, world model, task graph

## Style anchor

- No gold quiz in this phase yet — use `phases/10-llms-from-scratch/15-speculative-decoding-eagle3/quiz.json` for depth reference (that lesson tests a specific mechanism against its failure modes — same pattern here)
- pre = why a naive single-pass LLM call fails for multi-step tasks, check = specific mechanism, post = code loop or production deployment pattern

## Common distractor patterns

- Confuse ReAct (interleaved reasoning + action) with chain-of-thought-only (no external actions)
- Confuse episodic memory (past interaction log) with semantic memory (world knowledge / vector store)
- Confuse working memory (context window contents) with long-term memory (external store)
- Conflate planning (generating a task graph ahead of time) with reactive execution (deciding one step at a time)
- Mix agent self-correction (detect and fix own errors) with human-in-the-loop approval

## Do not

- Import facts from other phases unless `docs/en.md` lists them as prerequisites.
- Ask the user questions — mark `blocked` in manifest instead.
