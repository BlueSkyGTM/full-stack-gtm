# Phase 16 — Multi-Agent and Swarms (quiz factory)

## Focus

Architectures where multiple AI agents collaborate or compete: orchestrator/sub-agent topologies, agent communication protocols, shared memory, consensus mechanisms, emergent behaviour, and production multi-agent frameworks (LangGraph, AutoGen, CrewAI).

## Scrape hints

- `docs/en.md`: topology diagrams (hub-and-spoke vs peer-to-peer vs hierarchical), message-passing semantics, conflict resolution strategies, convergence conditions
- `code/main.py`: often demonstrates a simple two-agent protocol or a shared message bus — quiz the protocol mechanics and failure modes
- Vocabulary: `glossary/terms.md` for orchestrator, sub-agent, shared state, consensus, broadcast, delegation

## Style anchor

- No gold quiz in this phase yet — use `phases/07-transformers-deep-dive/15-attention-variants/quiz.json` for structural reference
- pre = why a single agent fails at tasks that require parallel specialisation, check = topology or protocol mechanism, post = code demo or real-world deployment pattern

## Common distractor patterns

- Confuse orchestrator (directs sub-agents) with peer-to-peer coordination (no central authority)
- Confuse shared memory (all agents read/write a common store) with message passing (point-to-point)
- Conflate emergent coordination (not explicitly programmed) with programmed choreography
- Mix sub-agent specialisation (each agent has a narrow skill) with ensembling (same task, majority vote)
- Confuse multi-agent debate (adversarial) with multi-agent collaboration (additive)

## Do not

- Import facts from other phases unless `docs/en.md` lists them as prerequisites.
- Ask the user questions — mark `blocked` in manifest instead.
