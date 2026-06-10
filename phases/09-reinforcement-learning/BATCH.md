# Phase 09 — Reinforcement Learning (quiz factory)

## Focus

Classical and deep RL: Markov decision processes, dynamic programming, temporal-difference learning, policy gradients (REINFORCE, PPO, GRPO), Q-learning and DQN, actor-critic methods, RLHF for LLMs, and reward modelling.

## Scrape hints

- `docs/en.md`: Bellman equations, return definitions, policy gradient derivations, value vs policy vs actor-critic distinctions
- `code/main.py`: often gymnasium/grid-world environments or pure-Python Q-table implementations
- Vocabulary: `glossary/terms.md` for return, episode, policy, value function, advantage, KL constraint

## Style anchor

- No gold quiz in this phase yet — use `phases/07-transformers-deep-dive/16-speculative-decoding/quiz.json` for structural reference
- pre = motivating failure (why naive supervised fails for sequential decisions), check = mechanism, post = code result or production deployment

## Common distractor patterns

- Confuse on-policy (PPO) with off-policy (DQN, SAC) data requirements
- Confuse reward vs return vs advantage (return = sum of future rewards, advantage = Q − V)
- Confuse REINFORCE (high variance, no baseline) with actor-critic (baseline via value function)
- Conflate RLHF reward modelling with direct preference optimisation (DPO — no separate reward model)
- Mix KL constraint in PPO clipping with KL penalty in RLHF fine-tuning

## Do not

- Import facts from other phases unless `docs/en.md` lists them as prerequisites.
- Ask the user questions — mark `blocked` in manifest instead.
