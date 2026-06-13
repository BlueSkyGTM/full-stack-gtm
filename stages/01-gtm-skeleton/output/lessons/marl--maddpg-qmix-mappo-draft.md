# MARL — MADDPG, QMIX, MAPPO

## Beat 1: Hook — Why Multi-Agent Changes Everything

Single-agent RL assumes a stationary environment. Add a second agent and the environment becomes non-stationary — each agent's policy shift changes the optimal response for every other agent. This is the core problem MADDPG, QMIX, and MAPPO solve: how to train policies when the world keeps rewriting itself.

## Beat 2: Concept — Centralized Training, Decentralized Execution (CTDE)

All three algorithms share one design pattern: give the learner full global information during training (observations and actions of all agents), then execute with local observations only. The mechanism that enforces this split — and what each algorithm sacrifices to make it work — is what differentiates them.

## Beat 3: Model — Three Factorizations of the Joint Problem

**MADDPG**: Each agent owns an independent actor (policy) and critic (value function). The critic conditions on all agents' observations and actions; the actor conditions on local observations only. Works for continuous action spaces. The mechanism is DDPG extended with per-agent critics that see the full picture during backprop.

**QMIX**: Factorizes the joint action-value function Q_tot into per-agent Q_i values, constrained by a monotonic mixing network — the partial derivative of Q_tot with respect to each Q_i is guaranteed non-negative. This means argmax over individual Q-values produces the joint argmax. Discrete actions only.

**MAPPO**: Extends PPO with a centralized critic that takes global state. The policy is still local. The mechanism is the standard PPO clipping objective, but the advantage estimates come from a value function that sees everything.

Compare-contrast anchor: MADDPG is actor-critic with full-info critics. QMIX is value-based with structured factorization. MAPPO is policy-based with centralized value. The trade is expressiveness (MADDPG) vs. guaranteed decentralized consistency (QMIX) vs. stability (MAPPO).

## Beat 4: Implement — Minimal Working Mechanisms

Each algorithm implemented small enough to run in a terminal, on a toy environment (e.g., a simple 2-agent grid or PettingZoo's `simple_spread`).

**Exercise hooks:**
- **Easy**: Run a pre-built QMIX mixing network on random tensors and verify the monotonicity constraint holds (check that ∂Q_tot/∂Q_i ≥ 0).
- **Medium**: Implement the MADDPG critic update for 2 agents — concatenate both agents' observations and actions, pass through a feedforward network, compute TD loss, print the loss value per step.
- **Hard**: Implement MAPPO's centralized advantage computation for 3 agents. Compute GAE using a global-state critic, clip the policy ratio, print KL divergence between old and new policy to confirm the update stays in trust region.

## Beat 5: Use It — GTM Redirect

**GTM Cluster**: Zone 5 — AI-Native Operations, specifically multi-agent orchestration for coordinated outreach and pipeline optimization.

The direct application: when you have multiple AI agents operating in a GTM stack — a research agent, a sequencing agent, a scoring agent — their actions are coupled. Sending 50 emails from agent A changes the response rate agent B sees. CTDE is the pattern for training these agents jointly in simulation (centralized) then deploying them independently (decentralized).

[CITATION NEEDED — concept: multi-agent GTM orchestration simulation environment]

**Exercise hooks:**
- **Easy**: Sketch the observation and action spaces for a 3-agent GTM system (research, outreach, scoring). Identify which observations are local and which would be available only during centralized training.
- **Medium**: Implement a QMIX mixing network that takes per-agent Q-values (one for research, one for outreach, one for scoring) and produces a joint Q-value. Feed it sample Q-values from a mock GTM scenario and print the joint value.

## Beat 6: Ship It — From Simulation to Production

MARL policies are notoriously hard to debug because performance degrades from non-stationarity, not from any single agent's failure. This beat covers monitoring: log per-agent reward, joint reward, and policy divergence from training to deployment. If any agent's policy drifts, the joint performance can collapse even if each agent looks fine in isolation.

**Exercise hooks:**
- **Easy**: Run a 2-agent MAPPO training loop for 100 episodes on `simple_spread`. Log per-agent reward and joint reward to CSV. Print the episode where joint reward first exceeds per-agent reward sum by >10% (emergent coordination signal).
- **Medium**: Take a trained QMIX policy. Freeze agent 1's policy, let agent 2 continue training for 50 episodes against the frozen policy. Print the performance drop — this demonstrates the non-stationarity problem in deployment when one agent's behavior changes.
- **Hard**: Implement a simple environment with 3 agents competing for a shared resource (representing competing outreach channels for the same lead pool). Train with MADDPG, then switch one agent to a hand-coded greedy policy. Measure and print the performance degradation of the other two learned agents.

---

## Learning Objectives (draft)

1. **Implement** the CTDE pattern: write a centralized critic that conditions on all agents' observations and a local actor that conditions on its own observation only.
2. **Compare** MADDPG, QMIX, and MAPPO on three axes: action space support, factorization constraint, and decentralized execution guarantee.
3. **Verify** QMIX's monotonicity constraint by computing partial derivatives of the mixing network's output with respect to each agent's Q-value.
4. **Detect** non-stationarity failure modes in deployed MARL systems by monitoring per-agent vs. joint reward divergence.
5. **Map** multi-agent coordination patterns to multi-channel GTM orchestration problems where agent actions are coupled through shared prospects.