# MARL — MADDPG, QMIX, MAPPO

## Learning Objectives

- **Implement** a QMIX monotonic mixing network and verify the non-negativity constraint on the partial derivatives ∂Q_tot/∂Q_i.
- **Build** a MADDPG centralized critic that conditions on all agents' observations and actions, then compute TD loss across a simulated batch.
- **Compute** GAE advantages using a global-state critic for MAPPO, apply PPO clipping, and measure KL divergence between old and new policies.
- **Compare** the three factorizations of the multi-agent joint problem — independent actor-critic critics, monotonic value decomposition, and centralized-critic policy gradient — and articulate what each sacrifices.
- **Map** the CTDE training pattern onto a multi-agent GTM enrichment waterfall where coordination policies train centrally but execute with local context only.

## The Problem

Single-agent reinforcement learning assumes the environment is stationary. The transition function T(s'|s,a) does not depend on who is acting — it depends on the current state and the agent's action. This stationarity is what makes Bellman equations valid: the Q-value Q(s,a) is a fixed point because the dynamics that produced it are not shifting underneath the learner.

Add a second agent and the assumption breaks. Agent 1's environment now includes Agent 2, and Agent 2 is learning — its policy changes every episode, sometimes every step. From Agent 1's perspective, the transition function T(s'|s,a₁) is now T(s'|s,a₁,π₂(s)), where π₂ is a moving target. The Q-values Agent 1 computed last epoch are stale because the opponent (or teammate) behaves differently now. This is the non-stationarity problem, and it is the fundamental obstacle in Multi-Agent Reinforcement Learning (MARL).

Naive approaches fail in predictable ways. Independent Q-Learning (each agent runs its own DQN, treating other agents as part of the environment) suffers from chasing a