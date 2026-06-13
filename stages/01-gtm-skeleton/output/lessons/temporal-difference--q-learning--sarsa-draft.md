# Temporal Difference — Q-Learning & SARSA

## GTM Redirect Rules

Primary GTM cluster: **Zone 2 — Enrichment Waterfalls**. The temporal difference mechanism maps directly to sequential enrichment decisions where each step's outcome informs the next action. Specifically: the Clay waterfall implements a greedy policy over enrichment providers, and TD methods formalize the feedback loop that determines provider ordering and fallback logic.

---

## Beat 1: Define It

Introduce temporal difference learning as the mechanism that bootstraps value estimates from incomplete episodes—combining Monte Carlo sampling with dynamic programming's incremental updates. Distinguish the two control algorithms: Q-learning learns the optimal action-value function regardless of the agent's actual behavior (off-policy), while SARSA evaluates the action-value function for the policy the agent is currently executing (on-policy). State the update rules for both and identify the single term that differentiates them (max vs. actual next action).

## Beat 2: See It

Walk through a step-by-step trace of both algorithms on the same 4×4 gridworld with a single terminal state and one "cliff" penalty cell. Show identical episodes where Q-learning learns to approach the goal adjacent to the cliff (because it ignores the exploring policy's occasional cliff-falls), while SARSA learns a safer path. Display the divergence in learned Q-tables after 50 episodes side by side.

Exercise hooks:
- **Easy**: Trace 3 updates of Q-learning by hand given a state-action-reward sequence.
- **Medium**: Predict which algorithm produces higher average reward during training vs. after convergence.
- **Hard**: Modify the reward structure to force Q-learning and SARSA to converge to the same policy—explain why.

## Beat 3: Code It

Implement both algorithms from scratch against a minimal `Environment` interface (`reset()`, `step(action)`, `render()`) using only a dictionary-based Q-table. Run both on the same deterministic gridworld for 500 episodes with decaying epsilon. Print the learned policy as a grid of arrows and the per-episode reward totals so the practitioner can observe convergence and behavioral differences.

Exercise hooks:
- **Easy**: Add a second penalty cell and rerun—report how both policies change.
- **Medium**: Replace the decaying epsilon with a fixed epsilon of 0.1 and compare convergence speed.
- **Hard**: Implement double Q-learning alongside the existing two algorithms and benchmark all three on an environment with stochastic transitions.

## Beat 4: Debug It

Cover the three most common failure modes: (1) learning rate that doesn't satisfy the Robbins-Monro conditions causing oscillation instead of convergence, (2) reward shaping that introduces unintended optimal policies where the agent loops infinitely, (3) state representation that causes the Q-table to balloon because environment features are treated as distinct states when they should be generalized. Show symptom → diagnosis → fix for each.

Exercise hooks:
- **Easy**: Identify which Robbins-Monro condition is violated given a specific learning rate schedule.
- **Medium**: Given a reward function that produces an infinite-loop policy, modify the rewards to eliminate the loop without changing the environment dynamics.
- **Hard**: Debug a Q-learning implementation that converges on a 4×4 grid but diverges on a 10×10 grid—diagnose whether the issue is exploration, learning rate, or state representation.

## Beat 5: Use It

Map the TD mechanism to the Clay enrichment waterfall. Each enrichment provider is an "action," the current data state is the "state," and success/failure of a provider returning data is the "reward." Q-learning formalizes the feedback loop that determines provider ordering: the waterfall learns which provider to call first based on accumulated outcome history, then which to call second if the first fails. Show how the off-policy property of Q-learning means the waterfall can learn the optimal provider sequence even while A/B testing suboptimal orderings. SARSA applies when the business requires conservative fallback behavior—learning the actual deployed policy's value rather than the theoretical optimum.

[CITATION NEEDED — concept: Clay waterfall reinforcement learning provider ordering]

Exercise hooks:
- **Easy**: Model a 3-provider waterfall as an MDP—define states, actions, and rewards.
- **Medium**: Given historical success rates for 4 enrichment providers, compute the first 5 Q-learning updates for the waterfall ordering problem.
- **Hard**: Implement a simulation that trains a Q-table on synthetic provider outcomes, then compare the learned waterfall order against a hand-tuned heuristic order.

## Beat 6: Ship It

Address production deployment: handling non-stationary reward distributions (provider reliability changes over time), choosing between fixed and decaying exploration rates for long-running waterfall systems, and logging requirements for reconstructing the Q-table state at any point. Discuss the limit of tabular TD methods—when the state space exceeds practical memory (e.g., per-account enrichment with thousands of feature combinations), the practitioner must move to function approximation, which is the next lesson.

Exercise hooks:
- **Easy**: Write a logging function that records every `(state, action, reward, next_state)` tuple for audit purposes.
- **Medium**: Implement a sliding-window average that detects when a provider's success rate has shifted significantly and triggers Q-table reinitialization.
- **Hard**: Replace the dictionary Q-table with a linear function approximator using hand-crafted features, run both on a 20×20 grid, and compare sample efficiency.