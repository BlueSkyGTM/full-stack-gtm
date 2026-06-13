# Actor-Critic — A2C and A3C

## Hook
The policy gradient methods from the last lesson have a variance problem — the gradient signal is noisy because it uses full episode returns. Actor-Critic methods reduce that variance by introducing a learned baseline: a value function that estimates "how good is this state?" before the agent even picks an action. This lesson covers the two dominant synchronous/asynchronous variants: A2C and A3C.

## Concept
**Mechanism:** The actor network outputs a policy π(a|s). The critic network outputs a state-value estimate V(s). The advantage A(s,a) = R + γV(s') - V(s) becomes the signal that updates both networks — the critic minimizes TD error, the actor uses the advantage to reinforce good actions. A2C runs multiple environments synchronously on one process. A3C runs multiple workers asynchronously across threads, each with their own copy of the environment, periodically syncing gradients to a global network. The asynchronous version reduces wall-clock time but introduces stale gradient issues; A2C is the cleaner, reproducible variant that most practitioners default to today.

## Use It
**GTM Redirect:** Foundational for Zone 3 (Decision Systems). Multi-agent, sequential decision-making with feedback loops underpins autonomous GTM orchestration, but no direct Clay/waterfall mechanism maps here. The actor-critic split — one function proposes actions, another scores them — is the pattern behind any system that generates outreach and evaluates it simultaneously. [CITATION NEEDED — concept: GTM multi-agent decision systems mapping to actor-critic pattern]

## Code It
Build a minimal A2C agent on CartPole-v1 using PyTorch. Single network with two heads (actor and critic), computed advantage, single optimizer. Observable output: episode reward printed every 10 episodes, converging to ~195+ within 300 episodes.

**Exercise hooks:**
- **Easy:** Modify the advantage calculation to use a discounted return baseline instead of the critic. Print both advantage curves side-by-side to observe variance difference.
- **Medium:** Extend to 4 parallel environments using `SyncVectorEnv`. Concatenate advantages across environments before the backward pass. Print per-environment reward to confirm synchronization.
- **Hard:** Implement gradient accumulation with a fixed batch size across parallel envs. Compare sample efficiency (reward vs. total steps) between single-env and 4-env runs. Print the comparison table.

## Ship It
Production considerations: entropy regularization to prevent premature policy collapse, reward normalization across workers, gradient clipping at 0.5 to stabilize training, and the decision matrix for choosing A2C vs. PPO vs. SAC given environment latency and observation space. Save the model with `torch.save` and load it to run inference deterministically (no exploration noise). Print a validation run of 10 episodes with no gradient computation to confirm the saved policy behaves.

**Exercise hooks:**
- **Easy:** Add entropy bonus to the loss. Sweep coefficients [0.001, 0.01, 0.1] and print final mean reward for each — observe collapse at low entropy, instability at high.
- **Medium:** Implement a training loop checkpoint that saves the model every 50 episodes and resumes from the latest checkpoint on restart. Print "resumed from episode N" on reload.
- **Hard:** Profile the A2C loop. Time the forward pass, advantage computation, and backward pass separately. Print a table. Identify whether the bottleneck is env-stepping or gradient computation for CartPole vs. a heavier environment.

## Evaluate
Conceptual and implementation checks against the lesson objectives.

**Exercise hooks:**
- **Easy:** Given a printed advantage trace over 20 steps, identify where the critic's value estimate was too high vs. too low and explain what the actor gradient does in each case.
- **Medium:** Write a function that takes two trained models (one with critic, one without) and plots the variance of the gradient norm over 100 updates. Print the variance ratio.
- **Hard:** Remove the critic entirely and revert to REINFORCE with baseline. Run both agents on the same seed. Print a table: episodes to solve (reward ≥ 195), gradient variance, max reward achieved. Explain the tradeoffs in 3 sentences.

---

## Learning Objectives

1. **Implement** an A2C agent with separate actor and critic heads in PyTorch, training on CartPole-v1 to ≥195 mean reward.
2. **Compare** the variance of policy gradients with and without a learned critic baseline by measuring gradient norm statistics.
3. **Explain** why A3C's asynchronous gradient updates can produce stale gradients and why A2C is preferred for reproducibility.
4. **Configure** entropy regularization and gradient clipping to prevent policy collapse during training.
5. **Evaluate** the sample efficiency tradeoff between single-environment and multi-environment A2C by logging reward-per-timestep.