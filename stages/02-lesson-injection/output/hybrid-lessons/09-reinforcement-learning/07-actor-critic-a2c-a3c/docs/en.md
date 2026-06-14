# Actor-Critic — A2C and A3C

## Learning Objectives

- **Implement** an A2C agent with shared actor-critic network architecture on CartPole-v1, achieving 195+ average reward within 300 episodes.
- **Derive** the advantage function in both Monte Carlo and TD(0) forms, and explain why the TD residual reduces gradient variance without changing the expectation.
- **Compare** synchronous (A2C) versus asynchronous (A3C) training paradigms, identifying the tradeoffs in wall-clock time, gradient staleness, and reproducibility.
- **Trace** the gradient flow through a two-headed neural network, identifying which loss term updates the actor head versus the critic head.
- **Configure** production stabilization techniques — entropy regularization, gradient clipping, reward normalization — and measure their individual impact on training stability.

## The Problem

Vanilla REINFORCE works, but its gradient estimator is unbearably noisy. The policy gradient theorem says you should scale `∇ log π(a|s)` by the return `G_t`. But `G_t` is a Monte Carlo sample — it is the sum of rewards across an entire episode, and those sums swing wildly. On CartPole, a single episode might return 12 or 200 depending on early dynamics that had nothing to do with the quality of the action being reinforced. You are multiplying a noisy scalar by a gradient vector and trusting the average over thousands of episodes to point in the right direction.

The fix is to subtract a baseline. Any function of the state `b(s_t)` works — subtracting it from the return does not change the expectation of the gradient (because `b(s_t)` has no dependence on the action and thus zero expected gradient contribution), but it can dramatically reduce the variance. The math says: pick `b(s_t)` to be the expected return from that state, and the variance drops to its practical floor. That quantity is the value function `V(s_t)`. What remains after subtraction is the *advantage*: `A(s_t, a_t) = G_t - V(s_t)`. Positive advantage means the action was better than average; negative means worse. The gradient signal now points at *relative* quality, not absolute return.

This is the insight that splits every modern deep RL method into two halves. You need a policy network (the actor) that proposes actions, and a value network (the critic) that scores states. The actor trains on the advantage signal. The critic trains to minimize the gap between its prediction and the observed return. The two networks share representation — early layers see the same state — but diverge at the output heads. This architecture is actor-critic, and it is the backbone of A2C, A3C, PPO, SAC, IMPALA, and every serious policy-gradient method since 2015.

## The Concept

The actor-critic architecture is two functions of the same state, trained jointly. The actor `π_θ(a|s)` outputs a probability distribution over actions. The critic `V_φ(s)` outputs a scalar estimating expected discounted return from that state. Both networks can share trunk layers (common in practice) or be fully separate (cleaner gradient isolation). The advantage `A(s,a)` is the coupling signal: it tells the actor which direction to push, and the difference between the advantage and zero tells the critic how wrong its value estimate was.

Two forms of advantage are standard. The Monte Carlo advantage `A_t = G_t - V_φ(s_t)` uses the full episodic return — unbiased but high variance. The TD advantage `A_t = r_{t+1} + γV_φ(s_{t+1}) - V_φ(s_t)` uses bootstrapping — biased (because `V_φ` is wrong early in training) but much lower variance. Practitioners often use n-step returns, which interpolate: `A_t = Σ_{k=0}^{n-1} γ^k r_{t+k+1} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`. The n-step form captures more reward signal than TD(0) while bootstrapping sooner than full MC, hitting a practical sweet spot at n=5 for most continuous control tasks.

```mermaid
flowchart TD
    A[Environment State s_t] --> B[Shared Trunk Layers]
    B --> C[Actor Head π_θ]
    B --> D[Critic Head V_φ]
    C --> E[Sample Action a_t]
    E --> F[Environment Step]
    F --> G[r_{t+1}, s_{t+1}]
    D --> H[V_φ s_t]
    D --> I[V_φ s_{t+1}]
    G --> I
    G --> J[Compute Advantage<br/>A_t = r + γV_s' - V_s]
    H --> J
    I --> J
    J --> K[Actor Loss:<br/>-log π * A_t]
    J --> L[Critic Loss:<br/>A_t² or TD error²]
    K --> M[Total Loss = Actor + c1*Critic - c2*Entropy]
    L --> M
    M --> N[Backprop → Update θ, φ]
    N --> B
```

A2C and A3C differ only in how they parallelize the environment interaction. **A2C (Advantage Actor-Critic)** runs N environments synchronously, collects a batch of transitions from all of them simultaneously, computes one gradient update, and applies it. The synchronization guarantees that every gradient step sees a consistent snapshot of the policy. **A3C (Asynchronous Advantage Actor-Critic)** runs N workers in separate threads, each interacting with its own environment copy and computing gradient updates independently. Workers push gradients to a global network asynchronously — no locks, no coordination. The global network averages whatever arrives whenever it arrives.

The asynchronous design was a breakthrough when DeepMind published it in 2016 because it eliminated the need for experience replay (workers provide decorrelated gradients naturally) and cut wall-clock training time substantially. But A3C has a problem: *stale gradients*. Worker A computes a gradient based on a policy snapshot from 50ms ago, by which point Worker B has already updated the global network. The gradient A pushes is computed against parameters that no longer exist. In practice this is usually fine — the staleness acts as a form of noise regularization — but it makes debugging hard and reproducibility harder. Most practitioners who need parallelism today default to A2C with vectorized environments (via `gym.vector.SyncVectorEnv` or `EnvPool`) because synchronous execution is easier to reason about and nearly as fast on modern hardware. PPO, which we cover next, is essentially A2C with clipped objectives — the synchronous design won.

## Build It

We build a minimal A2C agent on CartPole-v1. The network has a shared trunk with two output heads — one producing action logits, the other producing the state-value estimate. We compute the TD advantage, combine actor loss and critic loss into a single objective, and train with one optimizer.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden, n_actions)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        logits = self.actor_head(h)
        value = self.critic_head(h).squeeze(-1)
        return logits, value

def compute_advantage(rewards, values, gamma, dones, next_value):
    returns = []
    R = next_value
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R * (1 - dones[t])
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = returns - values
    return returns, advantages

env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

model = ActorCritic(obs_dim, n_actions)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
gamma = 0.99
entropy_coef = 0.01
value_coef = 0.5
max_episodes = 300

reward_history = []

for episode in range(max_episodes):
    state, _ = env.reset()
    states, actions, rewards, dones = [], [], [], []
    ep_reward = 0
    done = False

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, value = model(state_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(float(done))
        ep_reward += reward
        state = next_state

    reward_history.append(ep_reward)

    states_t = torch.tensor(np.array(states), dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.long)

    with torch.no_grad():
        next_state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        _, next_value = model(next_state_t)
        next_value = next_value.item()

    logits, values = model(states_t)
    values = values.squeeze(-1)

    returns, advantages = compute_advantage(
        rewards, values, gamma, dones, next_value
    )

    dist = torch.distributions.Categorical(logits=logits)
    log_probs = dist.log_prob(actions_t)
    entropy = dist.entropy().mean()

    actor_loss = -(log_probs * advantages.detach()).mean()
    critic_loss = F.mse_loss(values, returns)
    total_loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    if (episode + 1) % 10 == 0:
        avg = np.mean(reward_history[-10:])
        print(f"Episode {episode+1:4d} | Avg Reward (last 10): {avg:6.1f} | "
              f"Actor Loss: {actor_loss.item():.4f} | Critic Loss: {critic_loss.item():.4f} | "
              f"Entropy: {entropy.item():.4f}")

final_avg = np.mean(reward_history[-10:])
print(f"\nFinal 10-episode average: {final_avg:.1f}")
print(f"Solved (>=195): {'YES' if final_avg >= 195 else 'NO'}")
env.close()
```

Running this produces output that converges upward over ~200–300 episodes. The actor loss and critic loss should trend in opposite directions early — the critic gets more accurate as the actor gets better at maximizing reward — and entropy should slowly decrease as the policy commits to a strategy. The gradient clipping at 0.5 prevents occasional large advantages from blowing up the shared trunk weights.

The advantage computation here uses full Monte Carlo returns internally (the `compute_advantage` function bootstraps from `next_value` but accumulates the full episode). This is the n-step advantage with n = episode length. For true TD(0) advantage, you would compute `A_t = r_{t+1} + γV(s_{t+1}) - V(s_t)` per-step — but since CartPole episodes are short, the MC-style return is fine and gives cleaner gradient signal at this scale.

## Use It

The actor-critic pattern — one function proposes, another scores — appears in GTM engineering wherever a system generates content or actions and simultaneously evaluates them. Consider the multichannel outreach pipeline: an enrichment layer (actor) proposes a sequence of touchpoints for a given account, and a scoring layer (critic) estimates the probability that the sequence converts based on historical response data. The advantage signal — the delta between predicted and observed conversion — updates both layers. This is not a metaphor; it is the same gradient structure, just operating on discrete campaign data instead of continuous RL environments. The actor-critic split is the decision-system backbone for autonomous GTM orchestration, where a task router (actor) decides what fires next across channels and an evaluation function (critic) scores the routing decision against downstream engagement signals. [CITATION NEEDED — concept: GTM multi-agent decision systems mapping to actor-critic pattern]

Zone 9 of the curriculum covers agents, tool use, and function calling — the infrastructure layer for this kind of orchestration. The task router pattern (deciding what fires next, each tool call as a node) maps directly onto the actor's role: given the current state of a prospect's journey, output a distribution over next actions. The critic's role maps to the feedback loop: did the action move the prospect closer to a meeting? The advantage — positive if the action outperformed the expected value of being in that state — tells the system which routing decisions to reinforce. Saruggia's handbook positions workflow automation tools (n8n, Make) and cold calling infrastructure as the execution substrate for these decision loops, but does not explicitly map them to RL architectures. The connection is structural, not documented. [CITATION NEEDED — concept: n8n/Make workflow automation as actor-critic execution layer, per Saruggia GTM handbook]

The practical takeaway: if you are building a system that *both generates and evaluates* GTM actions — outreach sequences, lead scoring, channel selection, send-time optimization — the actor-critic decomposition gives you a principled way to structure the code. Separate the proposal function from the evaluation function. Train both on the same signal (the advantage). Do not let the proposal function see raw outcomes without the baseline — that is the REINFORCE mistake, and it produces the same variance problem in GTM that it produces in RL: noisy, slow, hard-to-debug updates.

## Ship It

Production actor-critic training requires four stabilization techniques that are absent from the minimal implementation above. **Entropy regularization** (`-entropy_coef * H(π)`) prevents the policy from collapsing to a deterministic action too early — without it, the actor can commit to a single action and stop exploring before finding the reward-maximizing strategy. A coefficient of 0.01 is standard; too high and the policy stays random forever, too low and it collapses prematurely. **Gradient clipping** at 0.5 (via `clip_grad_norm_`) prevents rare large advantages from destabilizing the shared trunk. **Reward normalization** — dividing the advantage batch by its running standard deviation — keeps the gradient magnitude consistent across environments with different reward scales. **Value function coefficient** (typically 0.5) weights the critic loss relative to the actor loss; since the critic converges faster, underweighting it prevents the critic from dominating the shared representation.

Here is the production checkpoint and inference pattern:

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "episode": max_episodes,
    "final_avg_reward": final_avg,
    "hyperparams": {
        "lr": 3e-3,
        "gamma": 0.99,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "grad_clip": 0.5,
    },
}
torch.save(checkpoint, "a2c_cartpole.pt")
print("Checkpoint saved to a2c_cartpole.pt")

loaded = torch.load("a2c_cartpole.pt", weights_only=False)
inference_model = ActorCritic(obs_dim, n_actions)
inference_model.load_state_dict(loaded["model_state_dict"])
inference_model.eval()

eval_env = gym.make("CartPole-v1")
eval_rewards = []

for ep in range(10):
    state, _ = eval_env.reset()
    ep_reward = 0
    done = False
    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = inference_model(state_t)
            action = torch.argmax(logits, dim=-1).item()
        state, reward, terminated, truncated, _ = eval_env.step(action)
        ep_reward += reward
        done = terminated or truncated
    eval_rewards.append(ep_reward)

print(f"\nGreedy evaluation (10 episodes):")
print(f"  Mean: {np.mean(eval_rewards):.1f}")
print(f"  Min:  {np.min(eval_rewards):.1f}")
print(f"  Max:  {np.max(eval_rewards):.1f}")
print(f"  All >=195: {'YES' if min(eval_rewards) >= 195 else 'NO'}")
eval_env.close()
```

The greedy evaluation (argmax instead of sample) confirms that the learned policy is actually good, not just lucky on exploration noise. If training reward is high but greedy reward collapses, the policy is relying on stochastic exploration — a sign that entropy regularization is too high or training stopped too early.

The decision matrix for choosing A2C vs. PPO vs. SAC in production:

- **A2C**: Use when you need simplicity and fast iteration. Good for discrete action spaces, short episodes, and environments where you can afford on-policy training (no experience replay). Weakness: sample-inefficient compared to off-policy methods.
- **PPO**: Use when A2C training is unstable. PPO clips the policy ratio, preventing large updates that destroy the policy. The default choice for most discrete-action problems in 2025.
- **SAC**: Use for continuous action spaces and when sample efficiency matters. SAC is off-policy (uses a replay buffer) and maximizes entropy, producing more robust exploration. The default for continuous control.

For GTM-adjacent systems, the "environment latency" question matters: if each action step requires waiting for a human response (email open, meeting booking), on-policy methods like A2C and PPO are impractical because you cannot collect enough fresh on-policy data. Off-policy methods (SAC, DQN-family) that learn from a replay buffer of historical interactions are better suited to GTM decision systems where the "environment" is a prospect whose response arrives days later. [CITATION NEEDED — concept: off-policy vs on-policy selection criteria for GTM decision systems with high environment latency]

## Exercises

**Easy: Modify the advantage baseline.** Replace the learned critic `V_φ(s)` in the advantage calculation with a discounted return baseline: compute `A_t = G_t - mean(G)` across the episode. Run training for 300 episodes. Print the advantage variance (standard deviation of advantages per episode) for both the critic-baseline version and the mean-baseline version side-by-side at episodes 50, 150, and 250. The critic-baseline version should show lower variance as training progresses.

**Medium: Extend to 4 parallel environments.** Use `gym.vector.SyncVectorEnv` to create 4 CartPole environments. Collect transitions from all 4 environments simultaneously, concatenate the advantages across environments before the backward pass, and apply a single gradient update per step. Print per-environment reward every 10 steps to confirm synchronization. Target: 4x faster wall-clock convergence to 195+ compared to the single-environment version, with the same total number of gradient steps.

**Hard: Gradient accumulation with fixed batch size.** Modify the 4-environment A2C to accumulate gradients across K steps (K=5) before calling `optimizer.step()`. This gives an effective batch of 4×5=20 transitions per update. Compare sample efficiency — reward as a function of total environment steps — between (a) single-env with step-every-update, (b) 4-env with step-every-update, and (c) 4-env with 5-step accumulation. Print a comparison table at 10K, 25K, and 50K total steps. Hypothesis: gradient accumulation should improve sample efficiency by reducing gradient noise, at the cost of slightly slower wall-clock time per step.

## Key Terms

- **Actor**: The policy network `π_θ(a|s)` that outputs a probability distribution over actions. Trained via policy gradient, scaled by the advantage.
- **Critic**: The value network `V_φ(s)` that estimates expected discounted return from a state. Trained via TD error or MSE against observed returns.
- **Advantage**: `A(s,a) = R + γV(s') - V(s)` (TD form) or `A(s,a) = G_t - V(s_t)` (MC form). Measures how much better an action was compared to the average for that state. The coupling signal between actor and critic.
- **A2C (Advantage Actor-Critic)**: Synchronous variant. Multiple environments collected in parallel, single gradient update per batch. Deterministic, reproducible, easy to debug.
- **A3C (Asynchronous Advantage Actor-Critic)**: Asynchronous variant. Multiple worker threads with independent environment copies, gradients pushed to a global network without coordination. Faster wall-clock, but introduces stale gradients and reproducibility issues.
- **TD Residual / TD Error**: `δ_t = r_{t+1} + γV(s_{t+1}) - V(s_t)`. The one-step advantage. Biased (depends on critic accuracy) but lowest variance.
- **Entropy Regularization**: A penalty term `-entropy_coef × H(π)` added to the loss to discourage premature policy collapse. Keeps the actor exploring.
- **Gradient Clipping**: Capping the norm of the gradient vector (typically at 0.5) before the optimizer step. Prevents rare large advantages from destabilizing training.
- **n-step Return**: `G_t^{(n)} = Σ_{k=0}^{n-1} γ^k r_{t+k+1} + γ^n V(s_{t+n})`. Interpolates between TD(0) and full MC return. Standard choice is n=5 for most continuous control tasks.
- **On-policy vs. Off-policy**: A2C and A3C are on-policy — the policy being evaluated must be the same as the policy being improved. SAC and DQN are off-policy — they learn from data collected by a different (older) policy via a replay buffer.

## Sources

- Saruggia, Michael. *The 80/20 GTM Engineer Handbook*. Growth Lead LLC, 2025. Zone 09 (Agents, tool use, function calling) and multichannel execution sections provide the workflow automation and task router context referenced in "Use It." The handbook does not explicitly map GTM decision systems to actor-critic architecture; that mapping is structural inference. [CITATION NEEDED — concept: GTM multi-agent decision systems mapping to actor-critic pattern]
- [CITATION NEEDED — concept: n8n/Make workflow automation as actor-critic execution layer, per Saruggia GTM handbook]
- [CITATION NEEDED — concept: off-policy vs on-policy selection criteria for GTM decision systems with high environment latency]
- Mnih, V. et al. "Asynchronous Methods for Deep Reinforcement Learning." *ICML 2016*. The A3C paper introducing asynchronous parallel actors with gradient pushing to a global network.
- Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., Chapter 13 (Policy Gradient Methods). MIT Press, 2018. Baseline subtraction and variance reduction derivation.