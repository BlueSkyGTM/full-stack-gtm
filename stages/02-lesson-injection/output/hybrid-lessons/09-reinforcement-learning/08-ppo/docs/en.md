# Proximal Policy Optimization (PPO)

## Learning Objectives

- Implement the clipped surrogate objective from first principles and trace how the clip parameter $\epsilon$ controls policy divergence per update.
- Compare PPO's first-order clip against TRPO's second-order KL constraint, identifying what each guarantees and what each sacrifices.
- Run multiple epochs of minibatch updates on a single rollout without the policy collapsing, and measure the probability ratio across training to verify clipping is active.
- Diagnose when PPO is the right optimizer for RLHF and when alternative alignment methods (DPO, GRPO) are more appropriate.
- Evaluate how the clipped surrogate objective maps to constrained behavior shaping in GTM AI systems — specifically, why RLHF'd models stay helpful instead of drifting toward harmful or off-task outputs.

## The Problem

Policy gradient methods have a self-sabotage problem. The gradient $\hat{E}_{\pi_\theta}[A_t \cdot \nabla \log \pi_\theta(a_t|s_t)]$ tells your model to increase the probability of actions with positive advantage and decrease actions with negative advantage. That direction is correct — but the magnitude is not guaranteed. A single too-large update can push your policy so far from the distribution that collected the data that the gradient becomes meaningless, and the policy collapses into garbage. You went from a functioning agent to a random number generator in one step.

A2C (actor-critic) is on-policy: the gradient requires data sampled from the *current* $\pi_\theta$. Take one update, and $\pi_\theta$ has already changed. The data you used is now off-policy — re-using it introduces bias. On Atari, one rollout across 8 environments × 128 steps = 1024 transitions and several seconds of wall-clock time. Throwing that away after a single gradient step is wasteful, but re-using it naively will destroy your policy.

Trust Region Policy Optimization (TRPO, Schulman et al. 2015) was the first principled fix. TRPO constrains each update so the KL divergence between the old policy and new policy stays below a threshold $\delta$. This guarantees monotonic improvement — you cannot accidentally destroy your policy because the constraint prevents large moves. The cost: TRPO requires computing the natural gradient via conjugate gradient descent, a second-order method that involves Fisher-vector products and a line search. It works, but it is expensive and complex to implement. By 2017, most practitioners found TRPO too cumbersome for production-scale training.

PPO (Schulman et al. 2017) asks a different question: what if we replace the hard constraint with a soft penalty that is cheap to compute? The answer is the clipped surrogate objective — one extra line of code that lets you run ten or more epochs of updates on the same rollout without the policy exploding. No conjugate gradients. No second-order math. Nine years later, PPO is still the default policy-gradient algorithm for everything from MuJoCo robotics to RLHF alignment of production language models.

## The Concept

Three mechanisms make PPO work. Understand them in order, because each builds on the previous.

**The probability ratio.** Define $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$. This is the likelihood ratio of the new policy versus the policy that collected the data. A ratio of 1.0 means the new policy assigns the same probability to action $a_t$ as the old one — no change. A ratio of 2.0 means the new policy is twice as likely to take $a_t$. The vanilla policy gradient objective can be rewritten as $\hat{A}_t \cdot r_t(\theta)$, where $\hat{A}_t$ is the advantage estimate. This is an importance sampling ratio, and it lets you evaluate the new policy using data collected by the old one.

**The clip.** PPO clips the ratio to $[1 - \epsilon, 1 + \epsilon]$, typically $\epsilon = 0.2$. The clipped surrogate objective is:

$$L^{\text{CLIP}}(\theta) = \hat{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

The `min` of the two terms is what creates the "flat spot." When the advantage is positive and the ratio grows past $1 + \epsilon$, the clipped term stops rewarding further increases — the gradient goes to zero. When the advantage is negative and the ratio shrinks below $1 - \epsilon$, the clipped term stops penalizing further decreases — again, gradient goes to zero. The policy cannot profit from overshooting. It can move in the direction the advantage suggests, but only up to a bounded distance per update.

**Why this works (and why TRPO is overkill).** TRPO enforces a hard KL constraint via conjugate gradient descent and a line search. The guarantee is theoretically clean: monotonic improvement, bounded policy divergence. PPO removes the constraint entirely and relies on the clip to penalize large ratio deviations. This is *not* equivalent to a true trust region — the guarantee is looser, and a sufficiently pathological advantage landscape could still cause problems. In practice, PPO works well enough and trains significantly faster because it uses first-order optimization (standard Adam/SGD). The skeptical read: PPO's dominance is partly empirical rather than theoretically airtight. The clip is a heuristic that happens to generalize across a wide range of environments.

PPO typically combines the clipped surrogate with two additional terms. An entropy bonus $c_2 S[\pi_\theta](s_t)$ encourages exploration by penalizing overly confident (low-entropy) policies. A value-function loss $c_1 L^{\text{VF}}(\theta)$ trains the critic to produce better advantage estimates. The full objective:

$$L(\theta) = L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\theta) + c_2 S[\pi_\theta](s_t)$$

The training loop runs multiple epochs of minibatch updates over the same rollout buffer, which is the key efficiency win over A2C. The clip prevents destructive updates across those epochs.

```mermaid
flowchart TD
    A[Collect rollout with π_old] --> B[Compute advantages Â_t using critic V_φ]
    B --> C[For each epoch 1..K:]
    C --> D[For each minibatch:]
    D --> E[Compute ratio r_t = π_θ a_t|s_t / π_old a_t|s_t]
    E --> F{r_t in 1±ε?}
    F -- Yes --> G[Use unclipped objective: r_t · Â_t]
    F -- No --> H[Use clipped objective: clip r_t · Â_t]
    G --> I[L = min unclipped, clipped]
    H --> I
    I --> J[L_total = L^CLIP - c1·L^VF + c2·entropy]
    J --> K[Adam gradient step on θ]
    K --> L{More minibatches?}
    L -- Yes --> D
    L -- No --> M{More epochs?}
    M -- Yes --> C
    M -- No --> N[Update π_old ← π_θ. Collect new rollout.]
```

## Build It

The core of PPO is the clipped surrogate loss. Everything else — the rollout collection, the advantage estimation, the value function — is shared with actor-critic methods you have already built. What makes PPO PPO is the loss function and the multi-epoch training loop.

Here is a minimal PPO implementation that trains a policy on a simple environment. It uses a discrete action space (CartPole) so you can see the probability ratio and clipping behavior directly. The code is self-contained: install `gymnasium` and `torch`, and it runs.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * 1.0 - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages

env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
model = ActorCritic(obs_dim, act_dim)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

clip_epsilon = 0.2
epochs = 4
gamma = 0.99

reward_history = []

for iteration in range(50):
    obs_buf, act_buf, rew_buf = [], [], []
    val_buf = []
    obs, _ = env.reset()
    ep_reward = 0

    for step in range(2048):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, value = model(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action.item())

        obs_buf.append(obs)
        act_buf.append(action.item())
        rew_buf.append(reward)
        val_buf.append(value.item())
        ep_reward += reward
        obs = next_obs

        if terminated or truncated:
            obs, _ = env.reset()
            reward_history.append(ep_reward)
            ep_reward = 0

    advantages = compute_gae(rew_buf, val_buf, gamma=0.99, lam=0.95)
    returns = advantages + torch.tensor(val_buf, dtype=torch.float32)

    obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32)
    act_t = torch.tensor(act_buf, dtype=torch.long)

    with torch.no_grad():
        old_logits, _ = model(obs_t)
        old_dist = Categorical(logits=old_logits)
        old_log_probs = old_dist.log_prob(act_t)

    for epoch in range(epochs):
        logits, values = model(obs_t)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(act_t)
        entropy = dist.entropy().mean()

        ratio = torch.exp(log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = ((values.squeeze() - returns) ** 2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    if (iteration + 1) % 10 == 0:
        avg_r = np.mean(reward_history[-10:]) if reward_history else 0
        clipped_count = (ratio.abs() > (1 + clip_epsilon)).sum().item()
        print(f"Iter {iteration+1:3d} | Avg Reward (last 10 eps): {avg_r:6.1f} | "
              f"Ratio mean: {ratio.mean():.4f} | Clipped samples: {clipped_count}/{len(ratio)}")

env.close()
print(f"\nFinal avg reward (last 20 episodes): {np.mean(reward_history[-20:]):.1f}")
```

This trains a PPO agent on CartPole for 50 iterations. Each iteration collects 2048 transitions, then runs 4 epochs of minibatch-free gradient updates on the full buffer. The output prints every 10 iterations showing the average reward, the mean probability ratio (should hover near 1.0), and how many samples were clipped (should be a small fraction). If you see the ratio drifting far from 1.0 or the clipped count rising above ~15% of samples, your learning rate is too high or your clip range is too tight.

The key line is `torch.min(surr1, surr2).mean()`. That is the entire clipped surrogate objective. Everything else is standard actor-critic machinery — the rollout buffer, GAE advantage estimation, the value function loss. The clip is what makes multi-epoch training safe.

## Use It

PPO is the optimization backbone of RLHF (Reinforcement Learning from Human Feedback), the alignment process that trains language models to follow instructions and refuse harmful outputs. Every AI assistant you deploy in a GTM workflow — drafting personalized outreach, qualifying inbound leads, synthesizing account research — runs on a model that was RLHF'd with PPO. The connection is not metaphorical: the same clipped surrogate objective that keeps a CartPole policy from exploding also keeps a language model's text generation from drifting into incoherence or toxicity during alignment.

Here is why this matters concretely. In RLHF, the "policy" is the language model, the "action" is the token it generates, and the "reward" comes from a reward model trained on human preference data. The reward model scores how helpful, harmless, and honest a response is. PPO optimizes the language model to maximize that reward — but the clip prevents the model from overfitting to the reward signal in destructive ways. Without the clip, the model might discover that certain phrase patterns get high reward scores and collapse into repeating them. The clip bounds how far the model can move away from its pre-RLHF behavior per update, which is exactly the mechanism that keeps your outreach copilot writing varied, natural prose instead of degenerating into a reward-hacking loop.

The training loop in RLHF mirrors the PPO loop you just built. Collect a batch of prompts and responses (rollout). Score them with the reward model (advantage). Run several epochs of clipped updates on the language model weights. The probability ratio $r_t(\theta)$ measures how much the model's output distribution has shifted from its pre-update state. The clip at $1 \pm 0.2$ ensures that no single update pushes the model more than 20% in likelihood space for any given token sequence. This is what makes RLHF stable enough to run on 100B+ parameter models without catastrophic forgetting.

For a GTM practitioner, the practical implication is this: when you fine-tune a model for your specific use case — say, adapting a base model to write in your brand voice for outbound sequences — the stability properties of PPO (or its successors like DPO and GRPO) are what prevent the model from losing its general capabilities while gaining your domain-specific ones. [CITATION NEEDED — concept: exact GTM cluster mapping for RLHF/model alignment in gtm-topic-map.md] The Zone 9 row on agents and tool use applies here: "You built the algorithm. Now wire it into a loop." PPO is the algorithm; the RLHF pipeline is the loop; your deployed GTM copilot is the output.

## Ship It

When you deploy PPO-trained models into GTM infrastructure, the clip parameter $\epsilon$ has a direct business consequence. A tighter clip (smaller $\epsilon$) produces a model that stays closer to its pre-alignment behavior — safer, more conservative, but less optimized for your specific reward signal. A looser clip (larger $\epsilon$) allows faster adaptation but risks reward hacking, where the model exploits quirks of the reward model rather than genuinely improving. This tradeoff maps directly to personalization and follow-up cadence in outbound workflows: you want the model to adapt to what resonates with your prospects (positive reward from reply rates), but you do not want it to collapse into repeating a single subject line that happened to get one reply.

The skeptical practitioner should also know PPO's limitations for modern alignment. Direct Preference Optimization (DPO, Rafailov et al. 2023) eliminates the reward model entirely by deriving a closed-form solution to the preference optimization problem. Group Relative Policy Optimization (GRPO, used in DeepSeek-R1) eliminates the value function by using group-relative advantages. Both are simpler pipelines than PPO-based RLHF. PPO remains the most empirically validated method and is still used in production at OpenAI and Anthropic, but if you are building an alignment pipeline in 2026, you should evaluate DPO and GRPO as alternatives rather than defaulting to PPO.

For shipping a PPO-trained model into a GTM tool stack — whether that is an n8n workflow that routes account research tasks to an LLM, or a Clay waterfall that enriches prospect data with AI-generated summaries — the model's alignment training is the invisible foundation. The task router decides what fires next; each tool call is a node in the workflow graph; but the model that executes each node was shaped by PPO (or a successor) to be helpful rather than hazardous. When the model refuses to generate a competitor smear campaign or declines to hallucinate revenue numbers for a prospect, that refusal behavior was reinforced during RLHF — and the clip is what kept that reinforcement from overcorrecting into blanket refusal of legitimate requests.

## Exercises

1. **Vary the clip range.** Change `clip_epsilon` from 0.2 to 0.05, 0.1, 0.3, and 0.5. For each value, run training and record the final average reward and the percentage of samples that get clipped. Write a one-paragraph summary of how clip range affects training stability and final performance. Which value gives the best results on CartPole? Does the optimal value change if you increase the learning rate to 1e-3?

2. **Remove the clip.** Comment out the `torch.min(surr1, surr2)` line and replace it with just `-surr1.mean()` (vanilla policy gradient with importance weighting, no clipping). Run 50 iterations. At what iteration does the policy collapse? What happens to the ratio values? This demonstrates what PPO's clip prevents.

3. **Log the ratio distribution.** Modify the training loop to store all ratio values across all epochs and iterations. After training, plot a histogram of ratios. Confirm that most ratios cluster near 1.0 and that the clip range creates a visible boundary. How many epochs can you run before the ratio distribution starts to drift away from 1.0?

4. **Compare epochs.** Set `epochs` to 1, 4, 10, and 20. For each, measure the total wall-clock training time and the final average reward. At what point do additional epochs stop helping (or start hurting)? This is the sample-efficiency vs. stability tradeoff that PPO was designed to navigate.

5. **Trace the RLHF connection.** Write pseudocode (not full implementation) for a PPO-based RLHF loop where the "environment" is a prompt, the "action" is token generation, and the "reward" comes from a reward model. Identify which components map directly to the CartPole PPO loop you built and which require new infrastructure (the reward model, the reference policy for KL penalty, the token-level advantage computation).

## Key Terms

**Clipped Surrogate Objective** — The PPO loss function $L^{\text{CLIP}}(\theta) = \hat{E}_t[\min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t)]$. Replaces TRPO's hard KL constraint with a soft, first-order penalty on ratio overshooting.

**Probability Ratio** — $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$. Measures how much the current policy diverges from the policy that collected the data for a given state-action pair. A ratio of 1.0 means no divergence.

**Trust Region** — The set of policies within a bounded KL divergence of the current policy. TRPO enforces this via conjugate gradient; PPO approximates it via the clip. The approximation is looser but much cheaper to compute.

**Generalized Advantage Estimation (GAE)** — The advantage estimator $\hat{A}_t$ used in most PPO implementations. Combines rewards and value estimates with an exponential weighting parameterized by $\lambda \in [0,1]$ to balance bias and variance.

**RLHF (Reinforcement Learning from Human Feedback)** — The alignment pipeline where a language model is fine-tuned using PPO to maximize a reward model trained on human preference data. The mechanism by which base models become instruction-following assistants.

**Reward Hacking** — When a policy exploits imperfections in the reward signal rather than solving the intended task. The PPO clip mitigates this by bounding how far the policy can move per update, preventing rapid exploitation of reward model quirks.

## Sources

- Schulman, J. et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347 (2017). — Original PPO paper, defines the clipped surrogate objective.
- Schulman, J. et al. "Trust Region Policy Optimization." ICML 2015. — TRPO, the predecessor PPO simplifies.
- Ouyang, L. et al. "Training language models to follow instructions with human feedback." NeurIPS 2022 (InstructGPT). — Documents PPO as the RL optimizer in RLHF for GPT-3 alignment.
- Rafailov, R. et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023. — DPO, the successor approach that eliminates the reward model from RLHF.
- [CITATION NEEDED — concept: exact GTM cluster mapping for RLHF/model alignment in gtm-topic-map.md]
- Zone 9 row, GTM topic map: "Agents, tool use, function calling | Workflow Automation (n8n/Make), Cold Calling Infrastructure (2.2) | Agent Stack"