# Proximal Policy Optimization (PPO)

## GTM Redirect Rules

PPO is the optimization backbone of RLHF (Reinforcement Learning from Human Feedback), the alignment process that makes language models follow instructions and refuse harmful outputs. Every AI assistant deployed in GTM workflows — drafting outreach, qualifying leads, synthesizing account research — relies on a model that was RLHF'd with PPO. The redirect: **this is how your AI copilot was trained to be helpful rather than hazardous**. Foundational for Zone 1 (AI Fundamentals) and Zone 2 (Model Training & Fine-Tuning).

[CITATION NEEDED — concept: exact GTM cluster mapping for RLHF/model alignment in gtm-topic-map.md]

---

## Beat 1: Hook

Policy gradient methods have a self-sabotage problem: a single too-large update can collapse a working policy into garbage. Trust Region Policy Optimization (TRPO) solved this with a KL-divergence constraint, but it requires second-order optimization (conjugate gradient) — expensive and complex to implement. PPO replaces the hard constraint with a clipped surrogate objective that is first-order, simple to code, and empirically competitive. OpenAI published PPO in 2017; it became the default RL optimizer for RLHF in InstructGPT, ChatGPT, and most aligned LLMs since.

---

## Beat 2: Concept

Three mechanisms, in order:

1. **The probability ratio.** Define $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$. This is how much the new policy diverges from the old one for a given action. A ratio of 1.0 means no change. The vanilla policy gradient objective can be written as $\hat{A}_t \cdot r_t(\theta)$, where $\hat{A}_t$ is the advantage estimate.

2. **The clip.** PPO clips the ratio to $[1 - \epsilon, 1 + \epsilon]$ (typically $\epsilon = 0.2$). The clipped surrogate objective is:
$$L^{\text{CLIP}}(\theta) = \hat{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
The `min` means: take the lower of the clipped and unclipped objective. This creates a "flat spot" — once the ratio moves beyond the clip range, the gradient goes to zero. The policy cannot profit from overshooting.

3. **Why this works (and why TRPO is overkill).** TRPO enforces a hard KL constraint via conjugate gradient descent. PPO removes the constraint entirely and relies on the clip to penalize large ratio deviations. It is not equivalent to a true trust region — the guarantee is looser. In practice, it works well enough and is much faster. The skeptical read: PPO's success is partly empirical, not theoretically airtight. The clip is a heuristic that happens to generalize.

Additional details:
- PPO typically combines the clipped surrogate with an entropy bonus (to encourage exploration) and a value-function loss (for the critic).
- The full objective is $L(\theta) = L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\theta) + c_2 S[\pi_\theta](s_t)$.
- Multiple epochs of minibatch updates on the same trajectory batch are standard (unlike vanilla policy gradient, which does one update per rollout).

---

## Beat 3: Demonstration

Implement the PPO clipped objective in pure NumPy on a toy policy. Show:
- The unclipped objective $\hat{A} \cdot r$ vs. the clipped objective for a range of ratios.
- That the gradient goes to zero outside the clip range (print gradient values).
- A single PPO update step on a small batch, printing the before/after policy parameters.

Code output: a table showing ratio, unclipped loss, clipped loss, and gradient magnitude for each sample. Observable confirmation that the clip activates when $r > 1 + \epsilon$ or $r < 1 - \epsilon$.

---

## Beat 4: Use It

**GTM connection: RLHF and the AI assistant you actually use.**

When you prompt an LLM to "write a follow-up email to a cold lead that went silent," the model generates helpful, on-brand copy instead of garbage or harmful text because it was aligned with RLHF. PPO is the optimizer inside that RLHF loop. The process:

1. Supervised fine-tuning (SFT) on good demonstrations.
2. Train a reward model from human preference rankings.
3. Optimize the SFT model against the reward model using PPO.

The reward model scores each generated output. PPO updates the policy (the LLM) to maximize reward while the clip prevents the model from overfitting to the reward model's quirks — a failure mode called "reward hacking."

This is not theoretical: the InstructGPT paper (Ouyang et al., 2022) documents exactly this pipeline, and it is the direct predecessor to ChatGPT.

[CITATION NEEDED — concept: GTM cluster for AI copilot/assistant usage in sales workflows]

---

## Beat 5: Ship It

Implement a minimal PPO training loop on CartPole-v1 using `gymnasium` and PyTorch. The student will:

- Build an actor-critic network (shared backbone, two heads).
- Collect trajectories, compute advantages with GAE ($\lambda = 0.95$).
- Run the PPO clipped update for multiple epochs per batch.
- Plot episodic return over training steps — observable proof that the agent learns.

Hyperparameters to expose and justify: $\epsilon = 0.2$, $\gamma = 0.99$, GAE $\lambda = 0.95$, learning rate $3 \times 10^{-4}$, number of update epochs $= 10$.

**Exercise hooks:**
- (Easy) Run the provided PPO loop. Report the episode return at step 0, step 500, and step 2000.
- (Medium) Change $\epsilon$ from 0.2 to 0.02 and to 0.5. For each, report whether training converges and how many steps it takes. Explain the result in terms of the clip mechanism.
- (Hard) Replace the clipped surrogate with an unclipped policy gradient objective (remove the clip and the `min`). Compare sample efficiency and stability. Document any training collapses.

---

## Beat 6: Evaluate

**Quiz hooks (not full questions — hooks only):**

1. Given a ratio $r = 1.4$, $\epsilon = 0.2$, and advantage $\hat{A} = 2.0$, compute the clipped objective value. Show your work.
2. Why does PPO take the `min` of the clipped and unclipped objectives rather than just using the clipped one? What failure mode does the `min` prevent?
3. A practitioner reports that their PPO run is unstable — the policy oscillates and never converges. They are using 50 epochs of minibatch updates per trajectory batch. What is the likely cause and the fix?
4. PPO does not enforce a hard KL constraint. What is the skeptical argument against PPO's theoretical guarantees, and why does it work well in practice despite this?
5. In the RLHF pipeline, what is "reward hacking" and how does PPO's clip help mitigate it?

**Exercise hooks:**
- (Easy) Trace the PPO objective computation for 5 samples by hand. Verify against the code output from Beat 3.
- (Medium) Run the CartPole PPO loop. Produce the training curve. Annotate where the clip is activating most frequently (hint: log the fraction of clipped samples per update).
- (Hard) Implement PPO with an adaptive KL penalty (PPO-penalty variant from the original paper) instead of the clip. Compare against the clipped variant on CartPole. Report sample efficiency and wall-clock time.