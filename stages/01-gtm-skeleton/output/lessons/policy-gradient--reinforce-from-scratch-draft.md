# Policy Gradient — REINFORCE from Scratch

## Hook It

Value-based methods (Q-learning, DQN) learn *what* actions are worth. Policy gradient methods learn *which* actions to pick, directly. REINFORCE is the simplest policy gradient algorithm — and the foundation for every modern RL-from-sequence-decisions method (PPO, A2A, GRPO). If you want to optimize multi-step decision processes where the reward comes at the end, this is where you start.

## Ground It

The policy gradient theorem states that the gradient of expected return with respect to policy parameters θ is the expectation of the return multiplied by the gradient of the log-probability of the actions taken. REINFORCE estimates this by running full episodes, computing the return at each timestep, and updating θ in the direction of `G_t · ∇log π(a_t | s_t, θ)`. Derive the update rule from the log-derivative trick. Show why this works: actions followed by high returns get reinforced, actions followed by low returns get suppressed.

## Map It

Trace the data flow: environment step → action sampled from π → reward → episode buffer → return calculation (cumulative sum, possibly discounted) → loss = -Σ G_t · log π(a_t | s_t) → backprop → θ update. Identify where variance enters (Monte Carlo return estimates are noisy) and why baselines matter.

## Build It

Full working implementation of REINFORCE on CartPole-v1 (gymnasium) using PyTorch. Policy network (linear → softmax), episode collection loop, return computation, gradient update. No tricks — just the raw algorithm so every component is visible. Print episode return every 10 episodes to confirm learning.

## Use It

REINFORCE solves the problem of optimizing a sequence of decisions when feedback is delayed and sparse. This is foundational for Zone 3 (Orchestration & Decisioning) in GTM — any system that must choose among actions across multiple steps, observe the final outcome, and adjust its selection policy. Specific example: a multi-touch outreach sequence where you only observe converted/not-converted at the end, and need credit assignment across each touch. The algorithm doesn't cleanly map to a current Clay primitive, so the redirect is: this is foundational for Zone 3.

## Ship It

- **Easy:** Modify the discount factor γ in the provided implementation and observe how it changes learning speed and stability. Print a comparison table.
- **Medium:** Add a baseline (subtract the mean return from each G_t) and demonstrate variance reduction — print gradient magnitudes before and after.
- **Hard:** Replace CartPole with a custom environment that models a 3-step outreach sequence (send email → wait → follow up → observe conversion), define a stochastic conversion function, and train REINFORCE to discover the optimal action sequence.

---

**Learning Objectives:**
1. Implement the REINFORCE update rule from the policy gradient theorem in working code
2. Explain why `∇log π(a|s)` correctly scales gradient contributions by action probability
3. Detect high-variance gradient estimates from Monte Carlo returns and mitigate them with a baseline
4. Compare learning curves with and without discounting on an episodic environment
5. Build a training loop that collects full episodes before applying a single parameter update