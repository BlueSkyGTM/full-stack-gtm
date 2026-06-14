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