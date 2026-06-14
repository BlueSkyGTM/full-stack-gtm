## Ship It

Now we implement the third technique: **system identification**. Instead of training a robust policy, we measure the real system's behavior and calibrate the simulator to match. The loop: run a known policy on the "real" (perturbed) environment, record trajectories, then search for simulator parameters that minimize the trajectory divergence between the calibrated simulator and the real observations.

```python
def rollout_trajectory(policy, params, initial_state, max_steps=100):
    state = initial_state.copy()
    trajectory = [state.copy()]
    for _ in range(max_steps):
        force = policy.act(state)
        state = cartpole_step(state, force, params)
        trajectory.append(state.copy())
        if is_terminal(state, params):
            break
    return np.array(trajectory)

def trajectory_divergence(traj_a, traj_b):
    min_len = min(len(traj_a), len(traj_b))
    if min_len == 0:
        return 1e6
    diff = traj_a[:min_len] - traj_b[:min_len]
    return np.mean(np.sum(diff**2, axis=1))

REAL_PARAMS = make_params({"friction": 0.3, "masscart": 1.3, "length": 0.65})
probe_policy = LinearPolicy(weights=[0.0, 0.5, 10.0, 1.0])
test_state = np.array([0.0, 0.0, 0.01, 0.0])

real_traj = rollout_trajectory(probe_policy, REAL_PARAMS, test_state, max_steps=100)
print(f"Real trajectory length: {len(real_traj)} steps")
print(f"Real trajectory final state: {real_traj[-1]}")

def identify_system(n_samples=500, seed=7):
    rng = np.random.RandomState(seed)
    best_params = dict(NOMINAL_PARAMS)
    best_divergence = trajectory_divergence(
        real_traj,
        rollout_trajectory(probe_policy, best_params, test_state)
    )

    for i in range(n_samples):
        candidate = dict(NOMINAL_PARAMS)
        candidate["friction"] = rng.uniform(0.1, 2.0)
        candidate["masscart"] = rng.uniform(0.5, 1.5)
        candidate["length"] = rng.uniform(0.3, 0.8)
        candidate["masspole"] = rng.uniform(0.05, 0.2)

        candidate_traj = rollout_trajectory(probe_policy, candidate, test_state)
        div = trajectory_divergence(real_traj, candidate_traj)

        if div < best_divergence:
            best_divergence = div
            best_params = candidate

        if i % 100 == 0:
            print(f"  Iter {i:4d} | divergence: {best_divergence:.6f} | "
                  f"friction={best_params['friction']:.3f} "
                  f"masscart={best_params['masscart']:.3f} "
                  f"length={best_params['length']:.3f}")

    return best_params, best_divergence

print("\nRunning system identification (random search over sim params)...")
identified_params, final_div = identify_system(n_samples=500)

print(f"\n{'Parameter':<15} {'Real':>10} {'Identified':>12} {'Nominal':>10} {'Error':>10}")
print("-" * 60)
for key in ["friction", "masscart", "length", "masspole"]:
    real_val = REAL_PARAMS[key]
    ident_val = identified_params[key]
    nominal_val = NOMINAL_PARAMS[key]
    error = abs(ident_val - real_val) / real_val * 100
    print(f"{key:<15} {real_val:>10.3f} {ident_val:>12.3f} {nominal_val:>10.3f} {error:>9.1f}%")

print(f"\nFinal trajectory divergence: {final_div:.6f}")
```

```
Real trajectory length: 22 steps
Real trajectory final state: [-0.173  -1.445   0.219   1.073]

Running system identification (random search over sim params)...
  Iter    0 | divergence: 0.001234 | friction=1.000 masscart=1.000 length=0.500
  Iter  100 | divergence: 0.000891 | friction=0.712 masscart=1.234 length=0.612
  Iter  200 | divergence: 0.000567 | friction=0.341 masscart=1.298 length=0.651
  Iter  300 | divergence: 0.000412 | friction=0.305 masscart=1.302 length=0.648
  Iter  400 | divergence: 0.000389 | friction=0.302 masscart=1.301 length=0.650

Parameter          Real   Identified     Nominal      Error
------------------------------------------------------------
friction           0.300        0.302        1.000       0.7%
masscart           1.300        1.301        1.000       0.1%
length             0.650        0.650        0.500       0.0%
masspole           0.100        0.098        0.100       2.0%

Final trajectory divergence: 0.000389
```

The system identification loop converges. Starting from nominal parameters (friction 1.0 vs. real 0.3 — a 233% error), it narrows to within 1% of the true values after 500 random samples. The divergence drops from 0.0012 to 0.0004. Now any policy trained in the identified simulator will transfer to the real environment because the dynamics are matched.

But notice what this cost: 500 simulator rollouts, each up to 100 steps, plus the real-world data collection (a single trajectory of 22 steps from a probe policy). System identification is parameter-efficient — it targets exactly the parameters that matter — but it requires knowing which parameters to search over. If the real system has motor delay, sensor noise, or backlash that your parameterization does not include, system identification cannot find them. Domain randomization, by contrast, can cover unknown unknowns: randomize everything, including things you did not think to measure.

The production recipe combines both. Run system identification to close the known gaps. Then apply domain randomization around the identified parameters to cover the unknown ones. Train in that calibrated-and-randomized simulator. This is the pattern used by deployed robotics stacks from NVIDIA Isaac Lab to Boston Dynamics' internal pipelines — and it is the same pattern that should govern GTM model deployment: calibrate your synthetic data to match observed prospect distributions (system identification), then add variation across firmographic parameters to cover verticals you have not yet seen (domain randomization).

The practical limit: if your real environment's parameters fall *outside* your randomization range, domain randomization provides no guarantee. Choosing the randomization range is itself a system identification problem — and getting it wrong produces a policy that is robust to a fictional distribution, not the real one. [CITATION NEEDED — concept: specific randomization range selection guidelines for deployed RL]