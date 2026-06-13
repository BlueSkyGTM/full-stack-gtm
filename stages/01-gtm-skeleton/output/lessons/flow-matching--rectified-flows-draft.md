# Flow Matching & Rectified Flows

## Hook

Diffusion models produce sharp outputs but require dozens to hundreds of sampling steps because their learned paths curve through latent space. Flow matching collapses that complexity: instead of learning a score function and running a reverse SDE, you learn a velocity field and run an ODE. Rectified flow goes further by forcing straight-line paths between noise and data. The result: same generative quality, fewer steps, simpler training objective. This is the architecture behind Stable Diffusion 3 and Black Forest Labs' Flux.

## Concept

Flow matching frames generative modeling as a **transport problem**. You have a source distribution (Gaussian noise) and a target distribution (your data). You learn a time-dependent vector field v(x,t) such that integrating the ODE dx/dt = v(x,t) from t=0 to t=1 moves samples from noise to data. The training objective minimizes the discrepancy between your model's predicted velocity and the optimal velocity along the path. Rectified flow (Liu et al. 2022) and flow matching (Lipman et al. 2022) independently converged on this idea: use straight-line interpolation x_t = (1-t)x_0 + t·x_1, which makes the optimal velocity field simply u_t = x_1 - x_0. No noise schedule design, no variance term, no score function—just predict the direction from noise to data.

## Mechanism

**The transport formulation.** Continuous normalizing flows (CNFs) model a time-dependent change-of-variables via an ODE. The probability path p_t satisfies the continuity equation ∂p/∂t + ∇·(p·v) = 0. If you can learn v such that p_0 = N(0,I) and p_1 ≈ p_data, you have a generative model. The problem: computing the marginal vector field over all data points is intractable.

**Conditional flow matching.** The key insight (Lipman et al. 2022): instead of matching the marginal vector field, match the conditional vector field. For each pair (x_0 ~ noise, x_1 ~ data), define a conditional path p_t(x|x_1) that interpolates between them. The conditional vector field u_t(x|x_1) can be computed in closed form for Gaussian conditional paths. The flow matching loss is:

L_FM = E_{t,x_1,x_t} [||v_θ(x_t, t) - u_t(x_t|x_1)||²]

This is an unbiased estimator of the marginal flow matching loss, but it only requires sampling pairs and evaluating the conditional field.

**Rectified flow.** Liu et al. (2022) proposed an even simpler conditional path: the straight line x_t = (1-t)x_0 + t·x_1 where x_0 ~ N(0,I) and x_1 ~ data. The conditional velocity is constant: u_t = x_1 - x_0. The loss becomes:

L_RF = E_{t,x_0,x_1} [||v_θ((1-t)x_0 + t·x_1, t) - (x_1 - x_0)||²]

No noise schedule. No diffusion coefficients. The model learns to predict the displacement from noise to data.

**Reflow.** A trained rectified flow model may produce curved paths at inference because the velocity field is imperfect. Reflow straightens them: generate (x_0, x̂_1) pairs using the current model (run the ODE from noise), then retrain on these new pairs. Each reflow iteration produces straighter trajectories, which means fewer Euler steps at inference.

**Sampling.** At inference, draw x_0 ~ N(0,I), then integrate dx/dt = v_θ(x,t) from t=0 to t=1 using any ODE solver. Euler method: x_{t+Δt} = x_t + Δt · v_θ(x_t, t). With well-straightened paths, 1-5 Euler steps suffice. Without reflow, you might need 20-50.

**Comparison to diffusion.** Diffusion models learn ∇log p_t(x) (the score), run a reverse SDE, and need careful noise schedule tuning. Flow matching learns v(x,t) directly, runs deterministic ODE, and the schedule is baked into the linear interpolation. The tradeoff: flow matching doesn't have the stochastic sampler option that diffusion's Langevin-type corrections provide, but in practice the determinism is a feature for reproducibility.

## Use It

**GTM Redirect:** Foundational for Zone I. Flow matching is the generative architecture inside current production image models (Stable Diffusion 3, Flux). If your GTM stack generates images, avatars, or visual content at scale, the model serving layer is increasingly a flow-matching model. Understanding the ODE solver tradeoffs (step count, solver order, runtime vs. quality) directly affects inference cost and throughput for any API-call-based image generation pipeline.

**Exercise hooks:**
- *Easy:* Train a rectified flow model on a 2D mixture of Gaussians. Visualize the learned velocity field as a vector field plot. Observe the straight-line paths.
- *Medium:* Implement Euler and midpoint ODE solvers. Compare sample quality at 1, 5, and 20 steps. Plot FID-like distance ( Wasserstein or MMD on 2D) vs. step count.
- *Hard:* Run one full reflow cycle: train, generate new (x_0, x̂_1) pairs via ODE integration, retrain on those pairs. Measure path curvature before and after using the formula E[∫||dx/dt|| dt] on sampled trajectories.

## Ship It

**GTM Redirect:** Foundational for Zone I. For practitioners deploying generative image APIs, the practical levers are: (1) step count—fewer steps = lower latency and cost per generation, and flow matching's straight paths enable sub-10-step generation where diffusion needs 20-50; (2) solver choice—Euler is fastest, RK45 is highest quality, Heun's midpoint is the practical compromise; (3) model precision—FP8/BF16 quantization of the velocity field network is safe because the ODE integration smooths out small perturbations, unlike diffusion's score function which is more sensitive. If you're self-hosting SD3 or Flux, these three knobs control your cost-per-image.

**Production considerations:**
- Batch the ODE solver: run all steps for a batch simultaneously, not one sample at a time. The velocity network is the bottleneck; batching amortizes the launch overhead.
- Cache the first step: for deterministic samplers, x_0 → x_{Δt} is often predictable. Pre-computing or caching the first velocity evaluation saves ~10% of inference time.
- Torch compile the velocity network: the ODE loop calls the same network N times with different (x, t) inputs—ideal for torch.compile inductor backend.
- Memory: ODE solvers need only the current state, unlike diffusion's ancestral sampling which can require keeping the noise schedule in VRAM. Flow matching is lighter on memory at the same model size.

**Exercise hooks:**
- *Easy:* Profile inference time for a pretrained FlowMatch model (e.g., from diffusers) at 4, 8, and 20 Euler steps. Report latency and VRAM usage.
- *Medium:* Swap Euler for DPMSolverMultistep on the same flow-matching checkpoint. Compare visual quality at matched step counts.
- *Hard:* Implement dynamic step selection: early-exit the ODE solver when ||v_θ(x_t, t)|| drops below a threshold (velocity magnitude indicates how far the path has left to travel). Measure the distribution of actual steps used across 1000 samples.

## Code It

```python
import torch
import torch.nn as nn
import math

def sample_data(n, distribution="moons"):
    if distribution == "moons":
        t = torch.rand(n) * 2 * math.pi
        x0 = torch.stack([torch.cos(t), torch.sin(t)], dim=1) * 1.0
        x1 = torch.stack([1.0 - torch.cos(t), 1.0 - torch.sin(t)], dim=1) * 1.0
        x = torch.cat([x0, x1], dim=0)
        x = x + torch.randn_like(x) * 0.1
        labels = torch.cat([torch.zeros(n), torch.ones(n)])
        idx = torch.randperm(2 * n)
        return x[idx], labels[idx]
    elif distribution == "gaussians":
        centers = [(0, 0), (2, 0), (0, 2), (2, 2), (1, 1)]
        samples_per = n // len(centers)
        parts = []
        for cx, cy in centers:
            parts.append(torch.randn(samples_per, 2) * 0.3 + torch.tensor([cx, cy]))
        x = torch.cat(parts, dim=0)
        return x[:n], torch.zeros(n)

class VelocityNet(nn.Module):
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x, t):
        t_input = t.unsqueeze(-1) if x.dim() > 1 else t.unsqueeze(0).unsqueeze(-1)
        inp = torch.cat([x, t_input.expand(x.shape[0], 1)], dim=-1)
        return self.net(inp)

def train_rectified_flow(data, epochs=3000, lr=1e-3, batch_size=256):
    model = VelocityNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = data.shape[0]

    for epoch in range(epochs):
        idx = torch.randint(0, n, (batch_size,))
        x1 = data[idx]
        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size)
        xt = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
        target_v = x1 - x0
        pred_v = model(xt, t)
        loss = nn.functional.mse_loss(pred_v, target_v)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

    return model

def sample_flow(model, n_samples=500, n_steps=8):
    x = torch.randn(n_samples, 2)
    dt = 1.0 / n_steps
    trajectory = [x.clone()]

    for step in range(n_steps):
        t = torch.full((n_samples,), step * dt)
        v = model(x, t)
        x = x + v * dt
        trajectory.append(x.clone())

    return x, trajectory

def compute_path_length(trajectory):
    total = 0.0
    for i in range(1, len(trajectory)):
        delta = trajectory[i] - trajectory[i-1]
        total += delta.norm(dim=-1).mean().item()
    return total

def reflow(model, data, epochs=1500, lr=1e-3, batch_size=256):
    print("\n--- Running reflow ---")
    with torch.no_grad():
        x0_init = torch.randn(data.shape[0], 2)
        x1_hat, _ = sample_flow(model, n_samples=data.shape[0], n_steps=20)

    paired_data = x1_hat.detach()

    model2 = VelocityNet()
    optimizer = torch.optim.Adam(model2.parameters(), lr=lr)
    n = paired_data.shape[0]

    for epoch in range(epochs):
        idx = torch.randint(0, n, (batch_size,))
        x1 = paired_data[idx]
        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size)
        xt = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
        target_v = x1 - x0
        pred_v = model2(xt, t)
        loss = nn.functional.mse_loss(pred_v, target_v)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Reflow Epoch {epoch:4d} | Loss: {loss.item():.6f}")

    return model2

torch.manual_seed(42)
data, _ = sample_data(2000, "moons")
print(f"Dataset shape: {data.shape}")
print(f"Data range: x=[{data[:,0].min():.2f}, {data[:,0].max():.2f}], y=[{data[:,1].min():.2f}, {data[:,1].max():.2f}]")

print("\n=== Training initial rectified flow ===")
model_v1 = train_rectified_flow(data, epochs=3000)

samples_v1, traj_v1 = sample_flow(model_v1, n_samples=500, n_steps=8)
print(f"\nV1 sample mean: {samples_v1.mean(dim=0).detach().numpy()}")
print(f"V1 sample std:  {samples_v1.std(dim=0).detach().numpy()}")
print(f"V1 path length (8 steps): {compute_path_length(traj_v1):.4f}")

model_v2 = reflow(model_v1, data, epochs=1500)

samples_v2, traj_v2 = sample_flow(model_v2, n_samples=500, n_steps=8)
print(f"\nV2 (reflowed) sample mean: {samples_v2.mean(dim=0).detach().numpy()}")
print(f"V2 (reflowed) sample std:  {samples_v2.std(dim=0).detach().numpy()}")
print(f"V2 path length (8 steps):  {compute_path_length(traj_v2):.4f}")

samples_few, traj_few = sample_flow(model_v2, n_samples=500, n_steps=3)
print(f"\nV2 with 3 steps - sample mean: {samples_few.mean(dim=0).detach().numpy()}")
print(f"V2 with 3 steps - path length:  {compute_path_length(traj_few):.4f}")
```

**Observable output:** Training loss convergence for both initial and reflowed models, sample statistics (mean/std) showing the model hits the target distribution, and path length measurements quantifying straightening after reflow. The reflowed model should show shorter path lengths at the same step count, confirming trajectory straightening.