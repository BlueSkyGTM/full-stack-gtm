# Flow Matching & Rectified Flows

## Learning Objectives

- Implement a conditional flow matching training loop that learns a velocity field on 2D distribution data.
- Compare sample quality across 1-step, 5-step, and 20-step Euler integration to quantify the payoff of straight-line paths.
- Trace the reflow procedure and measure its effect on path straightness and low-step generation quality.
- Evaluate how the choice of interpolation path (straight-line vs. curved) affects the number of integration steps required for faithful generation.

## The Problem

DDPM's reverse process is a 1000-step stochastic walk from `N(0, I)` back to the data distribution. DDIM collapsed it to 20–50 deterministic steps by integrating the probability flow ODE. You want fewer steps — ideally one. The blocker is that the ODE solving the reverse process is stiff: the learned path from noise to data curves through latent space, and a single Euler step along a curved path overshoots or undershoots the target manifold.

If you could train the model such that the path from noise to data was a straight line, a single Euler step from `t=0` to `t=1` would land on the data manifold. No curvature, no error accumulation, no need for 50 function evaluations at inference. The question is how to impose that straight-line geometry during training.

Score-based diffusion cannot do this directly. The score function `∇ log p_t(x)` describes the local gradient of the log-density, not the global transport direction. You are learning a local quantity and hoping the integration produces a globally sensible trajectory. Flow matching inverts the problem: define the trajectory first (a straight line from noise to data), then learn the velocity field that produces it.

## The Concept

Flow matching frames generative modeling as an optimal transport problem. You have a source distribution (Gaussian noise) and a target distribution (your data). You learn a time-dependent vector field `v_θ(x, t)` such that integrating the ODE `dx/dt = v_θ(x, t)` from `t=0` to `t=1` moves samples from noise to data. The training objective minimizes the discrepancy between the model's predicted velocity and the optimal velocity along the path.

The key insight from Lipman et al. (2022): instead of matching the marginal vector field — which requires integrating over all data points and is intractable — match the *conditional* vector field. For each pair `(x_0 ~ noise, x_1 ~ data)`, define a conditional path that interpolates between them. The conditional vector field can be computed in closed form for straight-line interpolation, making the loss a simple regression target.

Rectified flow (Liu et al., 2022) converged on the same idea with an even cleaner framing. Use straight-line interpolation `x_t = (1-t)·x_0 + t·x_1` where `x_0 ~ N(0, I)` and `x_1 ~ data`. The conditional velocity is constant: `u_t = x_1 - x_0`. No noise schedule design, no variance preservationchedule, no score function — the model predicts a displacement vector from noise to data. The loss is:

```
L_RF = E_{t, x_0, x_1} [ || v_θ((1-t)·x_0 + t·x_1, t) - (x_1 - x_0) ||² ]
```

```mermaid
flowchart LR
    subgraph DDPM["DDPM Reverse Process"]
        D1["t=T<br/>Noise"] --> D2["t=T-1"] --> D3["t=T-2"] --> DN["...<br/>~50 steps"] --> DE["t=0<br/>Data"]
    end
    
    subgraph RF["Rectified Flow"]
        R1["t=0<br/>Noise x₀"] -->|"v = x₁ - x₀"| R2["t=1<br/>Data x₁"]
        R1 -.->|"straight line"| R2
    end
    
    style DDPM fill:#fee,stroke:#c33
    style RF fill:#efe,stroke:#3c3
```

The practical payoff: because the target paths are straight, the ODE is non-stiff. A single Euler step produces a reasonable sample. Five steps produce near-converged quality. This is why Stable Diffusion 3 and Flux switched from DDPM-style score matching to flow matching — the same model architecture, trained with a different objective, samples in 4–8 steps instead of 30–50.

**Reflow.** A trained rectified flow model may still produce slightly curved paths at inference because `v_θ` is imperfect — the learned velocity field does not exactly equal `x_1 - x_0` everywhere. Reflow straightens them: generate `(x_0, x̂_1)` pairs by running the ODE from noise through the current model, then retrain a new model on these pairs. The new model learns the average trajectory the old model actually produced, which tends to be straighter than the original because the ODE integration averages out local velocity errors. Two reflow iterations typically suffice for a 1-step or 2-step sampler that matches 50-step DDPM quality.

## Build It

The fastest way to internalize flow matching is to train a velocity field on a 2D distribution and sample from it. No images, no transformers — just a 2-layer MLP learning to predict a displacement vector. The entire training loop fits in 40 lines.

The target distribution is a ring with Gaussian radial noise. The model must learn to transport standard Gaussian samples onto that ring. After training, we sample with Euler integration at different step counts and measure whether the output distribution matches the target.

```python
import math
import torch
import torch.nn as nn

torch.manual_seed(42)

def sample_ring(n, radius=2.0, spread=0.3):
    theta = torch.rand(n) * 2 * math.pi
    r = radius + torch.randn(n) * spread
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)

class VelocityField(nn.Module):
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
        return self.net(torch.cat([x, t.unsqueeze(1)], dim=1))

data = sample_ring(4000)
model = VelocityField()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch_size = 512
for epoch in range(3000):
    idx = torch.randint(0, len(data), (batch_size,))
    x1 = data[idx]
    x0 = torch.randn_like(x1)
    t = torch.rand(batch_size)
    
    xt = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1
    target = x1 - x0
    pred = model(xt, t)
    loss = ((pred - target) ** 2).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1:4d}  loss={loss.item():.6f}")

print(f"\nFinal velocity MSE: {loss.item():.6f}")
print(f"(Target velocity magnitude: ~{torch.norm(data, dim=1).mean().item():.3f})")
```

Run this and you see the loss converge to a small value. The model has learned the displacement field from noise to ring. Now sample from it:

```python
def euler_sample(model, n_samples, n_steps, dim=2):
    x = torch.randn(n_samples, dim)
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t_curr = torch.full((n_samples,), step * dt)
        v = model(x, t_curr)
        x = x + v * dt
    return x

def ring_stats(samples, name):
    radii = torch.norm(samples, dim=1)
    print(f"{name:20s}  mean_r={radii.mean():.3f}  "
          f"std_r={radii.std():.3f}  "
          f"range=[{radii.min():.2f}, {radii.max():.2f}]")

ring_stats(data[:1000], "Ground truth ring")
ring_stats(euler_sample(model, 1000, 1), "1-step Euler")
ring_stats(euler_sample(model, 1000, 5), "5-step Euler")
ring_stats(euler_sample(model, 1000, 20), "20-step Euler")
```

The ground truth ring has mean radius ~2.0 and std ~0.3. The 1-step sampler should land close — mean radius within 0.1–0.3 of the target, slightly higher variance because a single Euler step along an imperfect velocity field accumulates error. The 5-step and 20-step samplers tighten toward the ground truth distribution. The gap between 1-step and 20-step output is the gap that reflow closes.

## Use It

Reflow is the procedure that makes flow matching practical for single-step or few-step generation. Conditional flow matching trains on pairs `(x_0, x_1)` where `x_1` is a real data point — but the model's actual trajectory from `x_0` may curve, meaning the endpoint it reaches is not exactly `x_1`. Reflow fixes this by distilling the model's own trajectories into new training pairs.

The mechanics: run the current model's ODE from noise `x_0` to produce `x̂_1`. The pair `(x_0, x̂_1)` now defines a path that the model *actually follows* — by construction, integrating the model from `x_0` reaches `x̂_1`. Retrain a fresh velocity field on these pairs. The new model learns straighter paths because it is fitting the model's own input-output mapping, which is smoother than the raw noise-to-data mapping.

In a GTM enrichment waterfall, the same structure appears. A waterfall transports a prospect from an anonymous distribution (just an email or domain) to a qualified distribution (ICP-matched, enriched, scored). Each enrichment step — Clearbit, Apollo, LinkedIn scrape, ads library lookup — is one Euler integration step along a velocity field defined by your routing logic. A poorly calibrated waterfall is like an un-reflowed flow matching model: the path from anonymous to qualified curves through unnecessary hops, accumulating latency and API cost at each step.

```python
# Reflow: generate trajectory pairs from the trained model
def generate_pairs(model, n_pairs, n_steps=20, dim=2):
    x0 = torch.randn(n_pairs, dim)
    x = x0.clone()
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t_curr = torch.full((n_pairs,), step * dt)
        v = model(x, t_curr)
        x = x + v * dt
    return x0, x

x0_rect, x1_rect = generate_pairs(model, 4000, n_steps=20)

# Train a rectified model on these pairs
model_rect = VelocityField()
optimizer_rect = torch.optim.AdamW(model_rect.parameters(), lr=1e-3)

for epoch in range(3000):
    idx = torch.randint(0, len(x1_rect), (batch_size,))
    x1 = x1_rect[idx]
    x0 = x0_rect[idx]
    t = torch.rand(batch_size)
    
    xt = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1
    target = x1 - x0
    pred = model_rect(xt, t)
    loss = ((pred - target) ** 2).mean()
    
    optimizer_rect.zero_grad()
    loss.backward()
    optimizer_rect.step()

print("=== Before reflow ===")
ring_stats(euler_sample(model, 1000, 1), "Original 1-step")
ring_stats(euler_sample(model, 1000, 5), "Original 5-step")

print("\n=== After one reflow iteration ===")
ring_stats(euler_sample(model_rect, 1000, 1), "Reflowed 1-step")
ring_stats(euler_sample(model_rect, 1000, 5), "Reflowed 5-step")
```

The reflowed 1-step sampler should produce a tighter ring (lower `std_r`, mean radius closer to 2.0) than the original 1-step sampler. This is the empirical signature of path straightening: fewer integration steps produce the same quality because the velocity field is closer to a constant displacement along the trajectory.

The analogy to enrichment waterfalls is direct. If your waterfall runs Clearbit → Apollo → LinkedIn → ads library → spend estimator, that is a 5-step Euler integration of an enrichment velocity field. If you analyze which prospects actually converted after passing through, and rebuild the waterfall to skip steps that did not change the outcome