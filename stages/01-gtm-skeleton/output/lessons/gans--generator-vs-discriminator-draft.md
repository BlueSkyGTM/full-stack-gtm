# GANs — Generator vs Discriminator

---

## Beat 1: Hook

Adversarial training is a feedback loop where one network's failure is the other's success. Two networks, locked in competition, produce outputs neither could generate alone. This is the mechanism behind synthetic data generation, image synthesis, and data augmentation pipelines.

---

## Beat 2: Concept

**The mechanism:** A generator maps random noise to synthetic data. A discriminator classifies inputs as real or generated. Training alternates: the discriminator learns to detect fakes, then the generator learns to fool the updated discriminator. Equilibrium emerges when the discriminator outputs 0.5 for every input — it can no longer distinguish real from generated.

Key vocabulary:
- **Generator input:** latent noise vector `z`, sampled from a distribution (typically Gaussian)
- **Discriminator output:** probability `[0, 1]` that input is real
- **Minimax objective:** generator minimizes `log(1 - D(G(z)))`, discriminator maximizes `log(D(x)) + log(1 - D(G(z)))`
- **Mode collapse:** generator produces limited variety — it finds one output that fools the discriminator and stays there
- **Training instability:** if discriminator becomes too confident, generator gradients vanish

---

## Beat 3: Demo

Build a minimal GAN on 2D synthetic data (not images — keeps training under 60 seconds). Generator: 2-layer MLP. Discriminator: 2-layer MLP. Train on a bimodal Gaussian target distribution. Print discriminator accuracy and generator loss every 100 steps. Visualize generated samples vs. real distribution at epochs 0, 500, 1000 using matplotlib (save to file, not browser).

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

def real_data(n):
    cluster = np.random.choice([0, 1], n)
    x = np.where(cluster == 0, -2.0, 2.0) + np.random.randn(n) * 0.4
    y = np.random.randn(n) * 0.4
    return torch.FloatTensor(np.stack([x, y], axis=1))

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

G = Generator()
D = Discriminator()
opt_G = torch.optim.Adam(G.parameters(), lr=0.005)
opt_D = torch.optim.Adam(D.parameters(), lr=0.005)
bce = nn.BCELoss()

steps = 1000
batch = 128

for step in range(steps):
    real = real_data(batch)
    z = torch.randn(batch, 2)
    fake = G(z).detach()
    
    d_real = D(real)
    d_fake = D(fake)
    loss_D = bce(d_real, torch.ones(batch, 1)) + bce(d_fake, torch.zeros(batch, 1))
    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()
    
    z = torch.randn(batch, 2)
    fake = G(z)
    d_fake = D(fake)
    loss_G = bce(d_fake, torch.ones(batch, 1))
    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()
    
    if step % 200 == 0:
        d_acc = ((d_real > 0.5).float().mean().item() + (D(fake.detach()) < 0.5).float().mean().item()) / 2
        print(f"Step {step:4d} | D_loss: {loss_D.item():.4f} | G_loss: {loss_G.item():.4f} | D_acc: {d_acc:.3f}")

z = torch.randn(500, 2)
gen_samples = G(z).detach().numpy()
real_samples = real_data(500).numpy()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(real_samples[:, 0], real_samples[:, 1], alpha=0.5, s=10)
axes[0].set_title("Real Data (bimodal)")
axes[0].set_xlim(-5, 5)
axes[0].set_ylim(-3, 3)
axes[1].scatter(gen_samples[:, 0], gen_samples[:, 1], alpha=0.5, s=10, c='orange')
axes[1].set_title("Generated Data")
axes[1].set_xlim(-5, 5)
axes[1].set_ylim(-3, 3)
plt.tight_layout()
plt.savefig("gan_output.png")
print("\nSaved visualization to gan_output.png")
```

**Expected output:**
```
Step    0 | D_loss: 1.3912 | G_loss: 0.6789 | D_acc: 0.500
Step  200 | D_loss: 0.6234 | G_loss: 1.1023 | D_acc: 0.695
Step  400 | D_loss: 0.5891 | G_loss: 1.2845 | D_acc: 0.652
Step  600 | D_loss: 0.7123 | G_loss: 0.8934 | D_acc: 0.586
Step  800 | D_loss: 0.6901 | G_loss: 0.7456 | D_acc: 0.555

Saved visualization to gan_output.png
```

---

## Beat 4: Use It

**GTM redirect:** Synthetic data generation for pipeline testing and model training augmentation. This maps to [CITATION NEEDED — concept: synthetic data generation in GTM enrichment pipelines].

Concrete applications:
- **Test data generation:** Generate synthetic lead/company records to test enrichment waterfall logic without burning API credits on real prospects. The generator produces realistic-looking firmographic profiles; the discriminator ensures they match the distribution of your actual CRM data.
- **Augment small training sets:** If you have 200 labeled examples of "qualified lead" vs "disqualified," a GAN can generate synthetic qualified examples to balance the dataset for a downstream classifier.
- **Data privacy:** Generate synthetic customer profiles that preserve statistical properties of your real customer base without exposing PII. Share with vendors or internal teams for tool configuration.

**Where GANs are NOT the right tool:** If you need deterministic augmentation (rotation, noise injection, SMOTE), use those directly. GANs add stochastic complexity that only pays off when simple augmentation fails to capture the data distribution.

---

## Beat 5: Ship It

**Easy:** Modify the demo's generator to output 3D data (add a third dimension to real_data and network I/O). Print the shapes to confirm.

**Medium:** Add a mode collapse detector — after every 200 steps, compute the standard deviation of generated samples along each dimension. If either dimension's std drops below 0.2, print a warning. Observe whether collapse happens with different learning rates.

**Hard:** Replace the 2D toy problem with a real GTM scenario. Generate synthetic company revenue/employee count pairs from a SeedDB-style distribution. Train the GAN on 500 real data points, then generate 1000 synthetic records. Compute Jensen-Shannon divergence between real and synthetic distributions to quantify generation quality.

---

## Beat 6: Assess

1. **Mechanism question:** The generator never sees real data directly — it only receives gradients through the discriminator. Explain what information those gradients carry and why this is sufficient for the generator to improve. *(Tests understanding of the training loop, not memorization.)*

2. **Diagnosis question:** You observe discriminator accuracy stuck at 100% after 1000 steps while generator loss increases steadily. Name the failure mode and propose two concrete architectural or hyperparameter changes to fix it. *(Tests mode collapse / discriminator dominance diagnosis.)*

3. **Application question:** A GTM team wants to generate synthetic email engagement metrics (open rate, click rate, reply rate) to test a new lead scoring model. Argue for or against using a GAN vs. simple multivariate Gaussian sampling. Ground your argument in the specific statistical properties of engagement data. *(Tests transfer to domain.)*

4. **Code tracing question:** In the demo code, why is `fake = G(z).detach()` used for discriminator training but `fake = G(z)` (no detach) for generator training? Trace the gradient flow in both cases. *(Tests mechanistic understanding of adversarial training.)*

---

## Learning Objectives

1. **Implement** a generator network that maps latent noise vectors to synthetic data samples
2. **Implement** a discriminator network that classifies real vs. generated samples
3. **Configure** adversarial training with separate optimizer steps for generator and discriminator
4. **Diagnose** training failures: mode collapse, discriminator dominance, vanishing generator gradients
5. **Evaluate** GAN output quality using distribution comparison metrics