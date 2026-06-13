# Autoencoders & Variational Autoencoders (VAE)

---

## Beat 1: Hook

An autoencoder learns to compress data into a smaller representation and reconstruct it. When the reconstruction is bad, you've found an anomaly — or a bad fit. This is the compression-reconstruction pattern that underpins representation learning.

---

## Beat 2: Concept

### Autoencoder Mechanism

An autoencoder is two neural networks glued at a bottleneck. The encoder maps input $\mathbf{x}$ to a latent vector $\mathbf{z}$ of lower dimensionality. The decoder maps $\mathbf{z}$ back to a reconstruction $\hat{\mathbf{x}}$. The model trains by minimizing reconstruction loss — typically mean squared error between $\mathbf{x}$ and $\hat{\mathbf{x}}$.

The bottleneck forces the network to discard noise and retain structure. The latent space $\mathbf{z}$ is a compressed representation that captures the most important features of the input distribution.

```
Input x → Encoder → z (latent) → Decoder → Reconstruction x̂
              ↑                    ↑
         Compresses to        Expands from
         fewer dimensions     fewer dimensions
```

Key property: the model learns without labels. The input is the supervisory signal.

### Variational Autoencoder Mechanism

A standard autoencoder maps each input to a single point in latent space. A VAE maps each input to a *distribution* — specifically, parameters $\mu$ and $\sigma$ of a Gaussian. The encoder produces $q_\phi(\mathbf{z}|\mathbf{x})$, and we sample $\mathbf{z} \sim \mathcal{N}(\mu, \sigma^2)$.

This introduces two losses:
1. **Reconstruction loss**: same as standard autoencoder — how well does $\hat{\mathbf{x}}$ match $\mathbf{x}$?
2. **KL divergence**: $\mathcal{D}_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| \mathcal{N}(0, I))$ — penalizes the learned distribution for deviating from a standard normal.

The KL term regularizes the latent space so that:
- Points cluster around the origin
- The space is continuous and smooth
- You can sample from $\mathcal{N}(0, I)$ and decode to generate new data

The **reparameterization trick** makes this differentiable: instead of sampling $\mathbf{z} \sim \mathcal{N}(\mu, \sigma^2)$ directly, compute $\mathbf{z} = \mu + \sigma \cdot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$. Gradients flow through $\mu$ and $\sigma$.

### Comparison

| Property | Autoencoder | VAE |
|----------|-------------|-----|
| Latent space | Arbitrary points | Learned distributions |
| Generative | No — gaps in latent space | Yes — smooth, continuous space |
| Loss | Reconstruction only | Reconstruction + KL divergence |
| Sampling | Not meaningful | Decode $z \sim \mathcal{N}(0, I)$ |
| Anomaly detection | Reconstruction error | Reconstruction error + KL |

---

## Beat 3: Build It

### Standard Autoencoder

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

np.random.seed(42)
torch.manual_seed(42)

normal_data = np.random.randn(1000, 10).astype(np.float32)
normal_data = normal_data * 0.5 + 1.0

anomalous_data = np.random.randn(50, 10).astype(np.float32)
anomalous_data = anomalous_data * 3.0 + 5.0

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

model = Autoencoder(input_dim=10, latent_dim=3)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

dataset = TensorDataset(torch.tensor(normal_data))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(50):
    total_loss = 0
    for batch in dataloader:
        x = batch[0]
        x_hat = model(x)
        loss = criterion(x_hat, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: loss = {total_loss / len(dataloader):.6f}")

model.eval()
with torch.no_grad():
    normal_recon = model(torch.tensor(normal_data))
    anomalous_recon = model(torch.tensor(anomalous_data))
    
    normal_errors = torch.mean((normal_recon - torch.tensor(normal_data))**2, dim=1)
    anomalous_errors = torch.mean((anomalous_recon - torch.tensor(anomalous_data))**2, dim=1)

print(f"\nNormal data reconstruction error:  mean={normal_errors.mean():.4f}, std={normal_errors.std():.4f}")
print(f"Anomalous data reconstruction error: mean={anomalous_errors.mean():.4f}, std={anomalous_errors.std():.4f}")
print(f"Anomaly signal strength: {anomalous_errors.mean() / normal_errors.mean():.1f}x")
```

### Variational Autoencoder

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

data = np.concatenate([
    np.random.multivariate_normal([2, 2], [[0.5, 0.1], [0.1, 0.5]], 500),
    np.random.multivariate_normal([-2, -2], [[0.5, -0.1], [-0.1, 0.5]], 500),
], axis=0).astype(np.float32)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder_mu = nn.Linear(input_dim, latent_dim)
        self.encoder_logvar = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    
    def encode(self, x):
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

def vae_loss(x_hat, x, mu, logvar):
    recon = nn.functional.mse_loss(x_hat, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl, recon, kl

model = VAE(input_dim=2, latent_dim=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

x_tensor = torch.tensor(data)

for epoch in range(100):
    x_hat, mu, logvar = model(x_tensor)
    loss, recon, kl = vae_loss(x_hat, x_tensor, mu, logvar)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: total={loss.item():.1f}, recon={recon.item():.1f}, kl={kl.item():.1f}")

model.eval()
with torch.no_grad():
    z_samples = torch.randn(5, 2)
    generated = model.decode(z_samples)
    print("\nGenerated samples from z ~ N(0, I):")
    print(generated.numpy())
    
    mu, logvar = model.encode(x_tensor[:10])
    print(f"\nLatent mu (first 5): {mu[:5].numpy()}")
    print(f"Latent std (first 5): {torch.exp(0.5 * logvar)[:5].numpy()}")
```

**Exercise Hooks:**
- Easy: Modify the autoencoder latent dimension and observe how reconstruction error changes. Document the threshold where compression hurts quality.
- Medium: Replace the Gaussian prior in the VAE with a mixture-of-Gaussians prior. Measure whether cluster separation in latent space improves.
- Hard: Train a VAE on account firmographic features and build an anomaly detector that flags accounts with reconstruction error above the 95th percentile. Evaluate precision at k=50.

---

## Beat 4: Use It

### GTM Redirect: Zone 30 Intent Signals — Anomaly Detection on Account Behavior

Autoencoders detect anomalies via reconstruction error. In GTM, this maps to **Zone 30 (Intent Signals)**: identifying accounts whose behavioral patterns deviate from the baseline. [CITATION NEEDED — concept: autoencoder-based anomaly detection for intent signal scoring in GTM pipelines]

The pattern:
1. Train an autoencoder on historical account engagement vectors (page visits, email opens, content downloads, time-on-site).
2. New accounts pass through the encoder-decoder. High reconstruction error = the account's pattern doesn't match the learned distribution.
3. Flag high-error accounts for manual review or automated outreach sequence.

This is not a classification problem — you don't have labeled "intent" data. It's an unsupervised deviation detector.

```python
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)

class AccountAnomalyDetector:
    def __init__(self, feature_dim, latent_dim=3, threshold_percentile=95):
        self.model = Autoencoder(feature_dim, latent_dim)
        self.threshold_percentile = threshold_percentile
        self.threshold = None
    
    def fit(self, account_features, epochs=100, lr=1e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        x = torch.tensor(account_features, dtype=torch.float32)
        
        for epoch in range(epochs):
            x_hat = self.model(x)
            loss = criterion(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        self.model.eval()
        with torch.no_grad():
            recon = self.model(x)
            errors = torch.mean((recon - x)**2, dim=1).numpy()
        self.threshold = np.percentile(errors, self.threshold_percentile)
        return self
    
    def score(self, new_accounts):
        x = torch.tensor(new_accounts, dtype=torch.float32)
        with torch.no_grad():
            recon = self.model(x)
            errors = torch.mean((recon - x)**2, dim=1).numpy()
        return errors
    
    def flag(self, new_accounts):
        errors = self.score(new_accounts)
        return errors > self.threshold, errors

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

np.random.seed(42)
baseline_accounts = np.random.exponential(scale=1.0, size=(500, 5)).astype(np.float32)
baseline_accounts[:, 0] = baseline_accounts[:, 0] * 2 + 1
baseline_accounts[:, 1] = baseline_accounts[:, 1] * 0.5

detector = AccountAnomalyDetector(feature_dim=5, latent_dim=2)
detector.fit(baseline_accounts, epochs=80)

new_accounts = np.vstack([
    np.random.exponential(scale=1.0, size=(20, 5)).astype(np.float32),
    np.random.exponential(scale=5.0, size=(5, 5)).astype(np.float32),
])

flags, scores = detector.flag(new_accounts)
print(f"Threshold (p95): {detector.threshold:.4f}")
print(f"Flagged {flags.sum()} of {len(new_accounts)} accounts")
print(f"Score range: {scores.min():.4f} — {scores.max():.4f}")
for i, (f, s) in enumerate(zip(flags, scores)):
    if f:
        print(f"  Account {i}: score={s:.4f} [FLAGGED]")
```

**Exercise Hooks:**
- Easy: Change the threshold percentile and observe the trade-off between flagged count and false positive rate on synthetic data.
- Medium: Wire the anomaly detector into a Clay waterfall as a conditional enrichment step — accounts above threshold trigger a deeper enrichment lookup.
- Hard: Train separate autoencoders per industry vertical. Compare global-vs-segmented anomaly detection precision on a held-out set of accounts with known conversion labels.

---

## Beat 5: Ship It

### Production Concerns

**Latent space drift.** The autoencoder learns a fixed representation of the training distribution. As account behavior shifts seasonally, reconstruction error baselines drift. Schedule retraining on a rolling window — weekly or monthly, depending on data velocity.

**Threshold calibration.** The 95th-percentile threshold is a starting point. In production, calibrate against downstream conversion data: what reconstruction error threshold maximizes pipeline generated per outreach?

**Batch vs. real-time.** Autoencoder inference is cheap — a forward pass on 10 features takes microseconds on CPU. Scoring a full CRM nightly is feasible. Scoring webhooks in real-time requires batching or async processing.

**Monitoring.** Track the mean reconstruction error of incoming accounts over time. If mean error rises, the training distribution has shifted. If mean error drops to near-zero, the model is memorizing — latent dimension may be too high.

```python
import torch
import torch.nn as nn
import numpy as np
import json

torch.manual_seed(42)

class ProductionAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def encode(self, x):
        with torch.no_grad():
            return self.encoder(x)
    
    def to_config(self):
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
        }

def train_and_export(training_data, input_dim, latent_dim=3, epochs=100):
    model = ProductionAutoencoder(input_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    x = torch.tensor(training_data, dtype=torch.float32)
    
    for epoch in range(epochs):
        x_hat = model(x)
        loss = criterion(x_hat, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        recon = model(x)
        errors = torch.mean((recon - x)**2, dim=1).numpy()
    
    config = model.to_config()
    config["threshold_p95"] = float(np.percentile(errors, 95))
    config["threshold_p99"] = float(np.percentile(errors, 99))
    config["mean_error"] = float(np.mean(errors))
    config["training_samples"] = len(training_data)
    
    print("Production config:")
    print(json.dumps(config, indent=2))
    
    return model, config

np.random.seed(42)
crm_features = np.random.randn(2000, 8).astype(np.float32)
crm_features[:, 0] = np.abs(crm_features[:, 0]) * 10
crm_features[:, 1] = np.abs(crm_features[:, 1]) * 0.1

model, config = train_and_export(crm_features, input_dim=8, latent_dim=3)

test_batch = np.random.randn(10, 8).astype(np.float32)
test_batch[9] = test_batch[9] * 5 + 3

with torch.no_grad():
    scores = torch.mean((model(torch.tensor(test_batch)) - torch.tensor(test_batch))**2, dim=1).numpy()

print(f"\nLive scoring batch (n={len(test_batch)}):")
for i, s in enumerate(scores):
    label = "ANOMALY" if s > config["threshold_p95"] else "normal"
    print(f"  [{label}] score={s:.4f}")
```

**Exercise Hooks:**
- Easy: Export the model config and thresholds to a JSON file. Write a scoring function that loads the config and applies thresholds to new data.
- Medium: Implement a drift monitor that compares the rolling mean error of the last 7 days against the training mean error. Alert when drift exceeds 2 standard deviations.
- Hard: Build a retraining pipeline that retrains the autoencoder weekly on the last 90 days of account data. Compare pre- and post-retraining thresholds.

---

## Beat 6: Own It

### Diagnostic Questions

1. A trained autoencoder has near-zero reconstruction error on all inputs, including anomalous ones. What hyperparameter do you change, and in which direction?
2. A VAE generates blurry, averaged outputs. Which loss component is dominating, and how do you rebalance?
3. You deploy an autoencoder anomaly detector trained on Q1 data. In Q3, the false positive rate spikes. What happened, and what is the fix?
4. Explain why a standard autoencoder cannot generate new data samples but a VAE can. Use the concept of latent space continuity in your answer.
5. You have 50,000 accounts with 200 features each. Latent dimension is set to 150. What will happen during training, and what should you set it to instead?

**Exercise Hooks:**
- Easy: Reduce the autoencoder to latent_dim=1 on the 10-feature dataset. Print reconstruction errors and explain why information loss concentrates in certain features.
- Medium: Implement a denoising autoencoder by corrupting inputs with Gaussian noise before encoding. Measure whether the denoised reconstructions are cleaner than the corrupted inputs — print PSNR for both.
- Hard: Train a VAE on a real dataset (e.g., tabular firmographics from a public source). Sample 100 latent vectors, decode them, and evaluate whether the generated samples are plausibly realistic by comparing their feature distributions to the training set using Kolmogorov-Smirnov tests.

---

## Learning Objectives

1. **Implement** a standard autoencoder in PyTorch and measure reconstruction error as an anomaly signal.
2. **Compare** autoencoder and VAE architectures by identifying the role of the KL divergence term and reparameterization trick.
3. **Deploy** an autoencoder-based anomaly detector with a calibrated threshold on account feature data.
4. **Diagnose** training failures caused by latent dimension misconfiguration (overcomplete bottleneck, KL collapse).
5. **Evaluate** generated samples from a VAE decoder by comparing their feature distributions to the training distribution.