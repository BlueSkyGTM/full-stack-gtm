## Ship It

Deploying a diffusion model in production requires addressing two constraints that the training code above ignores: inference latency and output quality monitoring. The reverse sampling loop runs T sequential neural network evaluations — 300 in the code above, 1000 in the original DDPM paper. For real-time GTM applications (synthetic profile generation during a qualification pipeline), this latency is unacceptable without optimization.

The standard production approach is to reduce the number of sampling steps. DDIM (Song et al., 2021) reformulates the reverse process as non-Markovian, allowing deterministic sampling in 10–50 steps instead of 1000 with minimal quality loss. For tabular data (synthetic CRM records), even 50 steps is fast on CPU. For image generation, 20-step DDIM with a distilled model is the baseline — Stable Diffusion XL ships with a 30-step default.

Monitoring is the second concern. Diffusion models can silently degrade: the loss looks fine, samples look plausible individually, but the overall distribution drifts away from `p_data`. In a GTM context, this means your synthetic ICP examples gradually stop matching real high-value accounts, and your qualification model trained on them degrades. Ship with distribution-level metrics — maximum mean discrepancy (MMD) or classifier two-sample test (C2ST) between generated and real samples — checked on a schedule, not just per-sample loss.

For the CRM data hygiene application in Zone 08, the deployment pattern is batch generation, not real-time. You train the diffusion model on closed-won account feature vectors, generate a batch of synthetic positive examples overnight, validate them against held-out real data, and feed them into your qualification scoring pipeline. The enrichment loop — query, retrieve, refine — runs on existing vector infrastructure. The diffusion model augments the training set, not the retrieval path.

```python
import numpy as np
from scipy.stats import ks_2samp

np.random.seed(42)

real_icp = np.random.randn(200, 2) * 0.3 + np.array([2.0, 2.0])

generated_good = np.random.randn(200, 2) * 0.35 + np.array([2.0, 2.0])
generated_drifted = np.random.randn(200, 2) * 0.8 + np.array([1.5, 2.5])

def mmd_rbf(x, y, gamma=1.0):
    from scipy.spatial.distance import cdist
    k_xx = np.exp(-gamma * cdist(x, x, 'sqeuclidean'))
    k_yy = np.exp(-gamma * cdist(y, y, 'sqeuclidean'))
    k_xy = np.exp(-gamma * cdist(x, y, 'sqeuclidean'))
    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

ks_dim0_good = ks_2samp(real_icp[:, 0], generated_good[:, 0])
ks_dim0_drift = ks_2samp(real_icp[:, 0], generated_drifted[:, 0])
mmd_good = mmd_rbf(real_icp, generated_good, gamma=2.0)
mmd_drift = mmd_rbf