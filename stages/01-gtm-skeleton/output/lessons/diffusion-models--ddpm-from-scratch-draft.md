# Diffusion Models — DDPM from Scratch

## Learning Objectives

1. **Implement** the forward diffusion process that adds Gaussian noise over T timesteps
2. **Derive** the reverse sampling loop that iteratively denoises from pure noise
3. **Configure** a linear noise schedule and explain how beta values control the corruption rate
4. **Build** a minimal U-Net that predicts the noise component of a noisy input
5. **Evaluate** generated samples against training data distribution using visual inspection

---

## Beat 1: Hook — Why Diffusion Ate Generative AI

GANs dominated image generation for five years. Then DDPM showed up with a simpler training objective — predict the noise, not the image — and took over. Stable Diffusion, DALL-E, Midjourney all run variants of this mechanism. This lesson builds the core loop from nothing: noise addition, noise prediction, iterative denoising. No pretrained weights. No magic.

---

## Beat 2: Concept — The Two Processes

**Forward process (fixed):** Starting from a data point $x_0$, add Gaussian noise incrementally over $T$ steps according to a variance schedule $\beta_1, \ldots, \beta_T$. At step $t$, the noisy sample is:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})$$

The key mathematical trick: you can sample $x_t$ directly from $x_0$ using the cumulative product $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

where $\epsilon \sim \mathcal{N}(0, \mathbf{I})$.

**Reverse process (learned):** A neural network $\epsilon_\theta(x_t, t)$ learns to predict the noise $\epsilon$ that was added. Training loss (simplified):

$$L = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

**Sampling:** Starting from $x_T \sim \mathcal{N}(0, \mathbf{I})$, iteratively apply:

$$x_{t-1} = \frac{1}{\sqrt{1 - \beta_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

where $z \sim \mathcal{N}(0, \mathbf{I})$ for $t > 1$, and $z = 0$ for $t = 1$.

The noise schedule ($\beta_t$ values) controls how fast information is destroyed. Linear schedule: $\beta_t$ linearly increases from $\beta_{\text{start}}$ to $\beta_{\text{end}}$.

---

## Beat 3: Implement — DDPM on 2D Point Clouds

Code a complete DDPM that learns to generate samples from a 2D point cloud (e.g., a ring or spiral distribution). Observable output: scatter plots of real data, noisy data at various timesteps, and generated samples after training.

**Exercise hooks:**
- *Easy:* Modify the noise schedule parameters and observe how generation quality changes
- *Medium:* Replace the linear schedule with a cosine schedule and compare sample quality
- *Hard:* Extend from 2D point clouds to 28×28 MNIST digits using a minimal U-Net

---

## Beat 4: Use It — Synthetic Data and GTM Applications

Diffusion models generate synthetic training data and personalized visual content. In GTM contexts:

- **Synthetic ICP data generation:** When real conversion data is sparse, diffusion models can augment training sets for predictive models. The mechanism: train on real feature distributions, sample synthetic examples that preserve statistical properties.
- **Personalized visual content at scale:** Product marketing images, ad creative variations. [CITATION NEEDED — concept: diffusion-based ad personalization in GTM workflows]

Foundational for Zone 1 (ICP/Persona mapping) — the generative modeling pattern (learn distribution, sample new instances) applies broadly to synthetic data augmentation across GTM systems.

**Exercise hooks:**
- *Easy:* Generate synthetic 2D "conversion score" distributions and compare statistics against real data
- *Medium:* Train a simple classifier on real vs. synthetic data and measure performance delta

---

## Beat 5: Ship It — Inference Cost and Production Constraints

DDPM sampling requires $T$ forward passes (typically 1000). This is the production bottleneck. Practical mitigations:

- **DDIM sampling:** Deterministic variant that produces quality samples in 20-50 steps by skipping timesteps. Same trained model, different sampling procedure.
- **Consistency models:** Distill a diffusion model into a single-step generator. Trade training complexity for inference speed.
- **ONNX export and batching:** Batch multiple samples through the denoiser in parallel to saturate GPU throughput.

The cost calculus: at $T=1000$ steps with a 100M parameter U-Net on a T4 GPU, expect ~2 seconds per 256×256 image. DDIM at 50 steps: ~100ms. This determines whether you use diffusion for batch content generation (offline) or real-time personalization (needs distillation).

**Exercise hooks:**
- *Easy:* Benchmark sampling time for T=1000 vs T=100 vs T=50 on the 2D model
- *Medium:* Implement DDIM sampling and compare sample quality against DDPM at matched step counts

---

## Beat 6: Drill — Exercises

1. **Noise schedule ablation:** Train three models with $\beta_{\text{start}} \in \{0.0001, 0.001, 0.01\}$, fixed $\beta_{\text{end}} = 0.02$. Plot generated samples. Which destroys information too fast?

2. **Timestep conditioning ablation:** Remove the timestep $t$ embedding from the network (feed only $x_t$). Train and sample. Explain what breaks and why.

3. **Few-step generation:** Implement a skip-step sampler that evaluates $\epsilon_\theta$ at timesteps $\{1000, 800, 600, 400, 200, 1\}$ only. Compare output to full 1000-step sampling.

4. **Distribution matching metric:** Implement a simple 2D Kolmogorov-Smirnov test between real and generated samples. Use this as a quantitative benchmark across schedule variants.

---

## GTM Redirect Rules (Summary)

- **Use It section redirects to:** Zone 1 (ICP/Persona mapping) — synthetic data augmentation for sparse conversion datasets
- **Ship It section redirects to:** Content generation pipelines — the inference cost discussion applies directly to batch ad creative generation workflows
- **No forced connections:** DDPM is fundamentally a generative modeling technique. Its GTM relevance is in synthetic data and content generation, not in core GTM orchestration flows. The redirect is honest about this boundary.