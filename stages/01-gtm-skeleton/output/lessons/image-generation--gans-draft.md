# Image Generation — GANs

## Beat 1: Hook

Two neural networks locked in a zero-sum game: one fabricates images, the other polices them. The equilibrium—when it exists—produces a generator that manufactures synthetic data indistinguishable from real samples. This adversarial framing, introduced by Goodfellow et al. in 2014, kicked off the generative AI era. Most production image pipelines have moved to diffusion, but the minimax training mechanism underpins enough downstream architectures that skipping it creates gaps later.

---

## Beat 2: Concept

**The minimax mechanism.** A generator *G* maps random noise *z* ~ *p(z)* to synthetic samples *G(z)*. A discriminator *D* receives either a real sample *x* or a fake sample *G(z)* and outputs a probability that its input is real. *G*'s objective: maximize *D*'s error rate. *D*'s objective: correctly classify real vs. fake. The formal game:

```
min_G max_D  E_x~p_data[log D(x)] + E_z~p_z[log(1 - D(G(z)))]
```

**Training dynamics, not architecture.** The insight is not the network topology—*G* and *D* can each be any differentiable function. The insight is that competitive pressure from *D* forces *G* to learn the data distribution implicitly, without ever computing an explicit likelihood.

**Pathologies.** Mode collapse (*G* outputs near-identical samples), vanishing discriminator gradients (when *D* wins too decisively, *G* receives no usable signal), and oscillatory training (neither network converges). WGAN, LSGAN, and spectral normalization each attack a specific failure mode—covered in Extend It.

---

## Beat 3: Use It

**GTM cluster redirect: foundational for Zone 1 (AI Fundamentals).** GANs are not the default production choice for image generation in 2024—diffusion models hold that slot. The direct mechanism transfer is to **synthetic data augmentation for training downstream models**. If your pipeline needs more labeled images than you can source, a conditional GAN (cGAN) can manufacture class-conditional samples to bulk up minority classes. The mechanism: condition *G* on a class label *y* so that *G(z, y)* produces a sample belonging to class *y*, then mix generated samples into your training set.

The GTM application is niche. [CITATION NEEDED — concept: GAN-based data augmentation improving downstream classifier performance in production GTM workflows]. If you are evaluating whether to adopt GANs for a GTM-specific task, the honest answer is: use diffusion unless your constraint is real-time single-sample generation latency, where a trained *G* is a single forward pass.

---

## Beat 4: Build It

**Exercise hooks (progressive):**

- **Easy.** Implement a minimal GAN on a 2D mixture-of-Gaussians dataset. Train until *D*'s accuracy hovers near 0.5. Print and inspect generated samples vs. real samples at epochs 0, 500, 2000. Observable output: scatter plots saved to disk, terminal print of *D* loss and *G* loss per epoch.

- **Medium.** Train a DCGAN (Radford et al., 2015) on Fashion-MNIST. Log *G* and *D* losses to a CSV file. At the end of training, print the final five rows of the CSV and the number of epochs where *D* accuracy exceeded 0.95 (a proxy for vanishing-gradient risk). Observable output: CSV file, terminal summary statistics.

- **Hard.** Introduce mode collapse intentionally by training *G* and *D* at a 1:5 update ratio. Then switch to 1:1 and add gradient penalty (WGAN-GP). Print a histogram of generated sample class distribution before and after the fix. Observable output: terminal histograms showing whether the generator recovered coverage of all ten Fashion-MNIST classes.

---

## Beat 5: Ship It

**Production considerations.**

Checkpoint *G* and *D* separately. At inference time you ship only *G*—*D* exists solely to train it. Quantify sample quality with FID (Fréchet Inception Distance) against a held-out real-image set; do not rely on visual inspection. Batch generation at inference to amortize any fixed costs.

**GTM redirect.** If you ship a trained generator into a GTM pipeline—say, generating synthetic profile images for A/B testing creatives—the metric that matters is not FID on ImageNet. It is the downstream metric: click-through rate, conversion rate, or whatever your funnel measures. FID is a proxy, and proxies detach from revenue at the worst moment. Measure the terminal metric.

---

## Beat 6: Extend It

- **WGAN and WGAN-GP.** Replace the binary classifier *D* with a critic that approximates Wasserstein distance. Gradient penalty replaces weight clipping. Result: smoother training gradients, reduced mode collapse.
- **Conditional GANs (cGAN, pix2pix).** Condition generation on structured input—a class label, a segmentation map, an edge map. The mechanism is concatenation of the conditioning signal into both *G* and *D*.
- **Progressive Growing (ProGAN) and StyleGAN.** Grow resolution during training. Map noise into an intermediate *W* space via a learned affine transform. Disentangle high-level attributes (pose, identity) from stochastic detail (freckles, hair).
- **Diffusion models (next lesson).** Replace adversarial training with iterative denoising. More stable training, higher sample diversity, slower inference. The tradeoff space between GANs and diffusion is the practical decision point for any image-generation pipeline.

---

## Learning Objectives (draft, 4)

1. **Implement** a minimax training loop for a generator-discriminator pair and confirm convergence via discriminator accuracy approaching 0.5.
2. **Diagnose** mode collapse by inspecting the class distribution of generated samples and comparing against the training set distribution.
3. **Compare** the computational and quality tradeoffs between GAN-based and diffusion-based image generation for a given throughput budget.
4. **Configure** a conditional GAN to generate class-specific samples and evaluate their utility as synthetic training data for a downstream classifier.