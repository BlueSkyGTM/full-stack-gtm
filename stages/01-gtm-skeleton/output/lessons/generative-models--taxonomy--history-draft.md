# Generative Models — Taxonomy & History

---

## Beat 1: Set It

**Description:** Opens with the core question: given a training distribution, how do you *sample* new examples that look like they came from it? Frames the difference between discriminative models (predict labels) and generative models (approximate distributions). Introduces the five families: autoregressive, VAE, GAN, flow-based, diffusion.

**Hook:** "Every 'AI-generated' thing you've seen comes from one of five mechanism families. Two of them are already obsolete in practice. This lesson is about why."

---

## Beat 2: See It

**Description:** Visual walkthrough of each family's mechanism using minimal 1D/2D examples. Shows what a mixture of Gaussians looks like when modeled by each approach. Demonstrates mode collapse (GAN), blurry samples (VAE), sequential cost (autoregressive), and compute cost (diffusion).

**Hook:** "You'll see the same target distribution through five different lenses. The artifacts tell you which family you're looking at."

---

## Beat 3: Shape It

**Description:** Formal taxonomy with mechanism-first explanations:

| Family | Mechanism | Tractability | Quality | Diversity |
|--------|-----------|-------------|---------|-----------|
| Autoregressive | Chain rule factorization of joint | Exact likelihood | High | High (but slow) |
| VAE | Variational bound on log-likelihood | Lower bound | Blurry | High |
| GAN | Adversarial minimax game | No likelihood | Sharp | Mode collapse risk |
| Flow | Invertible transformations | Exact likelihood | Moderate | High |
| Diffusion | Iterative denoising from Gaussian | Exact (via bound) | High | High |

Historical arc: VAEs (2013) → GANs (2014) → Flows (2015) → Transformers/autoregressive dominance (2017-2019) → Diffusion (2020-2021) → Current convergence.

**Mechanism deep dives:**
- Autoregressive: $p(x) = \prod p(x_i | x_{<i})$ — exact but $O(n)$ sampling
- VAE: encoder maps to latent $z$, decoder reconstructs; KL divergence regularizes; posterior collapse problem
- GAN: generator vs discriminator; Nash equilibrium is the goal; training instability is the reality
- Flow: bijections with tractable Jacobian determinants; $f^{-1}$ gives you exact density
- Diffusion: forward process adds noise on schedule; reverse process learns to denoise; classifier-free guidance for controllability

**Why autoregressive won for language, diffusion won for images:** modality alignment with mechanism strengths.

---

## Beat 4: Solve It

**Exercise Hooks:**

- **Easy:** Given pseudocode for each family, identify which family it implements. Match mechanism to name.
- **Medium:** Trace a single forward + reverse pass through a minimal diffusion process on a 1D distribution. Print the noise schedule, the noisy samples at each step, and the final denoised output.
- **Hard:** Implement a toy GAN and a toy VAE on the same 2D dataset. Compare sample quality, mode coverage, and training stability. Document which fails and why.

---

## Beat 5: Show It

**Description:** Student selects one generative family and writes a mechanism comparison against another family for a specific use case (text, images, audio, tabular). Must argue from mechanism properties, not marketing claims. Deliverable: a decision matrix with mechanism-justified recommendations.

**GTM connection seeded:** "If you were generating personalized outreach copy at scale, which family would you pick and why? (Hint: the answer is autoregressive, and the reason is tractable conditioning on context.)"

---

## Beat 6: Ship It

**Description:** Connects taxonomy to production model selection. Explains why GPT-4 is autoregressive, DALL-E 3 is diffusion-based, and why you won't ship a GAN or VAE in 2024. Maps generative architecture choice to GTM use cases:

- **Zone 2 (Enrichment):** Autoregressive models for structured data completion (company descriptions, IC predictions)
- **Zone 3 (Engagement):** Autoregressive models for personalized copy generation via Clay workflows [CITATION NEEDED — concept: Clay GTM AI enrichment integrations]
- **Foundational:** Flow-based and VAE mechanisms inform understanding of latent space manipulation, even if you won't deploy them directly

**GTM Redirect:** This lesson is foundational for Zone 2 (Enrichment) and Zone 3 (Engagement). When you configure an LLM-powered enrichment in Clay, you're using an autoregressive model conditioned on firmographic context. The taxonomy explains why autoregressive — not diffusion, not GANs — is the architecture that ships for text generation tasks.

---

## Learning Objectives

1. **Compare** the five generative model families by their mechanism for approximating data distributions.
2. **Map** the historical progression from VAEs (2013) through diffusion models (2021+) to current production architectures.
3. **Explain** why autoregressive models dominate language generation while diffusion models dominate image synthesis, using mechanism properties as evidence.
4. **Identify** which generative architecture is appropriate for a given output modality and production constraint (latency, quality, controllability).
5. **Evaluate** sample outputs from different model families and diagnose which family produced them based on characteristic artifacts (blur, mode collapse, coherence).