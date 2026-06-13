# Evaluation — FID, CLIP Score, Human Preference

---

## Hook

You've generated images. Now someone asks: "Are they any good?" This lesson covers the three evaluation axes practitioners actually use: distributional fidelity (FID), semantic alignment (CLIP Score), and ground truth (human preference). Without these, you're staring at outputs guessing.

---

## Concept

**FID (Fréchet Inception Distance)** measures how close the distribution of generated images is to a reference set of real images. It extracts feature vectors from Inception-v3, fits two Gaussians, and computes the Fréchet distance between them. Lower = closer to real. Mechanism: multivariate Gaussian distance in feature space. Weakness: reference-set dependent; can't tell you if one specific image is good.

**CLIP Score** measures alignment between a text prompt and a generated image. It computes cosine similarity between the CLIP text embedding and the CLIP image embedding. Higher = better alignment. Mechanism: shared embedding space. Weakness: optimizable by adversarial images; doesn't capture aesthetic quality.

**Human Preference** is the ground truth but doesn't scale. Collected via pairwise comparisons or Likert ratings. Mechanism: aggregated subjective judgment. Two variants: (1) preference ranking against reference images, (2) text-image alignment rating. Strength: captures what metrics miss. Weakness: expensive, noisy, not reproducible without strict protocols.

Key tension: FID evaluates populations. CLIP Score evaluates individual prompt-image pairs. Human preference evaluates both but inconsistently. You need at least two of three for any credible evaluation.

---

## Demo

**Demo 1 — Computing CLIP Score on a single prompt-image pair.** Load CLIP model, encode a text prompt and a local image, compute cosine similarity. Print the scalar score.

**Demo 2 — Computing FID between two image directories.** Load Inception-v3, extract features from two sets of images (real vs. generated), compute Fréchet distance. Print FID score.

**Demo 3 — Simulating a human preference protocol.** Present two images side-by-side with a prompt, record forced-choice preference, aggregate into a win-rate matrix. Print preference matrix.

---

## Use It

GTM redirect: This is the **Evaluations cluster (Zone 11)** — the same eval-thinking you apply to A/B testing outbound sequences applies here. FID is your open rate (population-level signal). CLIP Score is your reply classification (semantic alignment check). Human preference is your pipeline conversion (ground truth that matters). In GTM, evals = A/B testing your sequences before they go live; reply classification is your eval feedback loop. [CITATION NEEDED — concept: mapping image eval metrics to GTM evaluation frameworks]

**Exercise (Easy):** Given a directory of 10 generated images and their prompts, compute CLIP Scores for all pairs and rank by alignment. Identify the worst-aligned image.

**Exercise (Medium):** Compute FID for two model outputs against the same reference set. Determine which model is closer to the reference distribution. Explain why FID alone is insufficient to pick a winner.

---

## Ship It

**Exercise (Hard):** Build a three-axis evaluation pipeline. Given a reference image directory, a generated image directory, and a prompts file: (1) compute FID, (2) compute CLIP Scores for all prompt-image pairs, (3) output a JSON report with both metrics plus per-image CLIP Scores ranked worst-to-best. Include a `threshold` flag that marks any image with CLIP Score below a configurable threshold as "needs review."

---

## Dig Deeper

- Heusel et al. (2017) — original FID paper: "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
- Radford et al. (2021) — CLIP paper: "Learning Transferable Visual Models From Natural Language Supervision"
- Hessel et al. (2021) — CLIPScore as a reference-free evaluation metric for image captioning
- Kirstain et al. (2021) — "Pick-a-Pic: Open Dataset of Human Preferences for Text-to-Image Generation"
- [CITATION NEEDED — concept: standard human preference evaluation protocols for image generation beyond Pick-a-Pic]
- Practical note: FID is sensitive to sample size (need 50k+ for stable estimates at low dimensions). For small evaluation sets, CLIP Score is more reliable per-image.