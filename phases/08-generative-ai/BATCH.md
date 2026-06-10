# Phase 08 — Generative AI (quiz factory)

## Focus

Generative modelling paradigms: diffusion (DDPM, DDIM, LDM, Stable Diffusion), flow-matching, score-based models, VQVAE/VQGAN discrete generation, autoregressive image generation, and the scaling laws that connect them.

## Scrape hints

- `docs/en.md`: focus on the generative process formulas (forward/reverse diffusion, score function, flow ODE), training objectives (noise prediction vs x0 prediction vs v-prediction), and architectural choices (U-Net vs DiT)
- `code/main.py`: signal-processing steps, sampling loops, latent-space dimensions
- Vocabulary: `glossary/terms.md` for denoising, score function, latent diffusion, VQ codebook, guidance

## Style anchor

- `phases/08-generative-ai/19-visual-autoregressive-var/quiz.json` — rebuilt gold quiz for this phase
- Backup: `phases/07-transformers-deep-dive/15-attention-variants/quiz.json` (same depth tier)

## Common distractor patterns

- Confuse score matching with score function vs log-likelihood gradient
- Confuse DDPM (discrete steps) with continuous-time score SDEs
- Confuse latent diffusion (LDM, Stable Diffusion) with pixel-space diffusion
- Conflate conditional generation mechanisms (classifier guidance vs classifier-free guidance)
- Mix up VQ-VAE, VQ-GAN, and DALL-E 1 tokenisation approaches

## Do not

- Import facts from other phases unless `docs/en.md` lists them as prerequisites.
- Ask the user questions — mark `blocked` in manifest instead.
- Reuse question wording from the lesson-planning SKILL.md flagship examples — that text is for orientation, not copying.
