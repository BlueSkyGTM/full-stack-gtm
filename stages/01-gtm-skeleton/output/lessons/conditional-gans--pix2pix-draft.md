# Conditional GANs & Pix2Pix

## Beat 1: Hook

Vanilla GANs sample from a learned distribution with no control over output. Conditional GANs inject a conditioning signal — a class label, a segmentation map, a sketch — so the generator produces output that corresponds to a specific input. Pix2Pix (Isola et al., 2017) applies this to paired image-to-image translation: given an input image from domain A, produce a corresponding image in domain B.

## Beat 2: Concept

In a conditional GAN (cGAN), both the generator G and discriminator D receive the same conditioning variable y. G takes (z, y) and produces a sample; D takes (x, y) and classifies real vs. fake for that specific condition. Pix2Pix narrows this to paired image translation: y is an input image, x is the target output, and G learns the mapping using adversarial loss plus an L1 reconstruction term. The discriminator uses a PatchGAN architecture — it classifies N×N patches rather than the full image, forcing the generator to produce locally coherent detail.

**Exercise hooks:**
- *Easy:* Trace the data flow through a cGAN — identify where y enters G and D.
- *Medium:* Compare PatchGAN vs. full-image discriminator on a fixed generator. Measure how loss curves differ.

## Beat 3: Mechanism

Three mechanisms distinguish Pix2Pix from a vanilla GAN:

1. **U-Net generator with skip connections.** The encoder contracts spatial resolution; the decoder expands it. Skip connections concatenate encoder activations to decoder layers at matching resolutions. This preserves low-level spatial detail (edges, textures) that would otherwise be lost in the bottleneck.

2. **PatchGAN discriminator.** D operates on N×N patches using convolutional layers, outputting a grid of real/fake classifications rather than a single scalar. Each output neuron corresponds to a receptive field patch. Loss is averaged across all patches. This enforces local high-frequency realism without requiring the discriminator to model global structure.

3. **Combined loss:** L = L_cGAN(G, D) + λ·L_L1(G). The adversarial term pushes toward realistic texture; the L1 term enforces pixel-level correspondence to the ground truth. λ is typically set to 100, weighting reconstruction heavily.

**Exercise hooks:**
- *Medium:* Ablate skip connections from the U-Net. Train on a paired dataset and measure PSNR/SSIM with and without skips.
- *Hard:* Implement PatchGAN at multiple scales (scale the input image to different resolutions before passing to D). Compare single-scale vs. multi-scale discriminator output.

## Beat 4: Code

Implement a minimal Pix2Pix training loop in PyTorch:

- Define a U-Net generator (encoder-decoder with skip connections). Print the shapes at each skip connection to confirm concatenation.
- Define a PatchGAN discriminator. Forward a real pair and a fake pair, print the output spatial dimensions to confirm patch-level classification.
- Run one training step on synthetic paired data (random tensors shaped like images). Print generator loss, discriminator loss, and L1 component separately.

**Exercise hooks:**
- *Medium:* Replace L1 with L2 (MSE) in the reconstruction term. Train for N steps and compare output sharpness visually and via FID.
- *Hard:* Add noise injection to the generator input at test time. Measure output variance across 10 forward passes — quantify how deterministic the mapping is.

## Beat 5: Use It

The core mechanism here — conditioning a generative model on structured input to produce controlled output — maps directly to Zone 2 (Content & Personalization Engine) and Zone 3 (Model Training & Fine-Tuning). In GTM, the analog is a content generation pipeline where firmographic signals (industry, company size, tech stack) condition the output: same model, different conditioning, different result. The paired-data requirement is the constraint — Pix2Pix needs aligned input-output pairs, which is the same constraint you hit when building supervised personalization models from historical outreach data.

[CITATION NEEDED — concept: conditional generation applied to personalized outbound content]

**Exercise hooks:**
- *Medium:* Map a GTM personalization problem to the cGAN framework. Define what y (condition), x (target), and z (noise) would be. Write out the loss function in domain-specific terms.

## Beat 6: Ship It

Pix2Pix inference is deterministic at test time (z is often set to zero or dropped entirely). Latency is dominated by the generator forward pass — measure it. The PatchGAN discriminator is not needed at inference. Deployment considerations: input images must match training resolution and normalization; domain mismatch between training pairs and production inputs causes mode collapse or artifacts; the model cannot generalize to unseen translation directions without retraining (CycleGAN addresses this, covered separately).

**Exercise hooks:**
- *Hard:* Profile generator inference latency across batch sizes 1, 4, 8, 16. Plot throughput vs. batch size. Identify the memory bottleneck.