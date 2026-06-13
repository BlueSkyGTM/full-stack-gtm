# 3D Gaussian Splatting from Scratch

## Beat 1: Hook — Why Gaussians, Not Neurons

Neural radiance fields made novel view synthesis possible but inference requires evaluating a neural network per pixel per ray. 3D Gaussian Splatting replaces the implicit network with an explicit, differentiable point cloud of anisotropic Gaussians—enabling real-time rasterization without a single forward pass through MLP weights. The tradeoff: storage explodes and the optimization pipeline is non-trivial to implement correctly.

---

## Beat 2: Concept — The Five Parameters of a Gaussian

Each Gaussian in the scene carries five learnable attributes: position **μ** ∈ ℝ³, covariance **Σ** ∈ ℝ³ˣ³ (decomposed as quaternion **q** and scale **s** for positive semi-definiteness), opacity **α** ∈ [0,1], and view-dependent color stored as spherical harmonics coefficients. The rendering equation projects 3D Gaussians to 2D, sorts by depth, and alpha-composites front-to-back—a process called "splatting." The differentiable rasterizer computes gradients for all five attributes via analytical derivatives through the projection and composition steps.

---

## Beat 3: Implement — Splatting Pipeline in Python

Build the core loop: (1) initialize Gaussians from a point cloud, (2) project to 2D via EWA splatting with the Jacobian of the perspective projection, (3) sort tiles and alpha-blend, (4) compute L₁ + SSIM loss against ground-truth images, (5) backpropagate to all five Gaussian parameters, (6) run adaptive density control—split large Gaussians, clone small ones with high positional gradient, prune near-transparent ones. Code renders a synthetic scene from multiple viewpoints and prints convergence metrics.

---

## Beat 4: Use It — GTM Redirect: Foundational for Spatial Content Generation

**GTM Cluster:** Foundational for visual AI content pipelines (Zone 3 — Content Generation). 3D Gaussian Splatting enables product visualization at scale—reconstruct real products from phone captures, embed interactive 3D viewers in e-commerce pages, or generate training data for computer vision models. [CITATION NEEDED — concept: Gaussian Splatting applied to product photography automation]. The mechanism matters because spatial content is expensive to produce manually; an explicit Gaussian representation can be edited, composited, and re-lit without re-scanning.

---

## Beat 5: Ship It — PLY Files, Web Viewers, and Compression

Export the trained Gaussian set as a .ply file (position, scale, rotation, opacity, SH coefficients per point). Serve via web viewers like `gsplat.js` or PlayCanvas that ingest .ply and rasterize in WebGL/WASM. For production: compress SH bands from degree 3 to degree 2, prune Gaussians below opacity threshold, quantize positions to 16-bit relative to bounding box. Print final file size and verify the .ply loads in a reference viewer.

---

## Beat 6: Evaluate — PSNR, SSIM, LPIPS, and Visual Fidelity

Evaluate rendered novel views against held-out camera angles using PSNR (peak signal-to-noise ratio), SSIM (structural similarity), and LPIPS (learned perceptual image similarity). Compare against NeRF baselines on the same scene. Profile rendering speed (ms per frame at 1080p) and storage cost (MB per scene). Exercise: compute all three metrics on a trained splat and determine whether the Gaussian count can be halved before visual quality degrades past a perceptual threshold.

---

## Exercise Hooks

- **Easy:** Render a single 3D Gaussian to a 2D image plane given position, covariance, and intrinsics. Print pixel coordinates of the splat center.
- **Medium:** Implement the EWA splatting projection for a 10-Gaussian scene and verify the rendered image matches a reference by computing PSNR > 30 dB.
- **Hard:** Build the full optimize-render-adapt loop on a synthetic multi-view dataset (4 cameras, 200 Gaussians). Print training loss curve and final PSNR/SSIM/LPIPS.