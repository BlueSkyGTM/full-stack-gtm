# 3D Vision — Point Clouds & NeRFs

## Beat 1: Hook

You have 200 photos of a product from different angles. Two paths: extract discrete 3D points from depth estimation, or train a neural network to synthesize any viewpoint you never captured. These are point clouds and NeRFs — explicit vs. implicit 3D representation. Same input, fundamentally different engineering tradeoffs.

## Beat 2: Concept

**Point clouds**: unordered set of (x, y, z) tuples, optionally with RGB, intensity, or normals. Sparse, explicit, editable. Sourced from LiDAR, structured-light sensors, or stereo depth estimation. Core operations — voxel downsampling, normal estimation, Iterative Closest Point (ICP) for registration.

**Neural Radiance Fields (NeRFs)**: a multilayer perceptron maps continuous 5D coordinates (x, y, z, θ, φ) → (RGB, σ). Density σ and color are queried along camera rays; composite via numerical volume rendering integral to produce a 2D pixel. Trained on multi-view images with known camera poses. The network encodes geometry implicitly — no explicit mesh or point cloud exists unless extracted post-hoc.

**Key distinction**: point clouds store where surfaces are. NeRFs store what you'd see from any angle. One is a data structure; the other is a learned function.

## Beat 3: Demo

Generate a synthetic point cloud of a torus using parametric equations. Compute centroid, bounding box, and surface normals via local plane fitting. Then implement a minimal volume rendering loop — march rays through a synthetic density field and print rendered pixel values. All numpy, no GPU, observable output.

Exercise hooks:
- **Easy**: Modify torus parameters (major radius, minor radius), recompute and print bounding box.
- **Medium**: Downsample the point cloud via voxel grid (bin points into 3D grid cells, keep one per cell) and print before/after point counts.
- **Hard**: Implement simplified ICP alignment between two point clouds with a known small rotation offset. Print convergence error per iteration.

## Beat 4: Use It

Point cloud processing is foundational for Zone 1 (enrichment pipelines that ingest spatial or sensor data) [CITATION NEEDED — concept: GTM application of 3D point cloud data in enrichment]. NeRFs connect to Zone 3 where product experience requires novel view synthesis from limited photography — e.g., generating 360° product rotations for e-commerce listings from a sparse photo set. The Clay waterfall does not apply here; this is a specialized computer vision capability, not a prospecting workflow.

Exercise hooks:
- **Easy**: Export a generated point cloud to PLY format and verify file structure.
- **Medium**: Given a directory of product images with EXIF data, extract focal length estimates (smartphone cameras embed this). Print per-image focal lengths.
- **Hard**: Implement a simplified pose estimation pipeline — estimate relative camera positions between two views using matched feature points (no external SLAM library).

## Beat 5: Ship It

Build a complete mini-pipeline: (1) generate synthetic multi-view renders of a 3D scene, (2) implement a toy NeRF training loop with positional encoding and ray sampling, (3) overfit to the synthetic views and print PSNR per epoch. This runs on CPU with a small scene — the mechanism is the output, not production speed.

Exercise hooks:
- **Easy**: Train the toy NeRF on a 1D density field (single ray, single color). Print loss convergence.
- **Medium**: Extend the toy NeRF to handle a 2D slice (fixed z, rays in x-y plane). Print rendered image as ASCII intensity values.
- **Hard**: Implement positional encoding (γ(p) = [sin(2^k p), cos(2^k p)] for k=0…L) and compare training convergence with vs. without it. Print loss curves.

## Beat 6: Review

Point clouds: explicit, sparse, directly editable, scale with scene complexity, require registration for multi-scan fusion. NeRFs: implicit, continuous, view-dependent effects handled natively, slow to train and render (unless accelerated via Instant-NGP or Gaussian Splatting — mention as follow-up topics). The tradeoff is storage vs. compute: point clouds store geometry upfront; NeRFs compute it on demand.

**Evaluation criteria**:
- Explain when to choose point cloud vs. NeRF for a given 3D task.
- Describe the volume rendering integral and how NeRF uses it as a training signal.
- Implement voxel downsampling and explain what spatial information is lost.
- Compare positional encoding's effect on high-frequency detail recovery.

---

**Learning Objectives (testable, action verbs):**

1. Implement point cloud generation from parametric surfaces and compute geometric properties (centroid, bounding box, normals).
2. Implement voxel downsampling on an unordered point cloud and quantify information loss.
3. Explain the NeRF volume rendering mechanism — ray sampling, density integration, and gradient flow back to the MLP.
4. Implement positional encoding and compare convergence with and without it on a synthetic scene.
5. Evaluate tradeoffs between explicit (point cloud) and implicit (NeRF) 3D representations for a given application scenario.