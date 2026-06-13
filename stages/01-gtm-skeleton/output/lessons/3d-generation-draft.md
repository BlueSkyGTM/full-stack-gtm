# 3D Generation

## Hook

3D asset creation traditionally requires specialized modeling skills and hours of manual work. Generative 3D pipelines compress this to a text prompt or single image input. This lesson covers the mechanisms that make that possible and where they break down.

## Concept

Three representations dominate: point clouds (unordered XYZ + color), neural radiance fields (continuous volume density + color queried by position and viewing angle), and polygon meshes (vertices + faces). Current generation pipelines use Score Distillation Sampling (SDS) to pull a pre-trained 2D diffusion model's knowledge into a 3D representation by iteratively rendering views, adding noise, denoising, and backpropagating the gradient into the 3D parameters. Direct regression models (Shap-E, Point-E) skip SDS and train end-to-end on paired text-3D datasets, trading quality for speed. Gaussian Splatting replaces NeRF's implicit volume with explicit 3D Gaussians, enabling real-time rendering. Each representation imposes different constraints on output fidelity, editability, and inference cost.

## Use It

GTM redirect: Zone 3 (Content Engine) — specifically product-led content clusters where visual assets differentiate. 3D object generation feeds product demo pages, interactive configurators, and social content at scale. The mechanism is asset multiplication: one prompt yields a mesh that can be rendered from any angle, lit any way, and placed in any scene — eliminating per-angle photoshoot overhead. [CITATION NEEDED — concept: 3D asset generation pipeline for GTM content velocity]

**Exercise hooks:**
- *Easy:* Generate a point cloud from text using Point-E and render three rotated views to PNG.
- *Medium:* Compare SDS-based output (via Shap-E) against direct point cloud output on the same prompt. Log inference time and vertex count for each.
- *Hard:* Build a script that takes a product name, generates a 3D asset, renders 12 orbital views, and writes a JSON manifest mapping each view angle to a file path — ready for ingestion by a static site.

## Ship It

A CLI tool that accepts a text prompt or input image, generates a 3D mesh (OBJ or GLB), writes it to disk, renders a configurable number of orbital preview images, and outputs a metadata JSON with vertex/face counts, generation parameters, and file paths. Uses Shap-E for the generation step and trimesh for rendering. All outputs go to a timestamped directory.

## Debug It

SDS-based generation produces the "Janus problem" — the model generates multiple faces on one object because 2D diffusion has no 3D consistency constraint. Direct regression models produce blobby or low-resolution geometry, especially on out-of-distribution prompts. Texture bleeding, non-manifold meshes, and degenerate faces (zero-area triangles) are common in exported outputs. This section covers detection scripts for each failure mode and mitigation strategies: prompt engineering constraints, post-processing mesh repair via pymeshlab, and multi-view consistency checks.

## Extend It

NeRF-to-mesh extraction (Marching Cubes on density fields), Gaussian Splatting for real-time rendering, video-to-3D reconstruction for turning existing product footage into navigable 3D assets, and fine-tuning Shap-E on domain-specific object categories. Pointer to ComfyUI 3D nodes for pipeline orchestration and to TripoSR for open-source image-to-3D with higher fidelity than Point-E.