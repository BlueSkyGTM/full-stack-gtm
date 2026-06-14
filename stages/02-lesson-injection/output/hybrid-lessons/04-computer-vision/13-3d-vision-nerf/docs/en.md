# 3D Vision — Point Clouds & NeRFs

## Learning Objectives

- **Distinguish** explicit 3D representations (point clouds, voxels, meshes) from implicit representations (NeRFs, signed distance fields) by their data structures and tradeoffs.
- **Generate** a synthetic point cloud from parametric equations and compute geometric properties (centroid, bounding box, surface normals) using numpy.
- **Trace** a NeRF forward pass from ray casting through positional encoding, MLP density/color query, to volumetric rendering compositing.

## The Problem

You have 200 photos of a product from different angles. Two paths stretch ahead. On the first, you extract discrete 3D points from depth estimation — a point cloud, sparse and explicit, each coordinate a tangible piece of geometry you can edit, downsample, or feed to a registration algorithm. On the second, you train a neural network to synthesize any viewpoint you never captured — a NeRF, dense and implicit, encoding the entire scene as learned weights inside a multilayer perceptron. Same input photos, fundamentally different engineering tradeoffs.

A camera produces a 2D image. A LiDAR produces a set of 3D points with no inherent ordering. A structure-from-motion pipeline produces a sparse cloud of 3D keypoints matched across frames. A NeRF reconstructs a full volumetric scene from a handful of posed images. All of these fall under "3D vision," yet none of them look like the dense rectangular tensor a CNN expects. The representations diverge because the problems diverge: some tasks need editable geometry (robotic grasping, CAD reconstruction), while others need photorealistic novel views (virtual tours, product visualization, AR/VR content).

This lesson covers both dominant approaches. Point clouds are what sensors hand you for free — raw, explicit, and immediately usable for measurement and registration. NeRFs and their successors (3D Gaussian splatting, neural SDFs) are what you get when you ask a neural network to internalize an entire scene's appearance and geometry from posed images. Knowing both means knowing which tool to reach for when a 3D problem lands on your desk.

## The Concept

### Point Clouds: Explicit Geometry

A point cloud is an unordered set of N points in R³, each optionally carrying features — RGB color, laser reflectance intensity, or estimated surface normals. There is no grid, no connectivity, no notion of "adjacent" points until you impose one via k-nearest-neighbor graphs or voxel grids. This is simultaneously the strength and the headache: the representation is minimal and sensor-native, but standard CNN architectures cannot consume it directly because they assume spatial regularity.

Core operations on point clouds follow from their geometry. **Voxel downsampling** bins points into a 3D grid and keeps one representative per occupied cell, reducing noise and equalizing density. **Normal estimation** fits a local plane to each point's k nearest neighbors using PCA; the smallest eigenvector of the local covariance matrix gives the surface normal. **Iterative Closest Point (ICP)** aligns two overlapping point clouds by iteratively matching nearest points, computing the optimal rotation+translation via SVD, and repeating until convergence. These three operations — downsample, characterize, align — cover the majority of real-world point cloud processing.

**PointNet** solved the architectural problem of feeding unordered sets to neural networks. The key insight: any function applied element-wise to each point and then aggregated via a symmetric operation (max pooling) is invariant to input permutation. If you swap the order of points in the cloud, the max-pooled features are identical. This symmetric-function trick made deep learning on point clouds practical and remains the foundation for most point-cloud architectures that followed.

### Neural Radiance Fields: Implicit Learned Scenes

A NeRF replaces explicit geometry with a learned continuous function. A multilayer perceptron maps a 5D coordinate — spatial position (x, y, z) plus viewing direction (θ, φ) — to a volume density σ and an emitted color (R, G, B). The MLP is small (typically 8 layers, 256 hidden units), but it is queried millions of times during training and rendering. The density field σ encodes geometry implicitly: where density is high, a surface exists; where it is near zero, the ray passes through empty space.

Rendering a single pixel requires **volume rendering along a ray**. You cast a ray from the camera through the pixel into the scene, sample N points along that ray, query the MLP at each point to get (color, density), and composite them using the numerical approximation of the volume rendering equation:

$$C(\mathbf{r}) = \sum_{i=1}^{N} T_i \left(1 - \exp(-\sigma_i \delta_i)\right) \mathbf{c}_i$$

where $T_i = \exp\left(-\sum_{j<i} \sigma_j \delta_j\right)$ is the accumulated transmittance (how much light has already been absorbed before reaching point $i$), $\delta_i$ is the distance between adjacent samples, and $\mathbf{c}_i$ is the color at point $i$. Early samples occlude later ones — this is how the density field creates the appearance of solid surfaces.

```mermaid
flowchart LR
    A["Camera Ray\n(pixel → 3D)"] --> B["Sample N points\nalong ray"]
    B --> C["Positional Encoding\nγ(x) = [x, sin(2⁰x), cos(2⁰x), ...]"]
    C --> D["MLP Query\n(x,y,z,θ,φ) → σ, RGB"]
    D --> E["Volume Rendering\nΣ T_i · α_i · c_i"]
    E --> F["Pixel Color"]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#0f3460,color:#fff
    style E fill:#1a1a2e,stroke:#16c79a,color:#fff
    style F fill:#16213e,stroke:#e94560,color:#fff
```

Training proceeds by rendering pixels from random camera poses, comparing against the ground-truth photos via MSE loss, and backpropagating through the entire volume rendering integral into the MLP weights. **Positional encoding** is critical: without expanding each coordinate into a series of sinusoids at increasing frequencies, the MLP cannot represent high-frequency detail (sharp edges, fine textures). The network sees `γ(x) = [x, sin(2⁰x), cos(2⁰x), sin(2¹x), cos(2¹x), ..., sin(2^(L-1)x), cos(2^(L-1)x)]` rather than raw coordinates. This trick comes from the Fourier features literature and is what separates blurry NeRFs from sharp ones.

### The Key Distinction

Point clouds store **where surfaces are** — a data structure you can index, edit, and measure. NeRFs store **what you would see from any angle** — a learned function you can query but not directly manipulate. One is a database of geometry; the other is a compressed representation of appearance. The choice between them depends on your downstream task: measurement and registration want point clouds, photorealistic rendering wants NeRFs or their successors.

## Build It

### Part 1: Synthetic Point Cloud and Geometric Operations

Generate a torus point cloud from parametric equations, then compute centroid, bounding box, and surface normals via local plane fitting. Every operation uses pure numpy and prints observable output.

```python
import numpy as np

np.random.seed(42)

R_MAJOR = 2.0
R_MINOR = 0.7
N_POINTS = 5000

u = np.random.uniform(0, 2 * np.pi, N_POINTS)
v = np.random.uniform(0, 2 * np.pi, N_POINTS)

x = (R_MAJOR + R_MINOR * np.cos(v)) * np.cos(u)
y = (R_MAJOR + R_MINOR * np.cos(v)) * np.sin(u)
z = R_MINOR * np.sin(v)
cloud = np.stack([x, y, z], axis=1)

centroid = cloud.mean(axis=0)
bbox_min = cloud.min(axis=0)
bbox_max = cloud.max(axis=0)

print(f"Points generated: {len(cloud)}")
print(f"Centroid: ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")
print(f"Bounding box min: ({bbox_min[0]:.4f}, {bbox_min[1]:.4f}, {bbox_min[2]:.4f})")
print(f"Bounding box max: ({bbox_max[0]:.4f}, {bbox_max[1]:.4f}, {bbox_max[2]:.4f})")
print(f"Bounding box size: ({bbox_max[0]-bbox_min[0]:.4f}, {bbox_max[1]-bbox_min[1]:.4f}, {bbox_max[2]-bbox_min[2]:.4f})")
```

Run this and you will see the centroid sitting near the origin, the bounding box spanning roughly ±2.7 in x and y and ±0.7 in z — consistent with a torus of major radius 2 and minor radius 0.7.

Now compute surface normals using PCA on each point's k nearest neighbors. The normal is the eigenvector corresponding to the smallest eigenvalue of the local covariance matrix.

```python
def compute_normals(cloud, k=10):
    diff = cloud[:, np.newaxis, :] - cloud[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    nn_indices = np.argsort(dists, axis=1)[:, 1:k+1]

    normals = np.zeros_like(cloud)
    for i in range(len(cloud)):
        neighbors = cloud[nn_indices[i]]
        local_centered = neighbors - neighbors.mean(axis=0)
        cov = local_centered.T @ local_centered / k
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normals[i] = eigenvectors[:, 0]

    return normals

normals = compute_normals(cloud, k=10)

print(f"\nNormal vectors computed: {len(normals)}")
print(f"First 5 normals:")
for i in range(5):
    n = normals[i]
    p = cloud[i]
    expected_normal = cloud[i].copy()
    expected_normal[2] = 0
    norm_len = np.linalg.norm(expected_normal)
    if norm_len > 0:
        expected_normal = expected_normal / norm_len
    expected_normal[2] = cloud[i][2] / R_MINOR
    print(f"  Point ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) -> Normal ({n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f})")

dot_products = np.abs(np.sum(normals * cloud, axis=1))
radial = np.sqrt(cloud[:, 0]**2 + cloud[:, 1]**2)
expected_radial = np.array([R_MAJOR + R_MINOR * np.cos(v[i]) for i in range(N_POINTS)])
normal_radial_dot = np.abs(normals[:, 0] * cloud[:, 0] + normals[:, 1] * cloud[:, 1])
cos_sim = normal_radial_dot / (np.sqrt(normals[:, 0]**2 + normals[:, 1]**2) * np.sqrt(cloud[:, 0]**2 + cloud[:, 1]**2) + 1e-8)
print(f"\nMean alignment between computed normals and radial direction: {cos_sim.mean():.4f}")
print(f"(Should be high — torus normals point radially outward from the tube center)")
```

The alignment metric should be high because a torus's surface normals point radially outward from the tube's circular cross-section. If you see values above 0.7, the PCA-based normal estimation is working correctly.

### Part 2: Minimal Volume Rendering Loop

Implement the core NeRF rendering operation — march rays through a synthetic density field and composite pixel colors using the volume rendering integral. No GPU, no deep learning framework, just the math that makes NeRFs work.

```python
import numpy as np

def synthetic_density_field(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    sphere_dist = np.sqrt((x - 0.5)**2 + y**2 + z**2)
    density = np.where(sphere_dist < 0.8, 3.0 * (1 - sphere_dist / 0.8), 0.0)
    r = np.where(density > 0, 0.8, 0.1)
    g = np.where(density > 0, 0.3, 0.1)
    b = np.where(density > 0, 0.2, 0.1)
    colors = np.stack([r, g, b], axis=1)
    return density, colors

def render_ray(ray_origin, ray_dir, near, far, n_samples):
    t_vals = np.linspace(near, far, n_samples)
    points = ray_origin[np.newaxis, :] + t_vals[:, np.newaxis] * ray_dir[np.newaxis, :]
    deltas = np.diff(t_vals, append=t_vals[-1] + (t_vals[-1] - t_vals[-2]))

    sigma, colors = synthetic_density_field(points)

    alpha = 1.0 - np.exp(-sigma * deltas)
    transmittance = np.cumprod(1.0 - alpha + 1e-10)
    transmittance = np.concatenate([[1.0], transmittance[:-1]])

    weights = transmittance * alpha
    pixel_color = np.sum(weights[:, np.newaxis] * colors, axis=0)
    depth = np.sum(weights * t_vals)
    hit_prob = np.sum(weights)

    return pixel_color, depth, hit_prob

ray_origin = np.array([0.0, 0.0, -3.0])
ray_dir = np.array([0.0, 0.0, 1.0])
ray_dir = ray_dir / np.linalg.norm(ray_dir)

color, depth, hit = render_ray(ray_origin, ray_dir, near=0.0, far=6.0, n_samples=64)
print("Ray through center of sphere:")
print(f"  Pixel color (RGB): ({color[0]:.4f}, {color[1]:.4f}, {color[2]:.4f})")
print(f"  Expected depth:    ~3.5 (sphere center at z=0.5, ray starts at z=-3)")
print(f"  Computed depth:    {depth:.4f}")
print(f"  Hit probability:   {hit:.4f}")

miss_dir = np.array([3.0, 0.0, 1.0])
miss_dir = miss_dir / np.linalg.norm(miss_dir)
color_miss, depth_miss, hit_miss = render_ray(ray_origin, miss_dir, near=0.0, far=6.0, n_samples=64)
print("\nRay missing the sphere (offset x=3):")
print(f"  Pixel color (RGB): ({color_miss[0]:.4f}, {color_miss[1]:.4f}, {color_miss[2]:.4f})")
print(f"  Hit probability:   {hit_miss:.4f}  (should be near 0)")

img_size = 16
image = np.zeros((img_size, img_size, 3))
for i in range(img_size):
    for j in range(img_size):
        px = (j / img_size - 0.5) * 4.0
        py = (i / img_size - 0.5) * 4.0
        d = np.array([px, py, 1.0])
        d = d / np.linalg.norm(d)
        c, _, _ = render_ray(ray_origin, d, near=0.0, far=6.0, n_samples=32)
        image[i, j] = c

brightness = image.mean(axis=2)
center_brightness = brightness[img_size//2, img_size//2]
corner_brightness = brightness[0, 0]
print(f"\nRendered {img_size}x{img_size} image")
print(f"Center pixel brightness: {center_brightness:.4f} (should be high — sphere is there)")
print(f"Corner pixel brightness: {corner_brightness:.4f} (should be low — empty space)")

rows_with_content = np.where(brightness.max(axis=1) > 0.3)[0]
print(f"Rows with visible sphere: {rows_with_content[0]} to {rows_with_content[-1]} (out of {img_size})")
```

The center ray should hit the synthetic sphere at roughly z ≈ 3.5 (origin at z=−3, sphere center at z=0.5). The miss ray should produce a near-zero hit probability. The rendered image grid should show a bright cluster of pixels in the center and darkness at the corners — the same pattern a trained NeRF produces, just with a hand-coded density field instead of an MLP.

## Use It

Point cloud geometric extraction — the mechanism that turns raw 3D sensor coordinates into structured measurements via bounding-box computation, volume estimation, and surface normal analysis — connects to GTM when physical products need measured attributes for catalog enrichment, shipping classification, or warehouse slotting [CITATION NEEDED — concept: point-cloud-based product measurement in e-commerce logistics]. The same Find → Enrich → Transform → Export waterfall pattern from Zone 04 applies: you find points from a depth sensor, enrich with geometric computations, transform into structured catalog fields, and export to a PIM or ERP. The data type is spatial; the pipeline logic is identical.

```python
import numpy as np

np.random.seed(7)

faces = []
for axis in range(3):
    for sign in [-1, 1]:
        pts = np.random.uniform(-0.5, 0.5, (250, 3))
        pts[:, axis] = sign * 0.5
        faces.append(pts)
cloud = np.vstack(faces) * np.array([0.15, 0.10, 0.075])

dims = cloud.max(axis=0) - cloud.min(axis=0)
volume = np.prod(dims)
diag = np.linalg.norm(dims)
category = "small parcel" if volume < 0.01 else "standard package"
slot = "A" if volume < 0.005 else ("B" if volume < 0.02 else "C")

print(f"Product scan: {len(cloud)} surface points")
print(f"Dimensions (W x D x H): {dims[0]:.3f}m x {dims[1]:.3f}m x {dims[2]:.3f}m")
print(f"Volume: {volume:.5f} m3  |  Diagonal: {diag:.3f}m")
print(f"Shipping class: {category}")
print(f"Warehouse slot: {slot}")
```

This slice demonstrates the enrichment step: raw 3D points enter, structured catalog attributes exit. In a production pipeline, the input cloud would come from a depth camera or LiDAR scan rather than synthetic parametric generation. The bounding-box extraction, volume computation, and classification logic remain identical. NeRFs occupy a different GTM lane — novel view synthesis for product visualization pages, virtual tours, or AR experiences — but they do not slot into a prospecting waterfall. They are a content-generation mechanism, not a data-enrichment one. If your GTM motion involves physical product data or spatial content, these are the representations you work with; otherwise, this lesson is foundational knowledge that sharpens your judgment about what computer vision can and cannot do.

## Exercises

**Easy — Torus Parameter Sweep.** Modify the torus generation code to accept `R_MAJOR` and `R_MINOR` as function parameters. Generate three toruses: (R=1.0, r=0.3), (R=3.0, r=0.5), (R=2.0, r=1.5). For each, print the bounding box dimensions and verify they match the analytical expectation: x and y span should be `2*(R+r)`, z span should be `2*r`. If any torus shows a mismatch, explain why.

**Hard — Two-Sphere Volume Rendering.** Extend the `synthetic_density_field` function to place two spheres: one centered at (−0.8, 0, 0) with radius 0.5 and red color (0.9, 0.1, 0.1), another at (0.8, 0, 0) with radius 0.5 and blue color (0.1, 0.1, 0.9). Cast a ray along the x-axis from origin (−3, 0, 0) through both spheres. Print the rendered color and explain which sphere's color dominates the final pixel and why. Then cast a ray from (−3, 0.5, 0) that grazes the edge of the red sphere and misses the blue one — verify the hit probability reflects partial intersection.

## Key Terms

- **Point Cloud** — An unordered set of N points in R³, each optionally carrying features like color or normals. Sensor-native, explicitly editable, but lacks spatial regularity for standard CNNs.
- **NeRF (Neural Radiance Field)** — A learned continuous function (MLP) mapping 5D coordinates (position + viewing direction) to volume density and emitted color. Encodes an entire scene implicitly in network weights.
- **Volume Rendering** — The compositing integral that accumulates color and density along a camera ray through sampled points, weighted by accumulated transmittance. The core operation that turns a NeRF's density field into a 2D image.
- **Positional Encoding** — Expansion of raw coordinates into sinusoids at geometrically increasing frequencies (γ(x) = [x, sin(2⁰x), cos(2⁰x), ...]). Enables MLPs to represent high-frequency detail; without it, NeRFs produce blurry reconstructions.
- **Transmittance** — The fraction of light that reaches sample *i* without being absorbed by earlier samples along the ray. Computed as a cumulative product of (1 − α) terms; governs occlusion in volume rendering.
- **PointNet** — Architecture that processes unordered point sets via element-wise MLPs followed by max pooling, achieving permutation invariance. Foundation for deep learning on point cloud data.
- **Voxel Downsampling** — Binning points into a 3D grid and retaining one representative per occupied cell. Reduces noise, equalizes density, and bounds memory for downstream processing.

## Sources

- Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). *NeRF: Representing Scenes as Neural Radiadiance Fields for View Synthesis.* ECCV 2020. — Original NeRF paper; volume rendering equation, positional encoding, MLP architecture.
- Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.* CVPR 2017. — Symmetric function approach to permutation-invariant point cloud processing.
- Tancik, M., Srinivasan, P., Mildenhall, B., et al. (2020). *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains.* NeurIPS 2020. — Theoretical basis for positional encoding.
- [CITATION NEEDED — concept: point-cloud-based product measurement in e-commerce logistics]
- [CITATION NEEDED — concept: NeRF-based 3D product visualization in GTM / e-commerce product experience]