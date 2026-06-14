# 3D Vision — Point Clouds & NeRFs

## Learning Objectives

- **Distinguish** explicit 3D representations (point clouds, voxels, meshes) from implicit representations (NeRFs, signed distance fields) by their data structures and tradeoffs.
- **Generate** a synthetic point cloud from parametric equations and compute geometric properties (centroid, bounding box, surface normals) using numpy.
- **Trace** a NeRF forward pass from ray casting through positional encoding, MLP density/color query, to volumetric rendering compositing.
- **Implement** a minimal volume rendering loop that marches rays through a synthetic density field and produces rendered pixel values.
- **Evaluate** when point cloud processing versus neural radiance field reconstruction is the appropriate tool for a given 3D vision task.

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

Point cloud processing connects to the enrichment waterfalls in Zone 04 — the Find → Enrich → Transform → Export pipeline — when the data being enriched has spatial or geometric structure. Consider a company building a product catalog for e-commerce: product dimensions extracted from 3D scans, spatial relationships between items in a warehouse layout, or depth data from AR-equipped smartphones. These are point clouds that need voxel downsampling, normal estimation, or registration before they can feed downstream systems. The waterfall pattern — try source A, fall back to source B, then source C — applies when you chain geometric processing steps: estimate from sensor data, refine with ICP alignment against a reference mesh, then export the registered result. The mechanism is the same fallback logic as the Clay waterfall; the data type is different.

NeRFs connect to a different GTM motion: novel view synthesis for product experience. A B2C company with offline stores — particularly food franchising chains where the pain point is new store ramp-up speed — benefits from 360° product visualization generated from sparse photo sets rather than expensive professional shoots [CITATION NEEDED — concept: NeRF-based 3D product visualization in e-commerce GTM]. The same technique applies to real estate listings, construction site documentation, or insurance claims: capture 20-40 posed photos, train a NeRF or Gaussian splat, and produce a navigable 3D scene that a prospect or customer can explore from any angle.

The Clay waterfall does not directly apply to NeRF training — there is no prospecting workflow where you fall back from NeRF to a data enrichment provider. This is specialized computer vision capability, not a Zone 04 data pipeline. The honest framing: 3D vision is foundational for any GTM motion that involves spatial data or visual product experience, but it is not itself a prospecting tool. If your enrichment pipeline ingests sensor data or your product experience requires 3D reconstruction, these are the representations you work with. If it does not, this lesson is background knowledge that sharpens your judgment about what computer vision can and cannot do.

```python
import numpy as np

def export_ply(cloud, colors=None, normals=None, filename="output.ply"):
    n = len(cloud)
    header = ["ply", "format ascii 1.0",
              f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    if normals is not None:
        header += ["property float nx", "property float ny", "property float nz"]
    header.append("end_header")

    lines = []
    for i in range(n):
        parts = [f"{cloud[i,0]:.6f}", f"{cloud[i,1]:.6f}", f"{cloud[i,2]:.6f}"]
        if colors is not None:
            parts += [str(int(np.clip(colors[i,0], 0, 255))),
                      str(int(np.clip(colors[i,1], 0, 255))),
                      str(int(np.clip(colors[i,2], 0, 255)))]
        if normals is not None:
            parts += [f"{normals[i,0]:.6f}", f"{normals[i,1]:.6f}", f"{normals[i,2]:.6f}"]
        lines.append(" ".join(parts))

    with open(filename, "w") as f:
        f.write("\n".join(header + lines) + "\n")

    print(f"Exported {n} points to {filename}")
    return filename

R_MAJOR = 2.0
R_MINOR = 0.7
N = 2000
u = np.random.uniform(0, 2*np.pi, N)
v = np.random.uniform(0, 2*np.pi, N)
x = (R_MAJOR + R_MINOR * np.cos(v)) * np.cos(u)
y = (R_MAJOR + R_MINOR * np.cos(v)) * np.sin(u)
z = R_MINOR * np.sin(v)
torus_cloud = np.stack([x, y, z], axis=1)

intensity = np.clip((z + R_MINOR) / (2 * R_MINOR) * 255, 0, 255)
torus_colors = np.stack([intensity, 200 - intensity * 0.3, np.full(N, 100.0)], axis=1)

export_ply(torus_cloud, colors=torus_colors, filename="torus.ply")

with open("torus.ply", "r") as f:
    for i, line in enumerate(f.readlines()):
        print(line.rstrip())
        if i > 10:
            print("... (truncated)")
            break
```

This PLY exporter produces a file that opens in MeshLab, Blender, or Open3D — the standard interchange format for point cloud data in any pipeline that feeds downstream GTM tooling.

## Ship It

Build a mini NeRF-style reconstruction pipeline that takes synthetic posed images of a 3D scene, estimates camera rays for each pixel, runs the volume rendering loop, and reports reconstruction quality metrics. This is the same architectural skeleton behind nerfstudio and instant-ngp — minus the GPU acceleration and the MLP optimization loop.

The pipeline mirrors the Clay waterfall's Find → Enrich → Transform → Export pattern, but for spatial data: Find (posed images + camera parameters), Enrich (ray sampling + density queries), Transform (volume rendering composite), Export (rendered novel views). The waterfall concept — progressive refinement through chained operations — is the structural parallel. The Clay waterfall chains data providers to enrich a prospect record. This pipeline chains geometric operations to enrich a set of 2D images into a 3D scene representation.

```python
import numpy as np

def create_posed_images(scene_func, camera_poses, img_size=16, fov=60):
    images = []
    focal = (img_size / 2) / np.tan(np.radians(fov) / 2)

    for pose in camera_poses:
        origin, look_at, up = pose
        forward = look_at - origin
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        cam_up = np.cross(right, forward)

        img = np.zeros((img_size, img_size, 3))
        for i in range(img_size):
            for j in range(img_size)):
                ndc_x = (j - img_size / 2) / focal
                ndc_y = -(i - img_size / 2) / focal
                ray_dir = ndc_x * right + ndc_y * cam_up + forward
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                img[i, j] = render_pixel(scene_func, origin, ray_dir)
        images.append(img)
    return np.array(images)

def render_pixel(scene_func, origin, direction, near=0.1, far=10.0, n_samples=32):
    t_vals = np.linspace(near, far, n_samples)
    points = origin[np.newaxis, :] + t_vals[:, np.newaxis] * direction[np.newaxis, :]
    sigma, colors = scene_func(points)
    deltas = np.diff(t_vals, append=t_vals[-1] + (t_vals[-1] - t_vals[-2]))
    alpha = 1.0 - np.exp(-sigma * deltas)
    T = np.cumprod(1.0 - alpha + 1e-10)
    T = np.concatenate([[1.0], T[:-1]])
    weights = T * alpha
    return np.sum(weights[:, np.newaxis] * colors, axis=0)

def cube_scene(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    inside = (np.abs(x) < 0.6) & (np.abs(y) < 0.6) & (np.abs(z) < 0.6)
    sigma = np.where(inside, 8.0, 0.0)
    face_colors = np.stack([
        np.where(x > 0.3, 0.9, 0.3),
        np.where(y > 0.3, 0.2, 0.7),
        np.where(z > 0.3, 0.1, 0.8)
    ], axis=1) * inside[:, np.newaxis]
    return sigma, face_colors

camera_poses = [
    (np.array([3.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    (np.array([0.0, 3.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    (np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    (np.array([-3.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
]

images = create_posed_images(cube_scene, camera_poses, img_size=12)
print(f"Generated {len(images)} posed images, shape: {images.shape}")

for idx, img in enumerate(images):
    bright = img.mean(axis=2)
    nonzero = np.count_nonzero(bright > 0.1)
    avg_color = img[bright > 0.1].mean(axis=0) if nonzero > 0 else np.array([0,0,0])
    print(f"  View {idx}: {nonzero} bright pixels, avg color ({avg_color[0]:.3f}, {avg_color[1]:.3f}, {avg_color[2]:.3f})")

novel_origin = np.array([2.1, 2.1, 0.0])
novel_pose = [(novel_origin, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))]
novel_view = create_posed_images(cube_scene, novel_pose, img_size=12)[0]
novel_bright = novel_view.mean(axis=2)
nonzero = np.count_nonzero(novel_bright > 0.1)
print(f"\nNovel view (interpolated between views 0 and 1):")
print(f"  {nonzero} bright pixels — scene visible from untrained angle")

known_views_brightness = [images[i].mean(axis=2).mean() for i in range(len(images))]
novel_brightness = novel_bright.mean()
print(f"  Known view avg brightness: {np.mean(known_views_brightness):.4f}")
print(f"  Novel view avg brightness: {novel_brightness:.4f}")
print(f"  Ratio: {novel_brightness / (np.mean(known_views_brightness) + 1e-8):.4f}")
```

Wait — there is a syntax error in the loop body above (`for j in range(img_size))` has an extra paren). Let me fix that and present the corrected, runnable version:

```python
import numpy as np

def create_posed_images(scene_func, camera_poses, img_size=12, fov=60):
    images = []
    focal = (img_size / 2) / np.tan(np.radians(fov) / 2)

    for pose in camera_poses:
        origin, look_at, up = pose
        forward = look_at - origin
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        cam_up = np.cross(right, forward)

        img = np.zeros((img_size, img_size, 3))
        for i in range(img_size):
            for j in range(img_size):
                ndc_x = (j - img_size / 2) / focal
                ndc_y = -(i - img_size / 2) / focal
                ray_dir = ndc_x * right + ndc_y * cam_up + forward
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                img[i, j] = render_pixel(scene_func, origin, ray_dir)
        images.append(img)
    return np.array(images)

def render_pixel(scene_func, origin, direction, near=0.1, far=10.0, n_samples=32):
    t_vals = np.linspace(near, far, n_samples)
    points = origin[np.newaxis, :] + t_vals[:, np.newaxis] * direction[np.newaxis, :]
    sigma, colors = scene_func(points)
    deltas = np.diff(t_vals, append=t_vals[-1] + (t_vals[-1] - t_vals[-2]))
    alpha = 1.0 - np.exp(-sigma * deltas)
    T = np.cumprod(1.0 - alpha + 1e-10)
    T = np.concatenate([[1.0], T[:-1]])
    weights = T * alpha
    return np.sum(weights[:, np.newaxis] * colors, axis=0)

def cube_scene(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    inside = (np.abs(x) < 0.6) & (np.abs(y) < 0.6) & (np.abs(z) < 0.6)
    sigma = np.where(inside, 8.0, 0.0)
    r = np.where(x > 0.3, 0.9, 0.3)
    g = np.where(y > 0.3, 0.2, 0.7)
    b = np.where(z > 0.3, 0.1, 0.8)
    colors = np.stack([r, g, b], axis=1) * inside[:, np.newaxis]
    return sigma, colors

camera_poses = [
    (np.array([3.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    (np.array([0.0, 3.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    (np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    (np.array([-3.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
]

images = create_posed_images(cube_scene, camera_poses, img_size=12)
print(f"Generated {len(images)} posed images, shape: {images.shape}")

for idx, img in enumerate(images):
    bright = img.mean(axis=2)
    nonzero = np.count_nonzero(bright > 0.1)
    mask = bright > 0.1
    avg_color = img[mask].mean(axis=0) if nonzero > 0 else np.array([0.0, 0.0, 0.0])
    print(f"  View {idx}: {nonzero} bright pixels, avg color ({avg_color[0]:.3f}, {avg_color[1]:.3f}, {avg_color[2]:.3f})")

novel_origin = np.array([2.1, 2.1, 0.0])
novel_pose = [(novel_origin, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))]
novel_view = create_posed_images(cube_scene, novel_pose, img_size=12)[0]
novel_bright = novel_view.mean(axis=2)
nonzero_novel = np.count_nonzero(novel_bright > 0.1)
print(f"\nNovel view (interpolated angle between views 0 and 1):")
print(f"  {nonzero_novel} bright pixels — scene visible from untrained angle")

known_brightness = np.mean([images[i].mean(axis=2).mean() for i in range(len(images))])
novel_brightness = novel_bright.mean()
print(f"  Known views avg brightness: {known_brightness:.4f}")
print(f"  Novel view avg brightness:  {novel_brightness:.4f}")
print(f"  Ratio (novel/known):        {novel_brightness / (known_brightness + 1e-8):.4f}")
```

This pipeline renders a cube from four cardinal viewpoints plus one novel interpolated viewpoint. The four training views correspond to the posed images you would feed to a NeRF. The novel view demonstrates the key NeRF capability: synthesizing an angle you never explicitly captured. In a real NeRF, an MLP would replace the hand-coded `cube_scene` function — the density and color fields would be learned from the training images rather than specified analytically. The volume rendering loop, camera ray computation, and compositing math are identical.

## Exercises

**Easy — Torus Parameter Sweep.** Modify the torus generation code to accept major radius and minor radius as parameters. Generate toruses with (R=1.0, r=0.