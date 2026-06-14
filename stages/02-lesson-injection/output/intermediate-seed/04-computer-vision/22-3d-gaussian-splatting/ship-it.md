## Ship It

Shipping a 3DGS scene to production requires three things: a `.ply` export with the correct attribute layout, a web viewer that can ingest it, and compression to get the file size under budget. The `.ply` format stores each Gaussian as a vertex with attributes: `x, y, z` (position), `scale_0, scale_1, scale_2` (scale), `rot_0, rot_1, rot_2, rot_3` (quaternion), `opacity` (scalar), and `f_dc_0..2, f_rest_0..44` (spherical harmonics). A production viewer like `gsplat.js` or PlayCanvas reads these attributes and uploads them to the GPU as vertex buffers.

Compression matters because a raw 1M-Gaussian scene is ~236 MB — too large for a web page. Three techniques reduce this by 5-10×. First, reduce SH bands from degree 3 (48 floats for color) to degree 1 (9 floats) or degree 0 (3 floats) — most scenes look acceptable at degree 1, and the difference is only visible at grazing angles with strong specular highlights. Second, prune Gaussians with opacity below a threshold (0.01-0.05) — they contribute almost nothing to the final image but consume full storage. Third, quantize positions from fp32 to fp16 relative to the scene's bounding box, cutting 12 bytes per Gaussian to 6. These are lossy but the visual quality drop is measurable with PSNR/SSIM before shipping.

```python
import numpy as np
import struct

np.random.seed(42)

def generate_scene(n_gaussians=50000):
    positions = np.random.randn(n_gaussians, 3).astype(np.float32)
    scales = np.random.exponential(0.01, (n_gaussians, 3)).astype(np.float32)
    rotations = np.random.randn(n_gaussians, 4).astype(np.float32)
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
    opacities = (np.random.rand(n_gaussians) * 0.5 + 0.3).astype(np.float32)
    sh_dc = np.random.randn(n_gaussians, 3).astype(np.float32) * 0.1
    sh_rest = np.random.randn(n_gaussians, 45).astype(np.float32) * 0.01
    return {
        "positions": positions,
        "scales": scales,
        "rotations": rotations,
        "opacities": opacities,
        "sh_dc": sh_dc,
        "sh_rest": sh_rest,
    }

def write_ply(filename, scene, write_sh_rest=True):
    n = len(scene["positions"])
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {n}\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    header += "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
    header += "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
    header += "property float opacity\n"
    header += "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
    if write_sh_rest:
        for i in range(45):
            header += f"property float f_rest_{i}\n"
    header += "end_header\n"

    with open(filename, 'wb') as f:
        f.write(header.encode())
        for i in range(n):
            row = list(scene["positions"][i])
            row += list(scene["scales"][i])
            row += list(scene["rotations"][i])
            row += [scene["opacities"][i]]
            row += list(scene["sh_dc"][i])
            if write_sh_rest:
                row += list(scene["sh_rest"][i])
            f.write(struct.pack(f'{len(row)}f', *row))

    import os
    return os.path.getsize(filename)

def compress_scene(scene, opacity_thresh=0.01, sh_to_degree=1):
    mask = scene["opacities"] >= opacity_thresh
    compressed = {
        "positions": scene["positions"][mask].copy(),
        "scales": scene["scales"][mask].copy(),
        "rotations": scene["rotations"][mask].copy(),
        "opacities": scene["opacities"][mask].copy(),
        "sh_dc": scene["sh_dc"][mask].copy(),
    }
    if sh_to_degree == 0:
        compressed["sh_rest"] = np.zeros((mask.sum(), 0), dtype=np.float32)
    elif sh_to_degree == 1:
        compressed["sh_rest"] = scene["sh_rest"][mask][:, :6].copy()
    else:
        compressed["sh_rest"] = scene["sh_rest"][mask].copy()

    if sh_to_degree < 3:
        n_pruned_sh = mask.sum() * (45 - compressed["sh_rest"].shape[1])
        print(f"  SH pruned: {n_pruned_sh:,} floats (degree 3 -> degree {sh_to_degree})")

    print(f"  Opacity pruned: {(~mask).sum():,} Gaussians removed (< {opacity_thresh})")
    return compressed

def quantize_positions(scene, bits=16):
    positions = scene["positions"]
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    normalized = (positions - bbox_min) / (bbox_max - bbox_min + 1e-8)
    max_val = (2 ** bits) - 1
    quantized = np.round(normalized * max_val).astype(np.float32)
    dequantized = quantized / max_val * (bbox_max - bbox_min) + bbox_min
    max_error = np.abs(positions - dequantized).max()
    print(f"  Position quantization ({bits}-bit): max error = {max_error:.6f}")
    return scene

scene = generate_scene(50000)

print("=== UNCOMPRESSED ===")
size_full = write_ply("/tmp/scene_full.ply", scene, write_sh_rest=True)
print(f"Degree 3 SH: {size_full / 1024 / 1024:.2f} MB")

size_dc_only = write_ply("/tmp/scene_dc.ply", scene, write_sh_rest=False)
print(f"Degree 0 SH: {size_dc_only / 1024 / 1024:.2f} MB")

print("\n=== COMPRESSED ===")
print("Step 1: Prune opacity < 0.05, reduce SH to degree 1")
compressed = compress_scene(scene, opacity_thresh=0.05, sh_to_degree=1)
size_compressed = write_ply("/tmp/scene_compressed.ply", compressed, write_sh_rest=True)
print(f"Compressed: {size_compressed / 1024 / 1024:.2f} MB")

print("\nStep 2: Quantize positions to 16-bit")
quantize_positions(compressed, bits=16)

print(f"\n=== COMPRESSION SUMMARY ===")
print(f"Original:   {size_full / 1024 / 1024:>8.2f} MB ({len(scene['positions']):,} Gaussians, SH deg 3)")
print(f"Compressed: {size_compressed / 1024 / 1024:>8.2f} MB ({len(compressed['positions']):,} Gaussians, SH deg 1)")
print(f"Ratio:      {size_full / size_compressed:>8.2f}x reduction")

print(f"\n.ply files written to /tmp/")
print(f"Load in SuperSplat (playcanvas.com/supersplat/editor) or gsplat.js viewer to verify")
```

The shipped `.ply` loads directly into browser-based viewers. SuperSplat (PlayCanvas) and gsplat.js both parse the binary PLY format, upload the Gaussian attributes as GPU vertex buffers, and rasterize using a WebGL2 or WebGPU fragment shader. No server-side rendering is needed — the rasterizer runs entirely on the client's GPU. For production GTM pipelines, this means a product page can embed a `<canvas>` element, fetch the `.ply` from a CDN, and render an interactive 3D product viewer with no backend compute cost beyond static file serving.