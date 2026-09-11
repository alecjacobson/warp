"""Precompute the swept-volume demo data (run on a GPU machine with the UR10 asset).

Produces ``swept_volume_demo.npz`` containing everything the viewer needs — the
robot-arm link meshes, their per-frame world transforms, and the swept-volume
surface extracted at two iso levels (``0`` and the conservative covering
radius). The viewer (``swept_volume_demo.py``) needs only NumPy + polyscope; no
Warp or GPU.

    uv run --with usd-core python demo/generate_demo_data.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "warp/examples/geometry"))

import warp as wp
import warp.geometry as geo
from example_swept_volume import load_usd_assembly

USD = os.environ.get(
    "UR10_USD",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "parallel-swept-volume/assets/ur10_animated.usda"),
)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swept_volume_demo.npz")

device = "cuda:0"
ENVELOPE_SAMPLES = 7200  # dense temporal stamping for a smooth envelope
VOX = 0.015
FIELD_CACHE = f"/tmp/_field_{VOX}_{ENVELOPE_SAMPLES}.npz"
N_FRAMES = 120  # animation frames the viewer scrubs through


def quat_from_R(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    if m[1, 1] > m[2, 2]:
        s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]


meshes, env_transforms, _ = load_usd_assembly(USD, num_samples=ENVELOPE_SAMPLES, device=device)
_, anim_transforms, _ = load_usd_assembly(USD, num_samples=N_FRAMES, device=device)  # (num_links, N_FRAMES, 7)
num_links = len(meshes)
print(f"loaded {num_links} links; {ENVELOPE_SAMPLES} envelope poses; {N_FRAMES} animation frames")

# Swept-volume field (correct full-domain path), cached across runs.
if os.path.exists(FIELD_CACHE):
    d = np.load(FIELD_CACHE)
    field_np, lo, up = d["field"], d["lo"], d["up"]
else:
    field, lower, upper = geo.swept_volume_field(
        meshes, env_transforms, voxel_size=VOX, narrow_band=False, sign_mode=geo.SweptVolumeSign.WINDING_NUMBER, device=device
    )
    wp.synchronize_device()
    field_np = field.numpy()
    lo = np.array([lower[0], lower[1], lower[2]])
    up = np.array([upper[0], upper[1], upper[2]])
    np.savez(FIELD_CACHE, field=field_np, lo=lo, up=up)

spacing = (up - lo) / (np.array(field_np.shape) - 1)
sigma = 0.5 * float(np.linalg.norm(spacing))  # conservative iso = grid covering radius
field_wp = wp.array(field_np, dtype=wp.float32, device=device)


def extract(iso):
    v, i = wp.MarchingCubes.extract_surface_marching_cubes(
        field_wp, threshold=iso, domain_bounds_lower_corner=wp.vec3(*lo), domain_bounds_upper_corner=wp.vec3(*up)
    )
    return v.numpy().astype(np.float32), i.numpy().reshape(-1, 3).astype(np.int32)


Vc, Fc = extract(sigma)
V0, F0 = extract(0.0)
print(f"envelope conservative(iso={sigma * 1000:.1f}mm): {len(Vc)}v  |  iso0: {len(V0)}v")

data = {
    "num_links": num_links,
    "n_frames": N_FRAMES,
    "xforms": anim_transforms.transpose(1, 0, 2).astype(np.float32),  # (N_FRAMES, num_links, 7)
    "iso_conservative_mm": np.float32(sigma * 1000.0),
    "env_c_V": Vc,
    "env_c_F": Fc,
    "env0_V": V0,
    "env0_F": F0,
}
for m_i, mesh in enumerate(meshes):
    data[f"link_V_{m_i}"] = mesh.points.numpy().astype(np.float32)
    data[f"link_F_{m_i}"] = mesh.indices.numpy().reshape(-1, 3).astype(np.int32)

np.savez_compressed(OUT, **data)
print(f"wrote {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB)")
