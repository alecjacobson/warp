"""Render a docs GIF of the UR10 swept volume with headless polyscope (EGL).

Shows the robot arm animating through its poses inside a translucent
swept-volume envelope, from a fixed viewpoint. Requires a GPU with EGL; runs
with no display.

    uv run --with usd-core --with polyscope --with pillow render_swept_gif.py
"""

import os
import sys

import numpy as np
from PIL import Image
import polyscope as ps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "warp/examples/geometry"))

import warp as wp
import warp.geometry as geo
from example_swept_volume import load_usd_assembly

USD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parallel-swept-volume/assets/ur10_animated.usda")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs/img/examples/swept_volume.gif")
device = "cuda:0"

ENVELOPE_SAMPLES = 1800  # dense time sampling stamped into the field (~2 min on GPU)
VOX = 0.015  # dense grid
N_FRAMES = 60  # arm-animation frames (camera is fixed)
W, H = 800, 800


def mat4(t, q):
    x, y, z, w = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


# Meshes + a dense set of poses for the envelope; a coarser set for the animation.
meshes, env_transforms, _ = load_usd_assembly(USD, num_samples=ENVELOPE_SAMPLES, device=device)
_, anim_transforms, _ = load_usd_assembly(USD, num_samples=N_FRAMES, device=device)
print(f"loaded {len(meshes)} meshes; envelope from {ENVELOPE_SAMPLES} poses, {N_FRAMES} animation frames")

# Cache the dense field so the iso level can be re-extracted without recomputing.
cache = f"/tmp/_field_{VOX}_{ENVELOPE_SAMPLES}.npz"
if os.path.exists(cache):
    d = np.load(cache)
    field_np, lo, up = d["field"], d["lo"], d["up"]
else:
    with wp.ScopedTimer("swept_volume_field", print=True):
        field, lower, upper = geo.swept_volume_field(meshes, env_transforms, voxel_size=VOX, sign_mode=geo.SweptVolumeSign.WINDING_NUMBER, device=device)
        wp.synchronize_device()
    field_np = field.numpy()
    lo = np.array([lower[0], lower[1], lower[2]])
    up = np.array([upper[0], upper[1], upper[2]])
    np.savez(cache, field=field_np, lo=lo, up=up)

# Conservative iso = grid covering radius = 0.5*hypot(hx,hy,hz) (= sqrt(3)/2 * h),
# so every stamped arm pose is guaranteed enclosed despite marching-cubes error.
spacing = (up - lo) / (np.array(field_np.shape) - 1)
sigma = 0.5 * float(np.linalg.norm(spacing))
print(f"conservative iso sigma = {sigma * 1000:.1f} mm (spacing {np.round(spacing, 4)})")
verts, indices = wp.MarchingCubes.extract_surface_marching_cubes(
    wp.array(field_np, dtype=wp.float32, device=device),
    threshold=sigma,
    domain_bounds_lower_corner=wp.vec3(*lo),
    domain_bounds_upper_corner=wp.vec3(*up),
)
V = verts.numpy()
F = indices.numpy().reshape(-1, 3)
print(f"envelope: {len(V)} verts, {len(F)} tris")

ps.set_allow_headless_backends(True)
ps.set_program_name("swept volume")
ps.init()
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_ground_plane_mode("shadow_only")
ps.set_background_color((1.0, 1.0, 1.0))
ps.set_transparency_mode("pretty")
ps.set_SSAA_factor(4)
ps.set_window_size(W, H)

env = ps.register_surface_mesh("swept volume", V, F, color=(0.93, 0.42, 0.20), material="wax", smooth_shade=True)
env.set_transparency(0.6)

arm = []
for i, m in enumerate(meshes):
    Vi = m.points.numpy()
    Fi = m.indices.numpy().reshape(-1, 3)
    s = ps.register_surface_mesh(f"link_{i}", Vi, Fi, color=(0.18, 0.22, 0.30), material="clay", smooth_shade=True)
    arm.append(s)

# Fixed camera.
ctr = 0.5 * (V.min(0) + V.max(0))
radius = np.linalg.norm(V.max(0) - V.min(0)) * 1.1
az, el = np.radians(-60.0), np.radians(14.0)
cam = ctr + radius * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
ps.look_at(cam.tolist(), ctr.tolist())

frames = []
for f in range(N_FRAMES):
    for i, s in enumerate(arm):
        s.set_transform(mat4(anim_transforms[i, f, :3], anim_transforms[i, f, 3:]))
    png = f"/tmp/_sv_{f:03d}.png"
    ps.screenshot(png, transparent_bg=False)
    frames.append(Image.open(png).convert("RGB").resize((W // 2, H // 2), Image.LANCZOS))
    print(f"frame {f + 1}/{N_FRAMES}", end="\r")

print()
pal = [im.convert("P", palette=Image.ADAPTIVE, colors=128) for im in frames]
pal[0].save(OUT, save_all=True, append_images=pal[1:], duration=70, loop=0, optimize=True, disposal=2)
print(f"wrote {OUT}  ({os.path.getsize(OUT) / 1024:.0f} KB)")
