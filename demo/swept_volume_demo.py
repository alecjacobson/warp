"""Interactive swept-volume viewer: scrub the UR10 animation and compare iso surfaces.

Runs locally with just NumPy + polyscope (no Warp, no GPU compute) on the
precomputed ``swept_volume_demo.npz`` (see ``generate_demo_data.py``):

    pip install polyscope numpy
    python demo/swept_volume_demo.py            # expects swept_volume_demo.npz alongside
    python demo/swept_volume_demo.py path/to/swept_volume_demo.npz

Controls (left panel): a **frame** slider (and a **play** toggle) scrub the arm
through its trajectory; a **show robot at n time samples** checkbox overlays the
arm at ``n`` uniformly spaced instants along the trajectory; a transparency
slider controls the envelope opacity. Toggle the two envelope surfaces (the
conservative covering-radius iso and the ``iso = 0`` iso, where sharp features
can poke through the marching-cubes reconstruction) with polyscope's built-in
per-structure checkboxes in the structure list.
"""

import os
import sys

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

# Colors (RGB in [0, 1]).
LINK_COLOR = (1.0, 0.0, 0.0)  # #FF0000
CONSERVATIVE_COLOR = (0x33 / 255, 0xED / 255, 0x34 / 255)  # #33ED34
ISO0_COLOR = (0x35 / 255, 0xA8 / 255, 0xEC / 255)  # #35A8EC

npz = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "swept_volume_demo.npz")
d = np.load(npz)
num_links = int(d["num_links"])
N = int(d["n_frames"])
xforms = d["xforms"]  # (N, num_links, 7): translation xyz + quaternion xyzw
iso_mm = float(d["iso_conservative_mm"])
link_V = [d[f"link_V_{i}"] for i in range(num_links)]
link_F = [d[f"link_F_{i}"] for i in range(num_links)]


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


ps.init()
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_ground_plane_mode("shadow_only")
ps.set_transparency_mode("pretty")

env_c = ps.register_surface_mesh("envelope (conservative iso)", d["env_c_V"], d["env_c_F"], color=CONSERVATIVE_COLOR, material="wax", smooth_shade=True)
env_0 = ps.register_surface_mesh("envelope (iso = 0)", d["env0_V"], d["env0_F"], color=ISO0_COLOR, material="wax", smooth_shade=True)
env_0.set_enabled(False)

links = [
    ps.register_surface_mesh(f"link_{i}", link_V[i], link_F[i], color=LINK_COLOR, material="clay", smooth_shade=True)
    for i in range(num_links)
]


def set_frame(f):
    for i, s in enumerate(links):
        x = xforms[f, i]
        s.set_transform(mat4(x[:3], x[3:]))


def build_time_samples(n):
    """Register one merged mesh of the arm posed at ``n`` uniform instants."""
    frames = np.unique(np.linspace(0, N - 1, max(n, 1)).round().astype(int))
    verts, faces, offset = [], [], 0
    for f in frames:
        for i in range(num_links):
            x = xforms[f, i]
            M = mat4(x[:3], x[3:])
            verts.append(link_V[i] @ M[:3, :3].T + M[:3, 3])
            faces.append(link_F[i] + offset)
            offset += len(link_V[i])
    g = ps.register_surface_mesh("time samples", np.vstack(verts), np.vstack(faces), color=LINK_COLOR, material="clay", smooth_shade=True)
    g.set_transparency(0.5)
    return g


state = {"frame": 0, "playing": False, "alpha": 0.5, "samples": False, "n": 10, "built_n": None}
set_frame(0)
env_c.set_transparency(state["alpha"])
env_0.set_transparency(state["alpha"])


def callback():
    _, state["playing"] = psim.Checkbox("play", state["playing"])
    if state["playing"]:
        state["frame"] = (state["frame"] + 1) % N
        set_frame(state["frame"])
    changed, state["frame"] = psim.SliderInt("frame", state["frame"], 0, N - 1)
    if changed:
        set_frame(state["frame"])

    psim.Separator()
    _, state["samples"] = psim.Checkbox("show robot at n time samples", state["samples"])
    n_changed, state["n"] = psim.InputInt("n", state["n"])
    state["n"] = max(1, state["n"])
    if state["samples"]:
        if state["built_n"] != state["n"] or n_changed:
            build_time_samples(state["n"])
            state["built_n"] = state["n"]
        ps.get_surface_mesh("time samples").set_enabled(True)
    elif ps.has_surface_mesh("time samples"):
        ps.get_surface_mesh("time samples").set_enabled(False)

    psim.Separator()
    ch, state["alpha"] = psim.SliderFloat("envelope transparency", state["alpha"], 0.0, 1.0)
    if ch:
        env_c.set_transparency(state["alpha"])
        env_0.set_transparency(state["alpha"])
    psim.TextUnformatted(f"conservative iso = {iso_mm:.1f} mm; toggle surfaces in the structure list.")


ps.set_user_callback(callback)
ps.show()
