"""Interactive swept-volume viewer: scrub the UR10 animation and compare iso surfaces.

Runs locally with just NumPy + polyscope (no Warp, no GPU compute) on the
precomputed ``swept_volume_demo.npz`` (see ``generate_demo_data.py``):

    pip install polyscope numpy
    python demo/swept_volume_demo.py            # expects swept_volume_demo.npz alongside
    python demo/swept_volume_demo.py path/to/swept_volume_demo.npz

Controls (left panel): a **frame** slider (and a **play** toggle) scrub the arm
through its trajectory; a **skip** button jumps forward by ``N / n`` frames (with
``n`` choosable) to step through the trajectory in ``n`` even hops; a transparency
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


state = {"frame": 0, "playing": False, "alpha": 0.5, "n": 10}
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
    _, state["n"] = psim.InputInt("n", state["n"])
    state["n"] = max(1, state["n"])
    step = max(1, round(N / state["n"]))
    if psim.Button(f"skip forward N/n ({step} frames)"):
        state["frame"] = (state["frame"] + step) % N
        set_frame(state["frame"])

    psim.Separator()
    ch, state["alpha"] = psim.SliderFloat("envelope transparency", state["alpha"], 0.0, 1.0)
    if ch:
        env_c.set_transparency(state["alpha"])
        env_0.set_transparency(state["alpha"])
    psim.TextUnformatted(f"conservative iso = {iso_mm:.1f} mm; toggle surfaces in the structure list.")


ps.set_user_callback(callback)
ps.show()
