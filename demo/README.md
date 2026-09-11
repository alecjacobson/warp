# Swept-volume interactive demo

A small polyscope viewer for the UR10 swept volume: scrub the robot arm through
its trajectory and compare the two extracted surfaces — the **conservative-iso**
envelope (`iso = sqrt(3)/2 * h`, guaranteed to enclose every stamped pose) and
the **`iso = 0`** envelope (where sharp features can poke through the
marching-cubes reconstruction).

All geometry is precomputed into `swept_volume_demo.npz` (arm link meshes,
per-frame world transforms, and both envelope surfaces), so the viewer needs
only NumPy + polyscope — **no Warp, no GPU, no CUDA**.

## Run it locally

Download `swept_volume_demo.py` and `swept_volume_demo.npz` from this folder, then:

```bash
pip install polyscope numpy
python swept_volume_demo.py                     # expects the .npz alongside
# or: python swept_volume_demo.py /path/to/swept_volume_demo.npz
```

### Controls (left panel)

- **frame** slider (and **play** toggle) — scrub the arm through its trajectory.
- **n** input (default 10) and a **skip forward N/n** button — jump forward by
  `N / n` frames to step through the trajectory in `n` even hops.
- **envelope transparency** slider.
- Toggle the **envelope (conservative iso)** (green) and **envelope (iso = 0)**
  (blue) surfaces with polyscope's built-in per-structure checkboxes in the
  structure list. Enable `iso = 0` and scrub to a pose where the arm reaches a
  sharp extent to see the tool tip protrude past the surface; the conservative
  iso keeps it enclosed.

Rotate/zoom/pan with the mouse as usual in polyscope.

## Regenerating the data (needs a GPU + the UR10 asset)

```bash
uv run --with usd-core python generate_demo_data.py
```

Reads the animated UR10 USD, stamps the swept-volume field (7200 poses, voxel
0.015) with `warp.geometry.swept_volume_field`, extracts marching cubes at both
iso levels, and writes `swept_volume_demo.npz`. Point it at the asset with
`UR10_USD=/path/to/ur10_animated.usda` if it isn't at the default location.
