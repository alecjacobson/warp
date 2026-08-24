<!-- MR summary for warp.geometry.swept_volume. Draft to post on GitLab. -->

# Add `swept_volume` to `warp.geometry` [GH-1824]

## Description

Adds a GPU building block that takes triangle meshes plus a time-sampled rigid
transform per mesh and returns a single closed mesh enclosing the union of all
poses over the sampled time range — the *swept volume* / motion envelope
requested in GH-1824.

**Approach.** This is the Track-1 dense-stamping baseline. Because the motion is
rigid, the swept signed-distance field factors as

```
D(p) = min_mesh min_sample  sdf_mesh( X[mesh, sample]^-1 · p )
```

so instead of transforming geometry we push each grid point back into every
mesh's rest frame and reuse Warp's existing closest-point queries
(`mesh_query_point_sign_normal` / `mesh_query_point_sign_winding_number`). The
field is stamped densely over a regular grid and its zero isosurface is
extracted with the existing marching-cubes path. Pure Warp; no new native code.

**Key design decisions.**

- **Dense stamping only.** The field is evaluated by brute force at the provided
  pose samples. No root finding, narrow band, or multigrid — those are the
  "fancier ideas" from the issue and are intentionally out of scope for this
  first MR. Consequence, documented in the API: motion *between* samples is not
  conservatively bounded; the caller controls tolerance via the sampling
  density.
- **Sign classification is selectable** (`SweptVolumeSign`). `NORMAL` (closest
  face normal) is the fast default for watertight input; `WINDING_NUMBER`
  (generalized winding number) is robust to the open, non-watertight visual
  shells typical of CAD assemblies. The UR10 example path defaults to winding
  for exactly this reason (see the example and the note below).
- **Two-tier API**, matching the rest of `warp.geometry`: an array-level
  `swept_volume` / `swept_volume_field`, plus a `@wp.func` `swept_volume_sdf`
  callable from user kernels (the reusable evaluator the issue asked for).
- **Conservative extraction level.** Marching cubes reconstructs the 1-Lipschitz
  field by linear interpolation, which at sharp convex features overestimates it
  and can pull the surface inside the true one, so a stamped pose can poke
  through the `iso = 0` isosurface by up to the grid's covering radius. The `iso`
  docstring gives the principled conservative value
  `iso = 0.5 * hypot(hx, hy, hz)` (= `sqrt(3)/2 * voxel_size` for a cubic cell),
  which guarantees every stamped pose stays enclosed (see Appendix C).

**Alternatives considered and rejected for this MR.** A narrow-band shortcut
(à la libigl's `isolevel`) was prototyped and rejected — it is *unsound* for
swept volumes, and the sound variants gave no compelling GPU speedup. Details in
the appendix so the reasoning isn't lost.

closes GH-1824

## Changes

- **`warp/_src/geometry.py`**: `swept_volume`, `swept_volume_field`,
  `swept_volume_sdf` (`@wp.func`), and the `SweptVolumeSign` enum, plus the
  grid-setup and bounds helpers and the fill kernel.
- **`warp/geometry.py`**: re-export the four public names.
- **`warp/tests/geometry/test_swept_volume.py`**: 20 tests (see Validation),
  registered in `warp/tests/unittest_suites.py`.
- **`warp/examples/geometry/example_swept_volume.py`**: runnable example — a
  self-contained procedural two-link arm by default, `--usd` to run on an
  animated USD assembly (e.g. the UR10), `--sign auto|normal|winding`.
- **`docs/img/examples/swept_volume.gif`** + **`render_swept_gif.py`**: a
  documentation GIF of the UR10 swept volume (arm animating inside its envelope)
  rendered headlessly with polyscope (EGL), and its reproducible render script.
- **`changelog/1824.added.md`**: changelog fragment.

## Checklist

- [x] New or existing tests cover these changes.
- [x] The documentation is up to date with these changes (Google-style
      docstrings on all public API; no API-reference page yet because the
      `warp.geometry` module page is still being established on its own branch).
- [x] I added a changelog fragment.

## Validation summary

- **`test_swept_volume.py` (20 tests, CPU + CUDA).** Analytic and structural
  checks: a translated sphere sweeps a capsule (field compared to the analytic
  capsule SDF within one voxel, and the extracted mesh spans the capsule AABB);
  a single static pose recovers the input sphere; a union of two spheres equals
  the per-sphere min; every posed input vertex lies inside the envelope
  (conservativeness); a rotated off-center sphere exercises the quaternion
  inverse-transform path; explicit-resolution and argument-validation paths.
- **Non-watertight regression.** `test_winding_number_handles_non_watertight`
  builds an open box shell and asserts `WINDING_NUMBER` classifies the interior
  correctly, guarding the CAD/robot use case. (The normal classifier is
  intentionally not asserted here — it is device-dependent on open meshes, which
  is *why* the USD example defaults to winding.)
- **Independent cross-check against libigl** (not in CI; see Appendix A): the
  field matches libigl's `swept_volume_signed_distance` to float precision
  (bunny: max |ΔS| = 0.025 voxels, 100% sign agreement; UR10: identical extracted
  surface, 0.13-voxel Hausdorff, 100% sign at the surface), and the dense GPU
  stamp is 27× faster than libigl's banded CPU path on the full UR10 case.

## New feature / enhancement

```python
import warp as wp
import warp.geometry as geo

mesh = wp.Mesh(points, indices)                 # rest-pose geometry
# transforms: (num_meshes, num_samples) wp.transform, or (M, S, 7) = xyz + xyzw
verts, tris = geo.swept_volume([mesh], transforms, voxel_size=0.02)

# Lower tiers:
field, lo, hi = geo.swept_volume_field([mesh], transforms, voxel_size=0.02)
# geo.swept_volume_sdf(p, mesh_ids, transforms, max_dist, sign_mode)  # @wp.func
```

Run the example:

```bash
uv run --with usd-core warp/examples/geometry/example_swept_volume.py
uv run --with usd-core warp/examples/geometry/example_swept_volume.py --usd ur10_animated.usda
```

---

# Appendix A — libigl cross-validation (investigation notes, not in this MR)

Validated the implementation against libigl's `swept_volume_signed_distance`
(Python bindings from libigl/libigl-python-bindings PR #311). Bunny (6,102 v)
swept along a 20-pose screw trajectory; grid matched exactly via
`igl.swept_volume_bounding_box` + `igl.voxel_grid` (100,800 points, spacing
h = 0.0221); winding-number sign on both sides. Warp on an L40, libigl on CPU.

**Correctness — same SDF to float precision:**

| metric | value |
|---|---|
| max \|ΔS\| | 0.025 voxels |
| correlation | 0.99999966 |
| sign agreement | 100.000% |
| same-grid MC surfaces | identical (Hausdorff 0.002 voxels) |

**Performance (100,800-point grid):**

| workload | libigl (CPU) | Warp (L40) | speedup |
|---|---|---|---|
| SDF field, exact (isolevel=inf) | 13,112 ms | 27.5 ms | **476×** |
| SDF field, banded (isolevel=0) | 1,249 ms | 27.5 ms | **45×** |
| full pipeline (SDF + marching cubes) | 1,236 ms | 25.3 ms | **49×** |

Take-away: the dense GPU stamp is verified correct against libigl and 45–476×
faster depending on which libigl path you compare against.

**UR10 (articulated, 7 links, 268,709 tris) — measured, not extrapolated.**
libigl's single-body API can't do an articulated assembly directly, so the
equivalent is 7 per-link `swept_volume_signed_distance` fields unioned on a
shared grid — exactly what Warp computes. On the full case (voxel 0.015, 1800
poses, 1,040,704-node grid):

| | time | vs Warp |
|---|---|---|
| Warp (L40, exact winding everywhere) | **124 s** | 1× |
| libigl (CPU, banded `isolevel=0`) | **3,381 s ≈ 56.4 min** | **27×** |

Outputs match: at voxel 0.03 / 300 poses on an identical grid, the extracted
0-isosurfaces are bit-identical (12,628 v / 25,204 f each, same bbox, symmetric
Hausdorff 0.13 voxels) with 100% sign agreement at the surface. (Raw field
values disagree away from the 0-level only because libigl was run banded, which
by design returns approximate far values; Warp returns the exact SDF.)

# Appendix B — why not a narrow band (yet)

A narrow-band shortcut was prototyped and **deferred** — recording why so it
isn't re-attempted blindly.

1. **A distance band is unsound for swept volumes.** libigl's `isolevel`-focused
   band works for a static SDF, but here a point deep inside the union can lie
   arbitrarily close to a *grazing* pose's surface (a mesh sweeping through it),
   so band-limiting the per-pose query drops the far pose that actually makes it
   interior. Measured failure: 100% of errors were deep-interior nodes flipped
   to exterior; spurious surfaces, Hausdorff of hundreds of voxels.

2. **Per-pose stamping (loop time outside the spatial launch) is sound but slow
   on GPU.** Each pose stamps only its own AABB, so a grazing-pose cell is caught
   by the pose that contains it (verified Hausdorff 0). But it measured 0.27–2.2×
   and *worsens* as the brush mesh gets heavier (0.27× at 82k tris). Cause is
   concurrency, not launch overhead (a CUDA graph changed nothing): the dense
   single launch keeps hundreds of thousands of high-latency winding queries in
   flight and hides their latency, whereas 20 sequential per-pose launches keep
   far fewer in flight and expose it.

3. **Exact in-kernel AABB pruning is sound but modest.** Outside a solid's AABB
   the winding number is 0, so the expensive sign is only needed inside some
   pose's AABB. Pruning per pose within one launch is bit-identical to dense and
   keeps concurrency, but only ~1.0–1.4× (rising to ~1.23× as the brush grows to
   82k tris) because the running `best` stays loose for exterior points. Two
   subtleties that make naive attempts wrong: (a) once `best < 0` you must still
   process poses whose AABB *contains* the point; (b) the query must use the
   fixed full `max_dist`, never the shrinking `best` (that is itself a distance
   band and misses deep-inside poses).

**Note:** Warp's `mesh_query_point_sign_winding_number` already uses the *fast
hierarchical* winding number (Barill et al. 2018; BVH + per-node solid-angle
Taylor expansion with a Barnes-Hut `accuracy` cutoff), ~O(log F) — so there is
no fast-winding gap vs libigl to close, and the sign query is not the bottleneck
a narrow band could remove.

**Conclusion.** The dense method is the right choice for this MR: it saturates
the GPU and already beats libigl CPU by 45–476×. A *robust* band speedup needs
the coarse-to-fine / multigrid path (locate the surface coarsely, refine only
the band), which is the deferred follow-up and would also give the sparse
narrow-band allocation for free.

# Appendix C — conservative `iso` (containment guarantee)

Marching cubes can pull the extracted surface inside the true one at sharp
convex features, so a stamped pose can poke through the `iso = 0` surface. Since
the field is 1-Lipschitz, the maximum overshoot of the (tri)linear interpolant
over the true field is the grid's **covering radius** — the farthest any point
in a cell sits from the nearest node:

```
iso = 0.5 * hypot(hx, hy, hz)   ( = sqrt(3)/2 * voxel_size ≈ 0.87 h, cubic cell )
```

Extracting at that level guarantees the true solid — hence every stamped pose —
is enclosed by the trilinear isosurface. This is the value the docstring
recommends and the docs GIF uses.

The *actual* marching-cubes triangulation (flat triangles vs. the curved
trilinear isosurface) needs slightly more. Brute-forcing the worst inward
intrusion over all MC cases (Warp's own triangle table) and 1-Lipschitz node
configurations gives `gamma_max ≈ 0.48 h` (at 4–5-corner "diagonal" cases;
exact for simple cases: single corner `8/27 h`, face saddle `1/8 h`), so the
MC-airtight level is `≈ sqrt(3)/2 h + 0.48 h ≈ 1.34 h`, comfortably below the
trivially-provable whole-cell diagonal `sqrt(3) h ≈ 1.73 h`.
