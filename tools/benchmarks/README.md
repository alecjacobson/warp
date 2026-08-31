# warp.geometry benchmarks

Standalone scripts (not part of the test suite) for measuring
`warp.geometry.PoissonDiskSampler` and `warp.geometry.register_rigid`.

## Rigid registration (ICP) vs. external baselines

`icp_vs_baselines.py` registers a point cloud onto a known-transformed copy of
itself and compares `warp.geometry.register_rigid` (point-to-plane and
symmetric) against optional, import-guarded baselines — Open3D's tensor ICP,
PyTorch3D, and fast_gicp — on accuracy (rotation/translation error vs. ground
truth) and wall-clock (full solve, including each library's internal spatial
structure). The harness records which baselines are installed and skips the rest.

```sh
uv run --with open3d --with scipy tools/benchmarks/icp_vs_baselines.py
```

Example (10,242-point ellipsoid, 15° + translation, 2e-3 noise, NVIDIA L40;
Open3D from the CPU-only PyPI wheel, best of 3):

| method                | rot err (deg) | trans err | time (ms) |
| --------------------- | ------------- | --------- | --------- |
| warp_point_to_plane   | 0.008         | 7e-5      | 48        |
| warp_symmetric        | 0.003         | 1e-4      | 48        |
| open3d_point_to_plane | 0.007         | 7e-5      | 201       |
| open3d_point_to_point | 2.024         | 1.8e-3    | 1901      |

Warp matches Open3D's point-to-plane accuracy at ~4x the throughput here, and the
symmetric variant is the most accurate. (A CUDA Open3D build would narrow the gap;
the numbers above use the CPU wheel, which the script reports.)

## Rigid registration ablation — do the options help?

`icp_ablation.py` sweeps `register_rigid`'s options on an ellipsoid (as both a
mesh and a point-cloud target) under four data conditions (clean, noisy,
outliers, partial overlap), scoring against the known transform. Run it:

```sh
uv run tools/benchmarks/icp_ablation.py
```

Findings on an NVIDIA L40 (10,242-point ellipsoid, 15° + translation):

**1. Computed normals vs. the query-closest direction
(`plane_normal="surface"` vs `"closest_point"`).** They answer *different*
questions for the two target types:

| target | plane_normal    | clean rot err | convergence basin |
| ------ | --------------- | ------------- | ----------------- |
| mesh   | surface         | 0.003°        | 70°               |
| mesh   | closest_point   | 0.003°        | 70°               |
| cloud  | surface (PCA)   | 0.003°        | 70°               |
| cloud  | closest_point   | **2.13° ✗**   | **0° ✗**          |

For a **mesh**, the closest-point direction `normalize(p − q)` *is* the face
normal for any query interior to a triangle, so the two are identical — computed
normals buy nothing, and `"closest_point"` is a fine no-normal shortcut. For a
**point cloud**, the closest-point direction points at a discrete sample rather
than along the surface, so it has essentially no convergence basin and stalls;
the PCA normals are what make point-to-plane work at all. **Conclusion: keep
`plane_normal="surface"` (the default). Computed normals are essential for
point-cloud targets and redundant (but harmless) for meshes.**

**2. Variants.** The symmetric objective (Rusinkiewicz 2019) widens the basin
modestly on this smooth shape — 75° vs 70° for point-to-plane — at ~2× the
iterations to converge; the gain grows on higher-curvature geometry. Plain
point-to-plane is the fastest default; reach for `variant="symmetric"` when the
initialization is poor.

**3. Robust weighting & subsampling.** On clean data, `robust="welsch"` is a
no-op (0.003° either way) and stochastic `sample_count` trades a little accuracy
(0.003°→0.015°) for ~15% less time. On **20% gross outliers**, robust weighting
is decisive — 0.008° vs 0.126° for plain least squares (~15× better) — while
subsampling alone does not help. A tighter robust scale rejects outliers harder
(`robust_k` 1.0→0.001°, 6.0→0.039° at 20% outliers); the default `robust_k=3`
is a safe middle ground, but drop it toward ~1.5–2 when inliers are clean and
outliers are heavy (raise it when inliers are noisy).

**4. `max_corr_dist` on partial overlap.** Too tight loses correspondences and
misses (0.02 → 1.1° ✗); past the point where all real overlap is captured,
accuracy plateaus and only runtime grows (0.05 → 39 ms, 0.8 → 211 ms). Set it a
few times the point spacing / expected overlap gap — large enough to match, no
larger.

**Best settings by scenario:**

| scenario                    | recommended options                                              |
| --------------------------- | --------------------------------------------------------------- |
| mesh, good init             | defaults (point-to-plane, surface)                              |
| point cloud, good init      | defaults; **do not** use `plane_normal="closest_point"`         |
| poor / unknown init         | `variant="symmetric"` (or a `register_rigid_batched` multi-start) |
| outliers present            | `robust="welsch"`, `robust_k≈2`                                 |
| partial overlap             | `max_corr_dist` ≈ a few × point spacing                         |
| very large source, speed    | `sample_count` ≈ N/4 (accept a small accuracy hit)              |

## Poisson-disk vs. farthest-point sampling

`poisson_vs_fps.py` compares our parallel Poisson-disk sampler against
farthest-point sampling (FPS). Both select points from the **same** dense
candidate pool (generated once with `warp.geometry.UniformSampler`), and FPS is
asked for exactly as many points as the Poisson sampler produced, so it is an
apples-to-apples comparison of two ways to thin the same pool.

`warp_fps.py` is a pure-Warp FPS (the block-aware, radix-sort + Tile-API
algorithm from [NVIDIA Kaolin](https://github.com/NVIDIAGameWorks/kaolin),
adapted to a NumPy interface so no PyTorch is needed).

Run it:

```sh
uv run --with usd-core --with matplotlib tools/benchmarks/poisson_vs_fps.py
```

### Fairness

FPS is timed at full speed, with no host stalls: the vendored implementation
uploads the pool *before* starting its timer, and its main loop never
synchronizes with the host (it tracks a min/max estimate of progress and reads
back the exact count only when it might be done). As a check, the benchmark
reproduces the FPS author's reference point, N=10^6, k=1024:

```
[validation] FPS N=1e6 k=1024: ~22 ms   (author: 26.5 ms on RTX 3090 Ti)
```

(The L40 lands a bit under the 3090 Ti figure, as expected.)

The comparison is also conservative toward FPS: FPS is handed the candidate pool
for free, while the Poisson sampler is shown both `solve` (thinning the same
pool -- the apples-to-apples number) and `total` (including generating the pool
that FPS got gratis).

### Results (Stanford bunny, NVIDIA L40, best of 3)

| radius | output pts | candidates | PDS solve | PDS total | FPS       | FPS / PDS-solve | min-dist/r (PDS / FPS) |
| ------ | ---------- | ---------- | --------- | --------- | --------- | --------------- | ---------------------- |
| 0.020  | 9,758      | 219,936    | 3.6 ms    | 4.1 ms    | 24.9 ms   | 7x              | 1.000 / 0.959          |
| 0.010  | 39,087     | 879,745    | 5.3 ms    | 5.8 ms    | 122.2 ms  | 23x             | 1.000 / 0.960          |
| 0.005  | 156,796    | 3,518,980  | 16.8 ms   | 19.3 ms   | 1282.5 ms | 76x             | 1.000 / 0.960          |

Candidate generation is cheap (`total - solve` < 1 ms), so the two PDS columns
nearly coincide. The Poisson sampler sustains ~9M samples/s (the phase passes
are cache-friendly because the candidates are cell-sorted first -- see below).

FPS cost scales with the **output** count `k`: each round radix-sorts the whole
pool and accepts a head-chunk of up to 512 points, so it needs ~`k/512` sorts.
That is cheap for small `k` (the author's k=1024 is ~2 rounds), but a
blue-noise *surface* sampling typically wants tens of thousands of points, where
FPS does hundreds of full-pool sorts. The Poisson sampler is instead a fixed
27-pass parallel sweep whose cost scales with the candidate pool, not with `k`.

### Spectral quality

![pair correlation](poisson_vs_fps_pcf.png)

The pair-correlation function `g(r)` (the differential-domain blue-noise
measure) for both methods on the same output count:

- **Poisson-disk** has a hard gap that ends exactly at the radius (`min-dist =
  1.000 r`, guaranteed) and a **sharp** first-neighbor peak (~2.7).
- **FPS** has a **softer, broader** peak (~1.8) and lets a few pairs fall
  slightly inside the radius (`min-dist ~ 0.96 r`), since it only greedily
  maximizes distance over a fixed candidate set with no hard radius.

Both are blue noise (no low-frequency power, a peak near the mean spacing,
decaying to 1), but the Poisson-disk sampler gives a stronger, cleaner
blue-noise signature and a guaranteed minimum distance -- at a fraction of the
cost.

### What the comparison taught us

FPS leans hard on `radix_sort_pairs` to keep its per-round work local. That is a
reminder that our own bottleneck is memory locality: the phase passes are bound
on random single-entry-hash lookups. Sorting the candidate pool by grid cell
first (one `radix_sort_pairs` + a gather, all on the GPU) makes neighboring
threads read overlapping `5x5x5` blocks, turning those lookups into cache hits.
That change alone roughly halved the solve time (~4.5M -> ~9M samples/s) with an
identical result, and is now built into `PoissonDiskSampler`.
