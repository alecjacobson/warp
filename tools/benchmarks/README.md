# warp.geometry benchmarks

Standalone scripts (not part of the test suite) for measuring
`warp.geometry.PoissonDiskSampler` and `warp.geometry.register_rigid`.

## Rigid registration (ICP) vs. external baselines

`icp_baselines/` compares `warp.geometry.register_rigid` against **Open3D, PCL,
fast_gicp, and PyTorch3D** on one fixed problem, scoring each against the known
transform. Because the baselines live in mutually-incompatible environments, the
harness writes the problem once and each library registers the identical data in
its own env; see `icp_baselines/README.md` for setup and the full table with
caveats.

Headline (10,242-point ellipsoid, 15° + noise, 50-iter budget). **Every method
runs on the GPU (NVIDIA L40) if it can** — Warp, Open3D tensor ICP, PyTorch3D, and
fast_gicp's `FastVGICPCuda` are CUDA; the other fast_gicp variants and PCL are
CPU. Throughput is registrations/second: batched methods (Warp, PyTorch3D) report
per-problem time at batch saturation, the rest `1000/latency`. Warp figures are
mesh / point-cloud target. The **Warp faster** column is how many times faster
Warp (batched, 365 reg/s) is than that row — bigger = Warp wins by more; every
row is Warp beating that method, not the reverse.

| method                       | objective      | device | batched | rot err (deg) | throughput (reg/s) | Warp faster |
| ---------------------------- | -------------- | ------ | ------- | ------------- | ------------------ | ----------- |
| **warp point-to-plane**      | point-to-plane | GPU    | yes     | 0.000         | **365**            | **1× (ref)** |
| pytorch3d point-to-point     | point-to-point | GPU    | yes     | 2.025         | 165                | 2.2×    |
| fast_gicp (FastGICP)         | plane-to-plane | CPU    | no      | 0.015         | 42                 | 9×      |
| pcl_gicp                     | plane-to-plane | CPU    | no      | 0.008         | 9                  | 42×     |
| open3d point-to-plane        | point-to-plane | GPU    | no      | 0.007         | 5                  | 73×     |
| fast_gicp (FastVGICPCuda)    | voxel GICP     | GPU    | no      | 0.007         | 2.5                | 146×    |

The only other batched GPU method is PyTorch3D: Warp is **2.2× its throughput and
far more accurate** (0.000° vs. 2.025° — PyTorch3D is point-to-point). Notably
fast_gicp's GPU `FastVGICPCuda` (~300 ms) is *slower* than its CPU `FastGICP`
(25 ms) at this size — its CUDA path targets large streaming LiDAR, so a single
10k cloud is overhead-bound (as it is for every method except Warp's batched
path). All GPU-vs-GPU comparisons favor Warp. Full table, per-row device labels,
and caveats in `icp_baselines/README.md`.

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

For a **mesh**, the two track closely on this smooth, finely-tessellated
ellipsoid: `normalize(p − q)` coincides with the face normal for a query
interior to a triangle, and differs only at edges and vertices, which are rare
here — so the numbers match to four digits. On a coarse or sharp-edged mesh,
where more closest points land on edges and vertices, expect them to diverge, so
this equivalence should not be assumed in general. For a **point cloud** the two
differ sharply: the closest-point direction points at a discrete sample rather
than along the surface, and in the table above it has essentially no convergence
basin, while the PCA normals recover the transform. **On this data, keep
`plane_normal="surface"` (the default): it is never worse, and for point-cloud
targets the computed normals are what make point-to-plane converge.**

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

**GPU residency / host syncs.** The whole iteration — correspondence search,
6x6 accumulation, the solve, and the transform update — runs on the device, so
nothing is read back between iterations. With `tol=0` (fixed iterations) the
loop performs no host synchronization at all and is CUDA-graph capturable; with
`tol>0`, early stopping costs a single scalar readback per iteration. Moving the
solve on-device (from an earlier host-side `numpy.linalg.solve`) removed a
per-iteration round-trip: on an L40, small problems dropped from ~3.0 to
~0.44 ms/iter, and per-iteration time now scales with GPU work instead of being
dominated by the host round-trip. All ICP kernels are declared
`enable_backward=False` (no adjoint kernels are generated).

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
