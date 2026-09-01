# Rigid registration (ICP) vs. external baselines

Compares `warp.geometry.register_rigid` against Open3D, PCL, fast_gicp
([koide3/fast_gicp](https://github.com/koide3/fast_gicp)), and PyTorch3D on one
fixed problem, scoring each against the known ground-truth transform.

Every baseline lives in a different (often conflicting) environment, so the
harness is split: `problem.py` writes the shared problem once, each runner
registers it and appends to a results JSON, and `report.py` prints the table.

## The problem

A ~10k-point ellipsoid displaced by a known 15° rotation + 0.11 translation with
2e-3 Gaussian noise; the source registers back onto the target. Target normals
are estimated once by PCA and handed to every normal-using method, so none is
charged for computing them. Runtime is the full solve including whatever spatial
structure each library builds internally (Warp's hash grid / BVH, the baselines'
KD-trees) — the fair measure for a one-shot registration. Each number is the
best of three runs.

## Running it

```sh
# 1. Write the shared problem (needs scipy for the PCA normals).
uv run --with scipy tools/benchmarks/icp_baselines/problem.py

# 2. Warp, in the project environment.
uv run tools/benchmarks/icp_baselines/run_warp.py

# 3. The Python baselines, each in its own venv (they don't co-install cleanly):
uv run --python 3.12 --with open3d tools/benchmarks/icp_baselines/run_python.py
#    fast_gicp: build koide3/fast_gicp's python binding (CPU, and GPU with
#    -DBUILD_VGICP_CUDA=ON — see the patch note below), then run with its
#    libfast_gicp.so on LD_LIBRARY_PATH; PyTorch3D: install a CUDA wheel.

# 4. PCL (C++): build the binary, then score it.
cmake -S tools/benchmarks/icp_baselines/pcl -B /tmp/pcl_build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/pcl_build -j
uv run --with scipy tools/benchmarks/icp_baselines/run_pcl.py /tmp/pcl_build/pcl_icp

# 5. Print the merged table.
uv run tools/benchmarks/icp_baselines/report.py
```

Baseline versions used below: Open3D 0.19 (PyPI wheel, CUDA), PCL 1.12 (CPU),
fast_gicp built from source (CPU variants, plus `FastVGICPCuda` on GPU),
PyTorch3D 0.7.5 (CUDA wheel for torch 2.1 + cu121, run on the L40).

**Building fast_gicp's `FastVGICPCuda` on CUDA 12.6.** Its CUDA headers
forward-declare `namespace thrust { template<...> class pair; }`, which clashes
with CCCL 2.5's `thrust::pair` (now a `cuda::std::pair` alias), so the build
fails with "thrust::pair is ambiguous". Replace that forward-declaration block in
`include/fast_gicp/cuda/fast_vgicp_cuda.cuh` and `ndt_cuda.cuh` with real
includes (`#include <thrust/pair.h>`, `<thrust/device_allocator.h>`,
`<thrust/device_vector.h>`), then configure with `-DBUILD_VGICP_CUDA=ON
-DCMAKE_CUDA_ARCHITECTURES=89`.

NVIDIA's [cuPCL](https://github.com/NVIDIA-AI-IOT/cuPCL) (`cuICP`) is also
included (GPU point-to-point ICP). Use its **`x86_64_lib`** branch — the default
branch ships aarch64/Jetson binaries. The prebuilt `libcudaicp.so` links against
the CUDA runtime and can be driven from a tiny harness (it takes float4-packed
clouds and returns a column-major 4×4).

## Results — accuracy and throughput

10,242-point problem, 50-iteration budget. **Every method runs on the GPU (an
NVIDIA L40) if it can** — Warp, Open3D tensor ICP, PyTorch3D, and fast_gicp's
`FastVGICPCuda` are CUDA; the other fast_gicp variants and PCL are CPU. (Building
`FastVGICPCuda` against CUDA 12.6 needs a one-line patch — see below.)
"throughput" is registrations/second: methods with a batch API (Warp, PyTorch3D)
report the per-problem time at batch saturation (B ≥ 64); single-problem methods
report `1000 / latency` (they have no batch API, so N registrations cost N× the
latency). Warp figures are mesh / point-cloud target. **speedup** = Warp's batched
point-cloud throughput (364 reg/s) ÷ that method's — the apples-to-apples
point-cloud comparison.

These tables are generated from a single consolidated run (all methods, one
session, identical fixed problems and settings); the raw numbers are in
[`results.json`](results.json). Warp figures are point-cloud target (batched =
per-problem at B=64; single in parentheses). `speedup` = Warp's batched
throughput (365 reg/s) ÷ that method's.

| method                              | objective      | device | batched? | rot err (deg) | throughput (reg/s) | speedup |
| ----------------------------------- | -------------- | ------ | -------- | ------------- | ------------------ | ------- |
| **warp point-to-plane**             | point-to-plane | GPU    | yes      | 0.000         | **365** (32 single)| **1× (ref)** |
| pytorch3d point-to-point            | point-to-point | GPU    | yes      | 2.025         | 165                | 2.2×    |
| fast_gicp `FastGICP`                | plane-to-plane | CPU    | no       | 0.015         | 42                 | 9×      |
| fast_gicp `FastVGICP`               | voxel GICP     | CPU    | no       | 0.491         | 28                 | 13×     |
| cupcl cuICP                         | point-to-point | GPU    | no       | 2.010         | 13                 | 29×     |
| pcl_gicp                            | plane-to-plane | CPU    | no       | 0.008         | 9                  | 42×     |
| open3d point-to-plane               | point-to-plane | GPU    | no       | 0.007         | 5                  | 73×     |
| pcl point-to-point                  | point-to-point | CPU    | no       | 2.022         | 3                  | 111×    |
| fast_gicp `FastVGICPCuda`           | voxel GICP     | GPU    | no       | 0.007         | 2.5                | 146×    |
| open3d point-to-point               | point-to-point | GPU    | no       | 2.024         | 0.5                | 730×    |

(`warp point-to-plane` on a *mesh* target is 47 reg/s single; the point-cloud
figure is shown above as the apples-to-apples comparison with the point-cloud
baselines.)

**fast_gicp on the GPU is *slower* than on the CPU here** — `FastVGICPCuda` is
accurate (0.000°) but ~300 ms (≈200 ms if the target voxelmap is prebuilt and
reused) vs. 25 ms for CPU `FastGICP`. Its CUDA path is built for large streaming
LiDAR (thousands of frames against one reused voxelmap), so at a single
10k-point cloud it is overhead-bound — the same reason a single small problem
underfills the L40 for Warp. It does not become competitive at larger clouds
either (≈1.2 s at 160k). So the CPU `FastGICP` is the fast_gicp variant to beat
here, and Warp still leads it 9× (batched).

Single-problem (no batch), the same methods are: Warp 47 / 32 reg/s (GPU),
PyTorch3D 15 reg/s (GPU, 68 ms), Open3D point-to-plane 5 reg/s (GPU, 202 ms),
fast_gicp `FastGICP` 40 reg/s (CPU, 25 ms), `FastVGICPCuda` 3 reg/s (GPU, 300 ms).
Batched registration is point-to-plane only for Warp; its symmetric variant is a
single-problem option (same ~0.000°, ~22 ms).

**The batched win:** the only other GPU method with a batch API is PyTorch3D, so
that is the fair head-to-head — Warp registers **364 clouds/s vs. PyTorch3D's
165/s** (a 2.2× throughput edge) *and* is far more accurate (0.000° vs. 2.025°,
since PyTorch3D is point-to-point). Against the single-problem methods it is 9× a
strong CPU baseline (`FastGICP`), 74× Open3D's GPU point-to-plane, and 110× the
GPU `FastVGICPCuda`. Single-problem, Warp is merely competitive with the fastest
CPU baseline (47/32 reg/s vs. `FastGICP`'s 40); the GPU pulls away with scale and
batch, because one 10k-point problem underfills the L40 while a batch saturates
it (see below).

## Reading the table honestly

- **Device: GPU wherever possible, labeled per row.** Warp, Open3D (tensor ICP),
  PyTorch3D, and fast_gicp's `FastVGICPCuda` run on the L40; PCL and the other
  fast_gicp variants are CPU-only. So the headline comparisons (Warp vs.
  PyTorch3D, vs. Open3D, vs. `FastVGICPCuda`) are all GPU-vs-GPU — and Warp still
  wins each, because a 10k-point cloud underfills the GPU for every method except
  Warp's batched path. Open3D's
  tensor ICP is ~5× faster on CUDA than CPU (202 vs. 1000 ms); PyTorch3D's
  point-to-point is ~220× faster on CUDA than CPU (68 ms vs. 15 s).
- **Batching is not universal.** Only Warp and PyTorch3D expose a batch API and
  amortize the fixed per-launch cost across problems; Open3D, PCL, and fast_gicp
  are single-problem, so their throughput is `1000 / latency` (running N in
  parallel across CPU cores does not raise the ceiling — the CPU is already busy
  on one). This is why Warp's throughput lead is largest over the non-batched
  methods.
- **The objectives differ.** Point-to-plane (Warp, Open3D) and plane-to-plane
  GICP (PCL, fast_gicp) slide points along the surface and converge tightly;
  point-to-point (PyTorch3D, and the PCL/Open3D point-to-point variants) minimizes
  a weaker metric and lands ~2° off on this noisy problem. Compare like-for-like:
  Warp point-to-plane vs. Open3D point-to-plane, PyTorch3D as the other batched
  GPU method, and GICP as a strong plane-to-plane reference.
- **Takeaway.** Among GPU methods Warp is both the most accurate (0.000°) and, at
  batch, the highest throughput (364 reg/s) — 2.2× the next batched GPU method
  (PyTorch3D) with far better accuracy, and 74× the non-batched Open3D. fast_gicp
  `FastGICP` remains the strongest CPU baseline (25 ms, 0.015°).

## Why isn't the GPU further ahead? (10k points is the CPU's best case)

Profiling the Warp path (L40, mesh target, 10k points) locates the cost
precisely:

- **It is compute-bound in the correspondence/accumulation kernel, not launch- or
  sync-bound.** Per iteration the BVH build is amortized (0.4 ms once) and the
  6×6 solve kernel is 0.02 ms, but the accumulate kernel is ~1.0 ms and
  dominates. That ~1.0 ms is essentially all the closest-point query, not the
  accumulation: dropping 26 of its 27 atomic-adds changes it by only ~1.4 %
  (1.043 → 1.028 ms), so atomic contention is not the bottleneck — the BVH
  traversal in `mesh_query_point` is. A whole mesh solve is ~18–30 ms
  (≈6–23 iterations).
- **Graph capture confirms this.** Capturing the fixed-iteration loop and
  replaying it is only ~1% faster than the eager (sync-free, `tol=0`) loop —
  there is essentially no per-launch overhead left to remove at this kernel size.
  Graph capture's value here is eliminating the per-iteration host sync and
  letting ICP be embedded in a larger captured pipeline, **not** a speedup for
  this workload. So the gap to fast_gicp isn't overhead Warp is failing to hide;
  it is that one 10k-point problem simply does not fill the L40 (low occupancy —
  the SMs are mostly idle), and a mature multithreaded C++ library is hard to
  beat there. Warp's *mesh* path (~18–30 ms) already matches or beats fast_gicp
  (25 ms); it is mainly the point-cloud variants that trail.
- **Batching fills the GPU and flips the result** (10k pts each, 50 iters):

  | batch B | mesh ms/problem (reg/s) | cloud ms/problem (reg/s) |
  | ------- | ----------------------- | ------------------------ |
  | 1       | 21.7 (46)               | 29.7 (34)                |
  | 16      | 4.25 (235)              | 4.33 (231)               |
  | 64      | 3.97 (252)              | 2.74 (364)               |
  | 256     | 3.77 (265)              | 2.48 (402)               |

  Per-problem cost falls ~6–12× and plateaus once the machine saturates (~B=64):
  ~250 (mesh) to ~360 (cloud) registrations/s vs. fast_gicp's ~40/s of sequential
  single-problem solves — a ~6–9× throughput win. This is the regime the
  multi-start / multi-view example actually uses.
- **Source points are Morton-reordered once so the queries stay coherent.** The
  closest-point traversal is cache-/warp-coherent when neighboring threads query
  neighboring points, so `register_rigid` Z-orders the source up front (inert to
  the result — the normal equations are an order-independent sum). This pays off
  once there is enough parallel work to expose the coherence: ~1.4× at 10k in a
  batch, and **~4.3× on a single 1M-point cloud** (88.8 → 20.6 ms/iter); it is
  roughly neutral on tiny single problems. The batched numbers above already
  include it.
- **The point-cloud path is also charged for `max_corr_dist`.** For a point-cloud
  target that distance sizes the hash grid used for nearest-neighbor search, so
  the query cost grows like `(max_corr_dist / point_spacing)³`. The benchmark's
  generous `max_corr_dist=0.3` costs ~2.6 ms/iter; a sensible bound (~0.05, a few
  × the spacing) drops it to ~1.0 ms/iter, matching the mesh path. A KD-tree (the
  CPU baselines) finds the nearest neighbor in `O(log N)` regardless of the bound,
  so a large bound penalizes only the grid — a real asymmetry (and the mechanism
  behind the ablation's "`max_corr_dist` past the overlap only costs time"). A
  mesh target's BVH is insensitive to the bound.
- **Where the GPU wins:** point count (throughput scales up while a CPU falls
  behind) and batch (many simultaneous registrations). Read the single-problem
  table as "competitive out of the box at small scale," not the ceiling.

## Scenario 2 — real LiDAR (KITTI velodyne), the regime a GPU should favor

A single small synthetic cloud is the GPU's worst case, so this second scenario
uses the two consecutive Velodyne scans shipped with fast_gicp
(`data/251370668.pcd` → `251371071.pcd`, ~69k points each, ~0.6° + 0.49 m real
motion, ground truth in their `relative.txt`). This is exactly what fast_gicp's
CUDA path and cuPCL target: large scans, and for the streaming methods a target
structure that is **built once and reused** across frames (odometry), which is
timed separately below.

| method                          | device | time (ms)         | rot err (°) | trans err | converged |
| ------------------------------- | ------ | ----------------- | ----------- | --------- | --------- |
| pcl_gicp                        | CPU    | 1466              | 0.000       | 0.002     | yes       |
| fast_gicp `FastGICP`            | CPU    | 617 / **505 reuse** | 0.021     | 0.001     | yes       |
| warp point-to-plane             | GPU    | 552 / **126 batch (B=16)** | 0.291 | 0.062 | yes    |
| open3d point-to-plane           | GPU    | 1200              | 0.139       | 0.017     | yes       |
| fast_gicp `FastVGICPCuda`       | GPU    | 704 / 206 reuse   | 1.05        | 0.51      | **no**\*  |
| cupcl cuICP                     | GPU    | 1725              | 0.564       | 0.177     | partial   |
| pytorch3d point-to-point        | GPU    | 363               | 0.537       | 0.815     | **no** (diverged) |

\* `FastVGICPCuda` does not converge on the **full 69k** cloud from identity
through `pygicp` at any voxel resolution (0.3–3.0 m) — it stays near identity
(trans err ≈ the 0.49 m motion) while the CPU `FastGICP` converges on the same
data. The cause is the dense velodyne ring structure; **voxel-downsampling to
0.3 m (≈5k points) fixes it** — `FastVGICPCuda` then converges to 0.18° and
cuICP, which also stalled on the full cloud, converges to 0.30°. So the honest
statement is "GPU GICP needs the standard LiDAR downsampling to converge here,"
and on the downsampled cloud the accuracy ordering scrambles (downsampling *also*
drops CPU `FastGICP` to 0.51°, since GICP has fewer points for its covariances).
fast_gicp's own C++ `align` benchmark reports the GPU faster, but it times the
solve without checking convergence; I could not run it headless (it opens a PCL
visualizer).

**Honest conclusion.** On real 69k-point LiDAR the strong result is CPU: fast_gicp
`FastGICP` is both fast (505 ms reuse) and the most accurate GICP (0.02°), and
`pcl_gicp` is exact but slow. The GPU point-to-point libraries (cuPCL `cuICP`,
PyTorch3D) under-converge or diverge from identity on the large real motion, and
`FastVGICPCuda` did not converge via its Python binding. Warp converges (0.29°)
and, batched over frames, is the fastest converged method at 126 ms/frame — but I
did **not** find a clean, converged case where fast_gicp's GPU beats its own CPU
`FastGICP` here. GPU ICP is not a free win at this scale; a well-tuned
multithreaded CPU GICP is a genuinely strong baseline.

### Why Warp's rotation error is higher here, and the optimal settings

Warp's 0.29° on KITTI (vs. fast_gicp `FastGICP`'s 0.02°) has two causes, one a
tuning issue and one fundamental:

1. **Defaults are tuned for compact meshes, not sprawling LiDAR.** Warp sizes the
   PCA normal-estimation neighborhood (and the hash-grid cell) from
   `diag / cbrt(N)`. On the bunny that is the point spacing; on a 95 m KITTI scan
   it gives 6.9 m, while the *actual* median nearest-neighbor spacing is 0.012 m —
   a **195× overestimate**. The remedy is preprocessing the way every LiDAR
   pipeline does: **voxel-downsample first** (0.3 m) to regularize velodyne's
   ring structure, and use the **symmetric** variant. That takes Warp to
   **0.225° / 0.034 m** — and is far faster (5k vs. 69k points). So no, the
   defaults are not optimal for KITTI; downsampling + symmetric is.

2. **Point-to-plane vs. GICP is the real gap.** Even with good normals and
   downsampling, Warp trails fast_gicp because it minimizes a *point-to-plane*
   objective (one target normal per correspondence), whereas fast_gicp minimizes
   *generalized ICP* — a plane-to-plane / distribution-to-distribution cost using
   a local covariance on **both** clouds. On anisotropic, noisy velodyne data
   (near-collinear points within a scan ring make single-point normals fragile)
   GICP's two-sided covariance model is markedly more accurate. Closing this gap
   is a feature, not a parameter: a GICP variant for `register_rigid` (reusing
   the per-point covariances it would need anyway) is the natural follow-up.

Net: Warp's point-to-plane is a good general-purpose registrant, but for
production LiDAR odometry a GICP objective (and LiDAR-aware neighborhood sizing)
is what would match the specialized CPU baselines on accuracy.

### Fair same-data comparison (downsampled KITTI, 5k points, one-shot)

Downsampling to 0.3 m is what makes every GPU method converge, so it is also the
fair common ground: the identical ~5k-point clouds, one-shot from identity, each
method using ~KNN-30 normals/covariances (Warp is given normals; the others
estimate their own internally). This is the apples-to-apples LiDAR comparison —
and with proper normals Warp is competitive-to-best:

| method                      | device | time (ms)          | rot err (°) | trans err |
| --------------------------- | ------ | ------------------ | ----------- | --------- |
| open3d point-to-plane       | GPU    | 199                | 0.169       | 0.028     |
| **warp point-to-plane**     | GPU    | 26 / **0.98 batch**| **0.169**   | 0.028     |
| fast_gicp `FastVGICPCuda`   | GPU    | 402                | 0.194       | 0.035     |
| fast_gicp `FastVGICP`       | CPU    | 10                 | 0.214       | 0.034     |
| warp symmetric              | GPU    | 28                 | 0.236       | 0.015     |
| cupcl cuICP                 | GPU    | 7                  | 0.405       | 0.054     |
| pcl_gicp                    | CPU    | 68                 | 0.510       | 0.026     |
| fast_gicp `FastGICP`        | CPU    | 15                 | 0.512       | 0.025     |

Two things flip versus the full-resolution table. First, **Warp's default normal
estimator is the culprit for its earlier poor accuracy** — its auto-radius on a
5k LiDAR cloud is ~16 m; *supplying* KNN-30 normals takes Warp point-to-plane
from 3.8° to **0.169°**, tied with Open3D for the best rotation accuracy here and
by far the fastest when batched (0.98 ms/frame at B=64). Second, **downsampling
reverses the GICP
advantage**: `FastGICP` needs the dense cloud for its covariances, so at 5k it
drops to 0.51°, while point-to-plane with good normals holds up. So on the fair
downsampled data the GPU methods lead, Warp among the best on both accuracy and
(batched) throughput, and the earlier full-resolution GICP win was really "GICP
with 14× more points." The remaining lesson for Warp stands: **fix the
normal-estimation neighborhood sizing for LiDAR** (or let the caller pass
normals, which it already supports) — that alone closes essentially all of the
gap here.

### Being generous to fast_gicp's GPU (reuse mode, their config): where the time goes

Measured in fast_gicp's own regime — their leaf-0.1 downsampling (~16k points),
odometry **reuse** (target voxelmap built once, per-frame `set_input_source` +
`align`), both converged — the GPU is *still* slower end-to-end, but profiling
shows why and vindicates the GPU kernel:

| stage (FastVGICPCuda)          | time    |
| ------------------------------ | ------- |
| `align` (GICP optimization, **GPU**) | **1.7 ms** |
| `set_input_source` (covariance est.) | ~200 ms |
| **per-frame total (reuse)**    | ~200 ms |

vs. **FastGICP CPU reuse: 25 ms/frame** (0.17°) — 8× faster end-to-end, matching
FastVGICPCuda's accuracy (0.13°). The GPU GICP *solve* is blazing (1.7 ms for 16k
points, converged), but per-frame cost is dominated by **covariance estimation**,
which fast_gicp runs on a CPU parallel KD-tree by default (its own source notes
this is usually faster than GPU brute force). Through the `pygicp` binding that
covariance NN method isn't switchable, so the CPU covariance (200 ms) swamps the
GPU solve. fast_gicp's C++ `align` benchmark can select a GPU covariance path and
amortize it, which is where its published "GPU faster" numbers come from; that
knob is simply not exposed to Python. So the fair, generous summary: **fast_gicp's
GPU accelerates the ICP solve to ~2 ms, but its end-to-end per-frame speed is
gated by covariance estimation, and via the Python binding the CPU `FastGICP`
remains the faster and simpler choice at these sizes.**

### FastVGICPCuda's solve vs. Warp's solve (69k KITTI)

The two take opposite architectural bets, so "the solve" means different things.
To be clear: FastVGICPCuda's **`align` is the whole optimization loop** (voxel
correspondence + a 6×6 covariance-weighted solve, iterated to convergence) —
*not* a single linear solve. It is ~1 ms whether or not it converges: ~1.4 ms on
the full 69k cloud where it **stalls** (~1 iteration, roterr 1.05°), and **0.93 ms
on the downsampled cloud where it genuinely converges** (roterr 0.18°) doing all
its iterations. It is that cheap only because the voxel map and per-point
covariances are precomputed in `set_target`/`set_source`, so each iteration is
O(1) voxel lookups + accumulate + a 6×6 solve.

| quantity                                   | FastVGICPCuda | Warp |
| ------------------------------------------ | ------------- | ---- |
| a single 6×6 linear solve                  | — (inside align) | **0.028 ms** (on-device kernel) |
| whole optimization loop (`align`, converged) | **~0.9 ms**  | 519 ms (50 iters) |
| per-iteration cost                         | O(1) voxel lookups + solve | **11.7 ms** (hash-grid NN) + 0.028 ms solve |
| one-time preprocessing                     | voxelmap 255 ms + covariances ~200 ms/frame | none |
| per-frame, odometry (reuse)                | ~202 ms (200 cov + ~1 align) | 519 (single) / **125 (batched)** |

- **FastVGICPCuda front-loads correspondence.** It builds a voxel map and per-point
  covariances up front, so each `align` iteration is O(1) voxel lookups + a small
  linear system — the whole optimization loop is ~1 ms (converged). The price is
  ~455 ms of preprocessing (voxelmap once + covariances per frame).
- **Warp precomputes nothing.** Its on-device 6×6 solve kernel is 0.028 ms — a
  *single* linear solve, so it is not comparable to FastVGICPCuda's ~1 ms whole
  align (which folds in correspondence + many iterations). The honest comparison
  is per-iteration: Warp re-runs a fresh nearest-neighbour search every iteration
  (11.7 ms here, inflated by the large-`max_corr_dist` grid) whereas FastVGICPCuda
  reuses its precomputed voxels (~30 µs/iter), so Warp's full 50-iteration solve
  is 519 ms one-shot.
- **Net:** for streaming odometry with a fixed map, FastVGICPCuda's amortized align
  is the right design; but per *frame* (covariance must be rebuilt each scan) it is
  ~202 ms, which Warp beats when batched (125 ms/frame). The real lesson for Warp:
  its linear solve is not the cost — correspondence is; adopting a precomputed
  voxel/covariance structure (i.e. a GICP variant) would make its per-frame solve
  as cheap as FastVGICPCuda's, on top of Warp's batching.
