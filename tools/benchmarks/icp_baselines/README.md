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

A GPU-native alternative worth adding is NVIDIA's
[cuPCL](https://github.com/NVIDIA-AI-IOT/cuPCL) (`cuICP`), not yet benchmarked
here.

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

| method                              | objective      | device | batched? | rot err (deg) | throughput (reg/s) | speedup |
| ----------------------------------- | -------------- | ------ | -------- | ------------- | ------------------ | ------- |
| **warp point-to-plane**             | point-to-plane | GPU    | yes      | 0.000         | **252 / 364**      | **1× (ref)** |
| pytorch3d point-to-point            | point-to-point | GPU    | yes      | 2.025         | 165                | 2.2×    |
| fast_gicp `FastGICP`                | plane-to-plane | CPU    | no       | 0.015         | 40                 | 9×      |
| fast_gicp `FastVGICP`               | voxel GICP     | CPU    | no       | 0.491         | 29                 | 12×     |
| pcl_gicp                            | plane-to-plane | CPU    | no       | 0.008         | 8                  | 44×     |
| open3d point-to-plane               | point-to-plane | GPU    | no       | 0.007         | 5                  | 74×     |
| fast_gicp `FastVGICPCuda`           | voxel GICP     | GPU    | no       | 0.000         | 3                  | 110×    |
| pcl point-to-point                  | point-to-point | CPU    | no       | 2.022         | 3                  | 111×    |
| open3d point-to-point               | point-to-point | GPU    | no       | 2.024         | 0.4                | 837×    |

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
