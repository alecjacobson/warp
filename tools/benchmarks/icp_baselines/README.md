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
#    fast_gicp: build koide3/fast_gicp's python binding, then run with its
#    libfast_gicp.so on LD_LIBRARY_PATH; PyTorch3D: build against a matching torch.

# 4. PCL (C++): build the binary, then score it.
cmake -S tools/benchmarks/icp_baselines/pcl -B /tmp/pcl_build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/pcl_build -j
uv run --with scipy tools/benchmarks/icp_baselines/run_pcl.py /tmp/pcl_build/pcl_icp

# 5. Print the merged table.
uv run tools/benchmarks/icp_baselines/report.py
```

Baseline versions used below: Open3D 0.19 (PyPI wheel), PCL 1.12, fast_gicp
(built from source), PyTorch3D 0.7.8 (built from source against torch 2.1 CPU).

## Results — accuracy and throughput

10,242-point problem, 50-iteration budget; Warp on an NVIDIA L40, every baseline
on CPU. "throughput" is registrations/second: the single-problem CPU libraries
have no batch API, so N registrations cost N× the latency (their `reg/s` is
`1000 / latency`), whereas Warp registers a whole batch in one call, so its
throughput is the per-problem time at batch saturation (B ≥ 64). Warp figures are
given as mesh / point-cloud target.

| method                              | objective      | rot err (deg) | latency (ms)      | throughput (reg/s) |
| ----------------------------------- | -------------- | ------------- | ----------------- | ------------------ |
| **warp point-to-plane — batched**   | point-to-plane | 0.000         | 4.0 / 2.7 per prob| **252 / 364**      |
| warp symmetric — batched\*          | symmetric      | 0.000         | —                 | —                  |
| warp point-to-plane — single        | point-to-plane | 0.000         | 21 / 31           | 47 / 32            |
| fast_gicp `FastGICP`                | plane-to-plane | 0.015         | 25                | 40                 |
| fast_gicp `FastVGICP`               | voxel GICP     | 0.491         | 34                | 29                 |
| pcl_gicp                            | plane-to-plane | 0.008         | 121               | 8                  |
| open3d point-to-plane               | point-to-plane | 0.007         | 202               | 5                  |
| pcl point-to-point                  | point-to-point | 2.022         | 306               | 3                  |
| open3d point-to-point               | point-to-point | 2.024         | 2299              | 0.4                |
| pytorch3d point-to-point            | point-to-point | 2.025         | 14828             | 0.07               |

\* Batched registration is point-to-plane only; the symmetric variant is a
single-problem option (same ~0.000° accuracy, ~22 ms).

**The batched win:** Warp registers **250–360 clouds/s vs. the best CPU
baseline's 40/s** — a ~6× (mesh) to ~9× (point-cloud) throughput advantage —
because one 10k-point problem underfills the L40 while a batch saturates it (see
below). Single-problem, Warp is competitive but not dominant (47/s mesh, 32/s
cloud vs. fast_gicp's 40/s); the GPU pulls away with scale and batch.

## Reading the table honestly

- **Device is not held constant.** Warp runs on the L40 GPU; PCL, fast_gicp, and
  the PyTorch3D/Open3D wheels here run on CPU. The timings therefore reflect
  Warp's GPU advantage as much as the algorithm — this compares the tools as one
  would actually install and run them, not the kernels head-to-head. PyTorch3D
  and Open3D's tensor ICP would be far faster on a CUDA build; PyTorch3D's
  point-to-point in particular is slow on CPU (KD-tree-free knn each iteration).
- **The objectives differ.** Point-to-plane (Warp, Open3D) and plane-to-plane
  GICP (PCL, fast_gicp) both slide points along the surface and converge to a
  tight solution; point-to-point (PyTorch3D, and the PCL/Open3D point-to-point
  variants) minimizes a weaker metric and lands ~2° off on this noisy problem.
  Compare like-for-like: Warp point-to-plane vs. Open3D point-to-plane, and note
  GICP as a strong plane-to-plane reference.
- **Takeaway.** Warp's point-to-plane and symmetric variants are the most
  accurate here and the fastest of the point-to-plane / GICP group, while
  fast_gicp's `FastGICP` is a strong CPU baseline (25 ms, 0.015°). The wide
  quality gap to the point-to-point methods matches the ablation in
  `../README.md`.

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
