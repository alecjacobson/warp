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

## Results (NVIDIA L40 for Warp; CPU for the rest)

| method                     | objective       | rot err (deg) | trans err | time (ms) | device |
| -------------------------- | --------------- | ------------- | --------- | --------- | ------ |
| warp_symmetric_cloud       | symmetric       | 0.000         | 1e-4      | 43        | L40 GPU |
| warp_point_to_plane_mesh   | point-to-plane  | 0.000         | 7e-5      | 30        | L40 GPU |
| open3d_point_to_plane      | point-to-plane  | 0.007         | 7e-5      | 202       | CPU    |
| pcl_gicp                   | plane-to-plane  | 0.008         | 6e-5      | 121       | CPU    |
| fast_gicp_FastGICP         | plane-to-plane  | 0.015         | 7e-5      | 25        | CPU 8-thread |
| warp_point_to_plane_cloud  | point-to-plane  | 0.020         | 7e-5      | 41        | L40 GPU |
| fast_gicp_FastVGICP        | voxel GICP      | 0.491         | 1e-3      | 34        | CPU 8-thread |
| pcl_point_to_point         | point-to-point  | 2.022         | 2e-3      | 306       | CPU    |
| open3d_point_to_point      | point-to-point  | 2.024         | 2e-3      | 2299      | CPU    |
| pytorch3d_point_to_point   | point-to-point  | 2.025         | 2e-3      | 14828     | CPU    |

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

A single 10k-point registration barely exercises a GPU, and profiling the Warp
path (L40) shows exactly where its time goes:

- **It is latency-bound, not throughput-bound.** Building the target BVH is
  0.4 ms (negligible); a mesh solve is ~0.7 ms/iter and converges in ~6–23
  iterations, so the whole thing is ~18–30 ms — dominated by per-iteration kernel
  launches and the Python-driven loop, not arithmetic. The GPU sits far below
  saturation. **Batching amortizes that fixed cost: per-problem time drops from
  ~21 ms at B=1 to ~4 ms at B=64** (5× cheaper), which is where the GPU pulls
  clearly ahead. A mature multithreaded C++ library like fast_gicp is genuinely
  hard to beat on one small problem, and Warp's *mesh* path (~18–30 ms) already
  matches or beats it — it is mainly the point-cloud variants that trail.
- **The point-cloud path is charged for `max_corr_dist`.** For a point-cloud
  target that distance sizes the hash-grid used for nearest-neighbor search, so
  the query cost grows like `(max_corr_dist / point_spacing)³`. The benchmark's
  generous `max_corr_dist=0.3` costs ~2.6 ms/iter; a sensible bound (~0.05, a few
  × the spacing) drops it to ~1.0 ms/iter, matching the mesh path. The KD-tree in
  the CPU baselines finds the nearest neighbor in `O(log N)` regardless of the
  bound, so a large bound penalizes only the grid — a real asymmetry to keep in
  mind (and the mechanism behind the ablation's "`max_corr_dist` past the overlap
  only costs time"). For a mesh target the BVH is insensitive to the bound.
- **Where the GPU wins:** point count (throughput scales up while a CPU falls
  behind) and batch (many simultaneous registrations, e.g. the multi-start
  example). Neither is exercised by one small single-problem benchmark, so read
  these numbers as "competitive out of the box at small scale," not the ceiling.
