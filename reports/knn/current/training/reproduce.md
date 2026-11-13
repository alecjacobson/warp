# Reproduce the batched query and fixed-target Chamfer comparison

This experiment measures native batch dimensions through public Warp, PyTorch, PyTorch3D, and KeOps APIs. External library source is unchanged. The benchmark adapters compose public calls and loss reductions; they contain no replacement CUDA kernels, modified library internals, batch flattening, or Python loops over external batches.

## Configurations and supported tuning

| Configuration | Batched query | Symmetric Chamfer and backward |
| --- | --- | --- |
| `backend=wp.PointQueryBackend.WARP` | Prebuilt target indexes, `wp.knn()` | Prebuilt fixed target indexes; build changing reverse indexes each step; Warp Tape backward. |
| `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE` | Prebuilt target indexes, `wp.knn()` | Fixed targets use those settings; reverse indexes use `backend=wp.PointQueryBackend.WARP`. |
| `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.BUILD_AND_QUERY` | Prebuilt target indexes with this policy | The Chamfer column explicitly uses `objective=wp.PointQueryObjective.REUSE` for fixed targets and `objective=wp.PointQueryObjective.BUILD_AND_QUERY` for changing reverse targets. |
| PyTorch matrix multiplication | `torch.cdist(compute_mode="use_mm_for_euclid_dist")`, then `min` for k=1 or sorted `topk` | Compute one distance matrix, take minima in both directions, square the selected distances, average, and call `backward()`. |
| PyTorch direct distances | Same calls, `compute_mode="donot_use_mm_for_euclid_dist"` | Same loss and native autograd, using the direct distance mode. |
| PyTorch selected-pair autograd | Same search as direct distances; no duplicate query column | Search in direct mode under `torch.no_grad()`, gather the nearest neighbors in both directions, then differentiate their squared distances. This avoids the dense `cdist` backward graph; the forward distance matrix still occupies B×N×M storage. |
| PyTorch3D automatic | `knn_points(K=k, version=-1)` | Native `chamfer_distance(norm=2, batch_reduction="mean", point_reduction="mean")`, then `backward()`. |
| PyTorch3D tuned | Public `knn_points(version=...)` | Two public `knn_points(K=1, version=...)` calls, mean squared distances in each direction, then `backward()`. |
| KeOps | Batched `LazyTensor` squared distances and `Kmin_argKmin` | Public batched `argmin` in both directions under `torch.no_grad()`; gather selected neighbors and differentiate squared distances with PyTorch. |

PyTorch's documented `cdist` modes select either matrix multiplication or direct Euclidean distances. All measured cloud sizes trigger matrix multiplication in its default heuristic, so the explicit matrix-multiplication column also covers that default algorithm. TF32 is disabled and coordinates remain float32. See the [pinned PyTorch implementation](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/functional.py).

PyTorch3D exposes `version` through its public [`knn_points()` API](https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/knn.html). We time every supported version for D=3 (0–3 for k=1; 0–2 for k=11/32), using five samples on independent synthetic clouds. Selection uses batch size, query-count regime, k, and direction; it never chooses a winner from evaluation timings. The resulting [tuning record](tuning.json) retains all candidates. This is the best setting in that measured search, not a claim of optimality across versions, hardware, compilers, and workloads. The stock [`chamfer_distance()` API](https://pytorch3d.readthedocs.io/en/latest/modules/loss.html) remains a separate column because it does not expose the version override.

KeOps supports batch dimensions directly in [`LazyTensor`](https://www.kernel-operations.io/keops/python/api/pytorch/LazyTensor.html). Its integer neighbor selections are treated as fixed during backward, as is conventional for piecewise differentiable nearest-neighbor Chamfer. Only ordinary calls are measured here. There is no graph-replay or complete neural-network training claim.

## Data, timing, and checks

The dataset follows the existing 24-workload pulmonary/synthetic generator: nine original pulmonary clouds with N=1,024/8,192, two repeated-largest-cloud batches B=2/4, a four-cloud non-ragged subset with M=65,536, and three synthetic k>1 controls. Chamfer uses the 21 k=1 cases. [manifest.json](manifest.json) records original archive and derived coordinate/file hashes. B=1 rows are kept explicit; the three B>1 rows appear first in the colored tables.

All inputs are contiguous float32 `(B, points, 3)` arrays. Query output is sorted int64 IDs and float32 squared distances, including required allocation/conversion. Fixed target construction, data transfer, compilation, and two warmups are outside timing. Each median uses nine synchronized wall-clock samples on the same Warp-owned CUDA stream borrowed by Torch. Ordinary calls include output allocation and previous result retirement. Chamfer includes both query directions, construction of changing reverse indexes where applicable, loss reduction, gradient reset, and backward with respect to decoded points only. Warp uses its native Tape; external providers use PyTorch autograd. No Torch-to-Warp autograd bridge is included.

The loss is `mean(decoded_to_target_squared_distance) + mean(target_to_decoded_squared_distance)`, averaged over points and batches. Native `cdist` uses one matrix for both directions. Public PyTorch3D and KeOps calls retain their native batching. Warp retains 512 MiB in its CUDA pool; Torch uses its native caching allocator. Torch's cache is released between workloads, outside warmup/timing, to avoid carrying large pairwise allocations into the next case.

Reference neighbor IDs come from the exact default Warp route, with 32 sampled exhaustive float64 rank/distance checks per batch in both directions. Full loss and decoded gradients are then independently reconstructed in NumPy float64 from those IDs. Two CPU tests check this oracle against central differences of the exhaustive Chamfer loss and check batch isolation. Large-case gradients are compared in full, using relative L2 error ≤5e-4 and maximum absolute error ≤`1e-8 + 5e-4 * max(abs(reference_gradient))`; loss uses `rtol=2e-4`, `atol=1e-7`. Query tolerances match the earlier cross-library report. This is sampled neighbor validation plus complete selected-neighbor gradient comparison, not exhaustive verification of every large-case neighbor.

Failures remain evidence: numerical failures retain their medians but receive gray cells; runtime, allocator, and launch failures receive no speed ratio. A worker exits after a runtime/launch error; an explicit `OutOfMemoryError` is recorded per case. Missing cases are retried individually in fresh processes, without changing the provider or replacing its algorithm. PyTorch can fail while collecting NVML diagnostics for an allocation error on this MIG container; these cells are labeled `allocator_nvml_error`, separately from explicit `out_of_memory` and `launch_error` results. Source snapshots preserve the initial harness and the update that added per-case error records/restarts; timing and provider call implementations are unchanged by that update.

The pinned direct-distance [CUDA implementation](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cuda/DistanceKernel.cu) launches one block per pair using `grid(result.numel())`; very large B×N×M cases hit an invalid-launch-configuration error in this environment. The pinned [allocator implementation](https://github.com/pytorch/pytorch/blob/v2.7.1/c10/cuda/CUDACachingAllocator.cpp) calls NVML process-memory reporting on its allocation-failure path. Its `NVML_SUCCESS == r` assertion matches the recorded allocator failures. We retain that diagnostic rather than relabeling it as an ordinary `OutOfMemoryError`.

## Run and regenerate

Reuse the project's existing `.venv`, CUDA 12.9.1 toolkit, and unchanged PyTorch3D build under `.benchmarks/knn/third_party/pytorch3d`. Dependencies and source revision are those in the [external-library environment](../libraries/reproduce.md). This experiment uses Torch 2.7.1+cu128, the pinned PyTorch3D source build, and KeOps 2.3. It does not upgrade or patch them.

```bash
uv run --no-sync -m warp.examples.benchmarks.point_query_comparison_environment \
  --python .venv/bin/python --cuda /home/horde/cuda/12.9.1 \
  --module warp.examples.benchmarks.benchmark_point_query_training -- \
  prepare --data .benchmarks/knn/training/data \
  --pytorch3d-path .benchmarks/knn/third_party/pytorch3d

uv run --no-sync -m warp.examples.benchmarks.point_query_comparison_environment \
  --python .venv/bin/python --cuda /home/horde/cuda/12.9.1 \
  --module warp.examples.benchmarks.benchmark_point_query_training -- \
  tune --data .benchmarks/knn/training/data \
  --output .benchmarks/knn/training/tuning.json \
  --pytorch3d-path .benchmarks/knn/third_party/pytorch3d

uv run --no-sync -m unittest warp.examples.benchmarks.test_point_query_training
uv run --no-sync python reports/knn/current/training/run.py --smoke
uv run --no-sync python reports/knn/current/training/run.py
uv run --no-sync python reports/knn/current/training/run.py --complete
```

Keep original attempts and process receipts. The collector verifies all workload records, nine-sample medians, source snapshots, input hashes, and unchanged runtime/dependency evidence. It copies raw evidence into `results/`, compresses logs, and records [provenance](provenance.json). Then regenerate both charts, [all medians/statuses](cases.csv), [aggregate tables](tables.md), and [summary](summary.json):

```bash
uv run --no-sync python reports/knn/current/training/collect.py
uv run --no-sync python reports/knn/current/training/summarize.py
```

The supplied pulmonary ZIP and generated input/reference arrays remain outside Git. The previous 24-workload int32 query/graph experiment and 40-workload int64 Euclidean single-cloud comparison retain their own raw evidence; their timings are not pooled into these new tables.
