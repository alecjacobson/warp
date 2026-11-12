# Warp-only batch kNN and fast scalar use

The supported batch wrapper now applies the measured query policy automatically, using only Warp Python, arrays, and kernels. The existing native BVH and scalar selection primitives are unchanged. There is no Torch dependency, new point-cloud owner, or public query-result record.

```python
bvh, indices, distances_sq, counts = wp.point_query_knn_batch(points, queries, 8)

# Reuse the tree; refit it first if points moved in place.
_, indices2, distances_sq2, counts2 = wp.point_query_knn_batch(points, next_queries, 8, bvh=bvh)
```

The ordinary tuple contains the existing BVH and plain output arrays. The wrapper supports preallocated outputs, int32/int64 indices, squared or Euclidean distances, per-query exclusion indices, strict distance cutoffs, and partial results padded with -1/infinity. It preserves original query order. Keep the BVH, inputs and outputs alive during asynchronous work or graph replay.

`wp.point_query_knn()` remains the direct scalar primitive used by the batch kernel. Callers can consume neighbors inside their own computation without materializing an entire `(Q, k)` output. `wp.point_query_config(N, Q, device)` exposes the same leaf size, query stride, block size and ordering recommendation used by the batch wrapper. The [fused example](../../../warp/examples/geometry/example_point_query_batch.py) uses its stride mapping with the familiar `point_query_knn` / `point_query_next` loop. Global `wp.Bvh` and `wp.launch` defaults are unchanged: the scalar primitive cannot rearrange other threads in a caller's kernel.

[API guide](../../../docs/user_guide/point_queries.rst) · [Design](../../../design/point-query-batch.md) · [Implementation](../../../warp/_src/point_query.py) · [Tests](../../../warp/tests/geometry/test_point_query_batch.py)

## Performance validation

We remeasure the public batch wrapper, a manually configured scalar wrapper, and the dense leaf-size-4 scalar baseline in a **Warp-only process**. Each process records `torch_imported: false`. All return int32 indices and float32 squared distances, with the same padded-output contract. This is an internal integration comparison, not a replacement for the earlier int64/Euclidean cross-library matrix.

The 40 pilot cases are the frozen scan/surface/volume/noise cases. Eight additional cases use N=1M, Q=100,000, k=11 with two unseen seeds, surfaces/volumes, and clean/noisy data. These dense cases exercise the automatic query-ordering path. There are nine synchronized wall samples after two warmups per phase, with JIT/upload excluded, one GPU job at a time, and 512 MiB Warp memory-pool retention on the same 46-SM Blackwell MIG slice. The API does not set that global retention threshold itself.

| Suite | Cases | Dense scalar / batch, build + query | Dense scalar / batch, reused index | Batch / configured scalar, build + query | Batch / configured scalar, reused index |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pilot | 40 | 2.62x | 3.85x | 1.012x | 1.030x |
| Dense large-query holdouts | 8 | 1.41x | 1.48x | 1.026x | 1.036x |

Values are geometric means. The batch API beats the dense baseline in all 48 cases in both timing modes. Compared with manually configured scalar calls, the convenience/validation overhead is about **1.2% including construction and 3.0% with reuse** on the pilot matrix. Two pilot build-plus-query cases and seven reused-index cases are more than 5% slower; the largest pilot slowdown is about 7.5%. All eight dense build-plus-query cases are within 5%; one dense reused-index case is about 8.0% slower. These are observed timings, not a promise of zero wrapper overhead or statistical significance.

The large-query path computes Morton bounds from queries using Warp kernels, rather than reading the native BVH root through a native snippet. That adds bounds kernels and can change scheduling order, while preserving exact selection and output row order. Custom scalar fusion remains available for callers who want to avoid generic output allocation/validation or combine neighbor processing with downstream work.

[Raw pilot results](pilot.json) · [Raw dense results](dense.json) · [Summary](summary.json) · [Per-case CSV](cases.csv) · [Final source hashes](source-sha256.json)

## Correctness, gradients, and capture

- **42 tests pass**: eight new batch CPU/CUDA tests, 26 existing scalar point-query tests, and eight benchmark-oracle/data tests. Coverage includes empty/partial results, ties, ordering, strides 1/4/32, cutoffs, exclusions, output/BVH reuse, refits, input validation and gradients. [Test log](tests-final.log)
- All 144 measured method/case pairs pass recorded one-shot and reused-index structural and sampled exhaustive-oracle checks. The oracle checks 32 evenly spaced query rows; it does not exhaustively check every large-batch row.
- New tests capture complete construction plus query and refit plus query, with ordered and unordered launches. They change inputs and validate replay. Temporary sorting arrays are freed through Warp's graph allocation dependencies.
- A separate [external nonblocking-stream stress probe](probe_external_stream.py) passes repeated ordered and unordered calls. Outside capture, the wrapper inserts a device-side dependency before default-stream retirement on external nonblocking streams; it introduces no host synchronization. [Probe log](external-stream.log)
- Forward-only calls fuse selection and output distances. When input gradients are requested, a separate differentiable gather uses the saved indices; selection and ordering remain nondifferentiable. Existing native scalar fused-loss gradient tests still pass. Do not overwrite inputs or selected indices between forward and backward.
- The fused example matches the batch's summed neighbor distances. [Example log](example.log)
- Documentation builds without warnings and lint passes. No native rebuild was necessary because no native code changed. [Docs log](docs.log.gz), [lint log](lint-final.log)

## Reproduce

From the repository root, use the existing Warp environment and native build. No comparison venv or Torch import is needed:

```sh
uv run --no-sync -m unittest warp.tests.geometry.test_point_query_batch warp.tests.geometry.test_point_query warp.examples.benchmarks.test_point_query_comparison -v
uv run --no-sync -m warp.examples.geometry.example_point_query_batch --device cuda:0
uv run --no-sync -m warp.examples.benchmarks.benchmark_point_query_batch \
  --manifest ../data/knn-comparison/pilot/manifest.json \
  --output ../artifacts/batch-api/pilot.json
uv run --no-sync -m warp.examples.benchmarks.benchmark_point_query_batch \
  --manifest ../data/knn-comparison/hybrid-holdout/manifest.json --filter q100000 \
  --output ../artifacts/batch-api/dense.json
uv run --no-sync reports/knn/batch/probe_external_stream.py
uv run --no-sync build_docs.py
```

The external-stream probe uses the Linux CUDA driver library through Python's standard ctypes module. Dataset preparation and source terms remain in the [broad comparison](../comparison/reproduce.md) and [holdout instructions](../hybrid/reproduce.md). Arrays remain outside Git. Copy fresh JSONs into this directory and run `uv run --no-sync reports/knn/batch/summarize.py` to regenerate tables. Timing JSONs name parent commit `e4b7d96` because measurements preceded this implementation commit; their source hashes identify the measured changes.
