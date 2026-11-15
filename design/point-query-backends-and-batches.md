# Direct cuBQL queries and independent point-cloud batches

- **Status:** Initial scope implemented and validated on CPU and one CUDA device; wider tuning and application integration are follow-up work
- **API/main revision:** [#22](https://github.com/alecjacobson/warp/issues/22), rebased onto NVIDIA main `3afdd69addbf` on November 15
- **Issues:** [Direct cuBQL](https://github.com/alecjacobson/warp/issues/20), [fixed-size batches](https://github.com/alecjacobson/warp/issues/21)

## Motivation

Dense point-cloud encoders need differentiable nearest-neighbor distances without a B×N×M pairwise tensor. Targets can stay fixed while decoded clouds move. The existing query-batch API handles only one target cloud. Separately, the retained benchmarks show that native cuBQL queries often outperform Warp traversal for reused indexes, whereas Warp LBVH construction is inexpensive and fully capturable. Both execution paths have useful regimes.

## Design

The kernel layer uses the existing `wp.Bvh(points, points)` object, treating points as zero-volume boxes. `wp.point_query_nearest`, `wp.point_query_knn`, and `while wp.point_query_next(...)` support caller-defined fused processing. No second scalar point-cloud handle type is needed.

The array layer lives in the explicitly imported `warp.geometry` module. `geometry.PointIndex(targets, num_queries=N, k=k, backend=...)` owns one or B target indexes without query coordinates or neighbor outputs. `geometry.knn(index, queries, k)` reuses that owner; `geometry.knn(targets, queries, k)` constructs one. Every geometry call returns `(PointIndex, indices, distances, counts)`, including single clouds and empty batches. Retaining the owner makes asynchronous resource lifetime explicit. Target storage is retained by reference; coordinate updates require `refit()` or `rebuild()`, and allocation/shape replacement requires a new owner.

Inputs are one-dimensional vec3 arrays or two-dimensional vec3 arrays `(B, M)` / `(B, N)`, representing coordinate tensors `(B, M, 3)` / `(B, N, 3)`. Batched outputs are local IDs/distances `(B, N, k)` and counts `(B, N)`. The implementation submits an existing query operation per batch using zero-copy views that preserve parent gradient storage. Storage is O(BM + BNk) for fixed k, with linear construction/order scratch. Construction and query launches are not yet fused across batches.

`geometry.PointQueryBackend.WARP` is the default. Its owner exposes a tuple of ordinary Warp BVHs through `.bvhs` for kernel queries, without an ambiguous aggregate `.id`. `geometry.PointQueryBackend.CUBQL` selects the existing optimized native implementation with private, incompatible handles; `.bvhs` and `.refit()` raise for that backend. Rebuild is supported for both. `PointQueryObjective` selects retained construction-inclusive or reused-index cuBQL policies; it does not switch backends. Existing owners reject conflicting backend/objective and leaf-size overrides. Tuning hints may differ from subsequent query counts and k.

The former `wp.knn`, `wp.build_point_index`, `wp.point_query_config`, configuration types, and `wp.CubqlPointIndex` remain available with deprecation warnings. Their signatures and raw index/tuple returns are preserved. `wp.point_query_knn_batch` also remains functional. These compatibility entry points delegate to the same engine now housed under `warp._src.geometry`; new public examples use the geometry API. Core scalar query signatures and algorithms are unchanged.

The direct route invokes vendored `cuBQL::points::findKNN` with exact-capacity kernels for k=1/11/32 and a bounded 64-slot kernel for other k<=64. It retains upstream ranking arithmetic and tie selection. Strict original-index ties, exclusions, finite cutoffs, and explicit query ordering remain Warp-only; unsupported direct-route requests raise before launch. Forward output distances are computed from selected coordinates; a recorded Warp gather supplies gradients with neighbor indices fixed.

Construction and retirement can synchronize. Direct cuBQL construction/rebuild is rejected before CUDA capture reaches the upstream builder. Reused queries support capture, with outputs and owners retained by the caller. The private cuBQL allocation resource uses stream-ordered CUDA allocations without changing the global pool retention threshold. Warp retains its existing complete construction/refit/query capture route.

For symmetric Chamfer loss, build target indexes once for decoded-to-target queries. Rebuild/refit the changing decoded indexes for target-to-decoded queries. Each direction can choose its own backend: the example defaults to Warp for the changing decoded index, even when the fixed targets use cuBQL. Reduce each direction by its own point count and batch size; empty-cloud loss is undefined and explicitly rejected by the example. Discrete selection is not differentiated. Inputs and selected indices must remain unchanged until backward completes.

## Validation and follow-up

The November 15 API revision passes the 86-test CPU/CUDA point-query and BVH suite. Coverage includes overlapping clouds with different local index ordering, singleton equivalence, empty and partial results, output/index reuse, native dispatch, unsupported configurations, ordered capture, refits/rebuilds, and squared/Euclidean gradients of both Chamfer directions against an exhaustive reference and finite differences. Build-only tests forbid query dispatch during construction, verify reuse without rebuilding, and exercise changed query counts and k. Legacy calls retain their signature and behavior with a deprecation warning.

The [API/main regression report](../reports/knn/api-revision/README.md) measures the new public API against the published pre-rebase branch and the retained engine on new main. The [current comparison report](../reports/knn/comparisons.md) preserves November 13 measurements for 24 workloads, construction/query timings, retained memory, and symmetric Chamfer forward/backward on the current 94-SM slice, separately from the historical 46-SM report. The [completion report](../reports/knn/completion/README.md) compares integrated and standalone cuBQL on matching inputs and settings, and records final API and conditional-build validation. The supplied archive is read without adding raw data to Git.

Wider shape/hardware coverage, peak training memory, actual encoder integration, cross-batch launch fusion, and denser-query tuning remain follow-up work. They do not block the initial optional native route or linear-memory fixed-size batch contract. Ragged batches are outside this design.
