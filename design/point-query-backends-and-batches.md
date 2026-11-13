# Direct cuBQL queries and independent point-cloud batches

- **Status:** Initial implementation validated on CPU and one CUDA device; broader performance/portability work remains
- **Issues:** [Direct cuBQL](https://github.com/alecjacobson/warp/issues/20), [fixed-size batches](https://github.com/alecjacobson/warp/issues/21)

## Motivation

Dense point-cloud encoders need differentiable nearest-neighbor distances without a B×N×M pairwise tensor. Targets can stay fixed while decoded clouds move. The existing query-batch API handles only one target cloud. Separately, the retained benchmarks show that native cuBQL queries often outperform Warp traversal for reused indexes, whereas Warp LBVH construction is inexpensive and fully capturable. Both execution paths have useful regimes.

## Design

Extend `point_query_knn_batch` to accept either the existing one-dimensional vec3 arrays or two-dimensional vec3 arrays with shape `(B, M)` for targets and `(B, N)` for queries. The latter represent ordinary coordinate tensors `(B, M, 3)` and `(B, N, 3)`. Return batch-local indices/distances `(B, N, k)`, counts `(B, N)`, and a tuple of B reusable indexes. Existing single-cloud defaults and return shapes remain unchanged.

The first implementation submits one existing query operation per batch using zero-copy views, preserving parent gradient storage. It uses O(BM + BNk) storage for fixed k, with linear construction/order scratch, without a pairwise tensor. It does not yet fuse construction or query launches across batches. Returning ordinary tuples avoids introducing a second aggregate index owner.

`PointQueryBackend.WARP` remains the default. `PointQueryBackend.CUBQL` opts into a native cuBQL tree owned by `CubqlPointIndex`, without conversion into Warp's BVH layout. A supplied index selects its backend unless the caller explicitly requests one; mismatches raise errors. Separate `PointQueryObjective` values select the retained construction-inclusive or reused-index policies. An existing tree keeps its construction objective. The policies are measured heuristics from another GPU slice, not claims of a universal fastest configuration.

The direct route invokes vendored `cuBQL::points::findKNN` with exact-capacity kernels for k=1/11/32 and a bounded 64-slot kernel for other k<=64. It retains upstream ranking arithmetic and tie selection. Strict original-index ties, exclusions, finite cutoffs, and explicit query ordering remain Warp-only; unsupported direct-route requests raise before launch. Forward output distances are computed from selected coordinates; a recorded Warp gather supplies gradients with neighbor indices fixed.

Construction and retirement can synchronize. Direct cuBQL construction/rebuild is rejected before CUDA capture reaches the upstream builder. Reused queries support capture, with outputs and owners retained by the caller. The private cuBQL allocation resource uses stream-ordered CUDA allocations without changing the global pool retention threshold. Warp retains its existing complete construction/refit/query capture route.

For symmetric Chamfer loss, build target indexes once for decoded-to-target queries. Rebuild/refit the changing decoded indexes for target-to-decoded queries. Each direction can choose its own backend: the example defaults to Warp for the changing decoded index, even when the fixed targets use cuBQL. Reduce each direction by its own point count and batch size; empty-cloud loss is undefined and explicitly rejected by the example. Discrete selection is not differentiated. Inputs and selected indices must remain unchanged until backward completes.

## Validation and follow-up

The 85-test point-query/BVH/harness suite passes. New coverage includes overlapping clouds with different local index ordering, singleton equivalence, empty and partial results, output/index reuse, native dispatch, unsupported configurations, ordered capture, refits/rebuilds, and squared/Euclidean gradients of both Chamfer directions against an exhaustive reference and finite differences.

The [pulmonary benchmark report](../reports/knn/batched-cubql/README.md) records 24 workloads and raw timings on the current 94-SM slice, separately from the historical 46-SM report. The supplied archive is read without adding raw data to Git. Keep both issues open for standalone integration-overhead comparisons, wider shape/hardware coverage, peak-memory measurements, and review. Cross-batch launch fusion and denser-query cuBQL tuning remain possible follow-up optimizations; neither is required for the linear-memory batch contract.
