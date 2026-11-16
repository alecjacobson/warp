# Warp kNN: kernel queries and reusable array indexes

The branch builds on **`wp.Bvh` and scalar kernel queries**. The optional **`warp.geometry`** module supplies array queries and a concrete `PointIndex` owner for single clouds and fixed-size batches. The native cuBQL optimization remains available through that array API. See [comparisons](comparisons.md) for measurements and the [API revision report](api-revision/README.md) for validation after the November 15 rebase onto NVIDIA `main` at `3afdd69addbf`.

## Kernel queries come first

Index points as zero-volume boxes using the existing BVH object:

```python
import warp as wp


@wp.kernel
def neighbor_sum_kernel(
    bvh_id: wp.uint64,
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    sums: wp.array[float],
):
    row = wp.tid()
    search = wp.point_query_knn(bvh_id, points, queries[row], wp.inf, 8)
    index = int(-1)
    total = float(0.0)
    while wp.point_query_next(search, index):
        total += wp.length_sq(queries[row] - points[index])
    sums[row] = total


points = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=wp.vec3)
queries = wp.array([[0.9, 0, 0]], dtype=wp.vec3, device=points.device)
bvh = wp.Bvh(lowers=points, uppers=points)
sums = wp.empty(len(queries), dtype=float, device=points.device)
wp.launch(neighbor_sum_kernel, len(queries), [bvh.id, points, queries, sums], device=points.device)
```

For one neighbor, replace selection/enumeration with:

```python
index = wp.point_query_nearest(bvh_id, points, queries[row], wp.inf)
if index >= 0:
    distance_sq = wp.length_sq(queries[row] - points[index])
```

Scalar k is a positive compile-time integer. Selection and sorting finish before enumeration; breaking the loop does not reduce search work. The cursor yields only valid neighbors, ordered by squared distance and then original target index. It stops early when fewer than k points qualify. Use `while wp.point_query_next(...)`; direct `for index in search` iteration is not implemented.

Both selectors support a strict distance cutoff and optional `exclude_index` (-1 excludes nothing). Nonfinite queries and nonpositive/NaN cutoffs return no neighbors. Guard the -1 result from `point_query_nearest`. Selection is nondifferentiable; gathered coordinates and subsequent distances support gradients with selected IDs fixed. Keep inputs and the BVH unchanged until backward completes.

## Build fixed targets once

```python
import warp as wp
import warp.geometry

# target_numpy: float32 (B, M, 3); decoded_numpy: float32 (B, N, 3).
targets = wp.array(target_numpy, dtype=wp.vec3, device="cuda:0")
decoded = wp.array(decoded_numpy, dtype=wp.vec3, device=targets.device, requires_grad=True)

target_index = wp.geometry.PointIndex(
    targets,
    num_queries=decoded.shape[1],
    k=1,
    backend=wp.geometry.PointQueryBackend.CUBQL,
    objective=wp.geometry.PointQueryObjective.REUSE,
)
_, indices, distances_sq, counts = wp.geometry.knn(target_index, decoded, 1)
```

Construction needs no query coordinates or neighbor outputs. `num_queries` and `k` are tuning hints; later calls may use different counts. The owner retains target coordinates by reference. Updating decoded coordinates requires another query, without rebuilding fixed targets.

For construction plus query in one call:

```python
target_index, indices, distances_sq, counts = wp.geometry.knn(targets, decoded, 1)
```

**Every call returns `(PointIndex, indices, distances, counts)`**, with the same owner returned when supplied. Supplying a target array constructs a new owner. Keeping the owner in the result also makes its asynchronous lifetime explicit.

## Shapes and ownership

| Value | Single cloud | Independent batches |
| --- | --- | --- |
| Coordinate tensor | `(M, 3)` targets, `(N, 3)` queries | `(B, M, 3)` targets, `(B, N, 3)` queries |
| Warp `wp.vec3` array | `(M,)`, `(N,)` | `(B, M)`, `(B, N)` |
| Indices and distances | `(N, k)` | `(B, N, k)` |
| Counts | `(N,)` | `(B, N)` |
| Target owner | `wp.geometry.PointIndex` | `wp.geometry.PointIndex` |

Inputs are contiguous float32 arrays on the same device. Targets must be finite, with queried squared distances representable in float32. Each batch searches only its own targets; IDs are local to that cloud. B=1 preserves the leading dimension. Empty clouds and B=0 work; missing neighbors have index -1 and infinite distance, and counts report valid neighbors. Nonfinite queries return zero counts and padded results. Ragged batches and cross-batch broadcasting are unsupported.

`target_index.points` exposes the retained target allocation. With the default backend, `target_index.bvhs` is a tuple of ordinary Warp BVHs, one per cloud, available for caller kernels. A single cloud has one entry. The owner has no ambiguous aggregate `id`. Native cuBQL owners reject `.bvhs` access: their private handles cannot be passed to scalar Warp BVH queries.

After changing target coordinates in place, call `target_index.refit()` or `.rebuild()` for the Warp backend; call `.rebuild()` for native cuBQL. These methods update all batches. Replacing the allocation or changing point counts/batch layout requires a new owner. Native cuBQL rebuilding replaces handles: finish outstanding readers and recapture query graphs afterward.

## Which settings should I use?

Use `import warp as wp` and explicitly `import warp.geometry` to access the array API as `wp.geometry`.

| Workload or requirement | Public settings |
| --- | --- |
| Frequent construction, CPU execution, or a portable default | `backend=wp.geometry.PointQueryBackend.WARP` |
| Repeated queries against fixed CUDA targets | `backend=wp.geometry.PointQueryBackend.CUBQL`, `objective=wp.geometry.PointQueryObjective.REUSE` |
| Evaluate cheaper native cuBQL construction | `backend=wp.geometry.PointQueryBackend.CUBQL`, `objective=wp.geometry.PointQueryObjective.BUILD_AND_QUERY` |
| Capture construction/refit/rebuild, finite cutoff, exclusion, original-index ties, or k > 64 | `backend=wp.geometry.PointQueryBackend.WARP` |
| Capture queries against prebuilt indexes | Both backends; keep owners and allocations alive |
| Symmetric fixed-target Chamfer | Fixed: `backend=wp.geometry.PointQueryBackend.CUBQL`, `objective=wp.geometry.PointQueryObjective.REUSE`; changing reverse targets: `backend=wp.geometry.PointQueryBackend.WARP` |
| Fuse neighbor processing in a kernel | `wp.Bvh` plus scalar `wp.point_query_nearest()` / `wp.point_query_knn()` |

The backend is explicit. Objectives tune cuBQL; they do not switch backends or change Warp's policy. Existing owners reject conflicting backend/objective overrides and any non-None leaf-size override. Native cuBQL requires CUDA, memory pools, cuBQL-enabled builds, k=1-64, positive infinite cutoff, no exclusion, and `ordered=None` or `False`. It rejects `ordered=True` and uses upstream tie selection. Construction and rebuilding occur outside capture and can synchronize.

`wp.Bvh(constructor="cubql")` is a separate choice: it converts a cuBQL-built tree into Warp's BVH layout and uses Warp traversal. The native `backend=wp.geometry.PointQueryBackend.CUBQL` route retains its own handles. A raw `wp.Bvh` is not a `PointIndex` input to `wp.geometry.knn()`; query it inside a kernel or retain the legacy `wp.knn(..., bvh=...)` call while migrating. The historical converted-tree benchmark column therefore has no direct `wp.geometry.knn(bvh=...)` equivalent.

Leave `leaf_size`, `query_stride`, and `ordered` automatic initially. `wp.geometry.point_query_config(M, N, device)` exposes Warp's tuning policy for custom kernels; applying its launch spacing requires matching thread-to-query indexing. The default array route already uses the scalar primitive with tuned scheduling and tree choices. The native cuBQL route retains its separate optimized implementation.

## Outputs, capture, and gradients

Indices default to int32; caller-supplied int64 outputs are supported. Counts are int32 and distances float32, squared by default. Use `squared=False` for Euclidean distances. `max_dist` always uses coordinate units. Warp-only `exclude_indices` is int32 with the query shape, containing IDs to omit or -1. Optional `indices`, `distances`, and `counts` buffers must have matching shapes and cannot overlap inputs or one another.

After warmup, capture with retained outputs:

```python
with wp.ScopedCapture(device=targets.device) as capture:
    wp.geometry.knn(
        target_index,
        decoded,
        1,
        indices=indices,
        distances=distances_sq,
        counts=counts,
    )
decoded.assign(next_decoded_numpy)
wp.capture_launch(capture.graph)
```

Keep the owner, inputs, and outputs alive through replay. Changes to shape or allocation require new outputs and a new capture. Native cuBQL construction/rebuild stays outside capture; Warp supports complete construction/refit/rebuild capture with memory pools enabled.

For symmetric Chamfer:

```python
with wp.Tape() as tape:
    to_target = wp.geometry.knn(target_index, decoded, 1)
    to_decoded = wp.geometry.knn(decoded, targets, 1)
    # Reduce to_target[2] by B*N and to_decoded[2] by B*M; add the means.
```

Both directions contribute gradients to decoded coordinates. Set `requires_grad=True` on inputs needing gradients and preallocated distance outputs. Keep coordinates and selected IDs unchanged until backward completes. Empty clouds do not define a finite Chamfer mean. Sharing Torch storage does not automatically connect Warp's Tape to Torch autograd.

Run the [complete Chamfer example](../../warp/examples/geometry/example_point_query_chamfer.py):

```bash
uv run --no-sync -m warp.examples.geometry.example_point_query_chamfer \
  --device cuda:0 --backend cubql --reverse-backend warp
```

## Migrate the earlier branch API

| Earlier call | Recommended call |
| --- | --- |
| `wp.build_point_index(targets, ...)` | `wp.geometry.PointIndex(targets, ...)` |
| `wp.knn(targets, queries, k, bvh=indexes)` | `wp.geometry.knn(target_index, queries, k)` |
| `wp.knn(targets, queries, k)` | `wp.geometry.knn(targets, queries, k)` |
| `wp.point_query_config(...)` | `wp.geometry.point_query_config(...)` |
| Top-level configuration enums | Corresponding `wp.geometry` enums |

The old calls, configuration types, and `wp.CubqlPointIndex` remain functional with deprecation warnings. Their original signatures and raw BVH/cuBQL-index/tuple returns are preserved. `wp.point_query_knn_batch()` remains a compatibility alias with the same old return values. The geometry API returns a `PointIndex` as its first value; a legacy raw index is not a geometry owner. Scalar kernel queries and `wp.Bvh` construction are unchanged.

For fixed k, storage scales as `O(B*M + B*N*k)` with linear scratch, without a pairwise distance tensor. Measure construction amortization and the complete forward/backward step for your workload. The [comparison report](comparisons.md) retains external-library evidence with its original versions and dates; the [API revision report](api-revision/README.md) checks the current API and main update separately.
