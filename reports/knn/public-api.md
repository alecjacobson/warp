# Warp kNN public API and settings guide

The `alecjacobson/knn` branch exposes host array queries and scalar queries inside Warp kernels. `wp.build_point_index()` builds reusable target indexes, and `wp.knn()` queries single clouds or independent batches from Python. Inside a kernel, use `wp.point_query_knn()` with `wp.point_query_next()` to consume neighbors, or `wp.point_query_nearest()` for one neighbor. These APIs are implemented at [cafc98ee6](https://github.com/alecjacobson/warp/commit/cafc98ee6).

**Start with `backend=wp.PointQueryBackend.WARP` when targets change frequently. For repeated CUDA queries against fixed targets, start with `backend=wp.PointQueryBackend.CUBQL` and `objective=wp.PointQueryObjective.REUSE`.** For symmetric Chamfer loss, use that fixed-target choice with `backend=wp.PointQueryBackend.WARP` for the changing decoded cloud, then measure the complete training step. Leave the lower-level tuning overrides automatic initially. The [current comparisons](comparisons.md) contain the latest measurements and methodology.

## Choose where the query runs

| Caller | Public API | What it returns or consumes |
| --- | --- | --- |
| Python, querying whole arrays | `wp.build_point_index(...)` and `wp.knn(...)` | Reusable indexes and arrays of indices, distances, and counts. Supports `backend=wp.PointQueryBackend.WARP` or `backend=wp.PointQueryBackend.CUBQL`. |
| A Warp kernel or `@wp.func`, consuming up to k neighbors | `wp.point_query_knn(...)` followed by `wp.point_query_next(...)` | A kernel-local query state and one selected target index per iteration. Uses a `wp.Bvh`. |
| A Warp kernel or `@wp.func`, needing only one neighbor | `wp.point_query_nearest(...)` | One target index, or `-1` when no point is eligible. Uses a `wp.Bvh`. |

The array APIs run on the host and launch query work. The scalar APIs run inside caller kernels and can fuse per-neighbor work without storing all neighbor outputs. The [scalar example below](#scalar-queries-inside-a-kernel) shows the complete iterator and launch pattern.

## Choose a configuration

| Your workload or requirement | Starting choice | Reason |
| --- | --- | --- |
| General use, CPU execution, or one portable configuration | `backend=wp.PointQueryBackend.WARP` (default) | Supports CPU/CUDA and the full query contract. |
| Build a new target index for each query call | `backend=wp.PointQueryBackend.WARP` | Lowest build-plus-query median among public settings in 38/40 cross-library cases and 23/24 pulmonary/batched cases; see [comparisons](comparisons.md). |
| Many queries against unchanged targets on CUDA | `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE` | Faster reused queries in the measured workloads; pay construction once. |
| Capture construction or refitting as part of a CUDA graph | `backend=wp.PointQueryBackend.WARP` | Supports construction/refit capture with CUDA memory pools enabled. |
| Capture queries against an already built index | For fixed CUDA targets, start with `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE`; `backend=wp.PointQueryBackend.WARP` also supports query replay | Build native cuBQL indexes outside capture and retain them. |
| Finite distance cutoff, self-exclusion, deterministic original-index ties, or k greater than 64 | `backend=wp.PointQueryBackend.WARP` | `backend=wp.PointQueryBackend.CUBQL` does not support these requests. |
| Symmetric Chamfer with fixed targets and changing decoded points | Fixed direction: `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE`; changing direction: `backend=wp.PointQueryBackend.WARP` | The two directions have different construction costs. |
| Explicitly evaluate native cuBQL for targets that change often | `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.BUILD_AND_QUERY`, compared against `backend=wp.PointQueryBackend.WARP` | This policy trades query speed for cheaper construction; it does not guarantee a win over the default backend. |

Set these arguments on `wp.build_point_index(...)`, or on `wp.knn(...)` when constructing and querying in one call. For subsequent queries, pass `bvh=indexes` to reuse the selected backend and index.

The backend is an explicit choice. There is no automatic switch between Warp and cuBQL, and unsupported cuBQL options raise an error. A supplied index determines the backend unless the caller specifies one; a conflicting choice raises an error.

`PointQueryObjective` selects a **cuBQL tuning policy**. `REUSE` favors repeated searches; `BUILD_AND_QUERY` favors construction followed by one query set. Neither value restricts how many times an index can be reused. The objective does not select the backend or change Warp's policy. Existing cuBQL indexes retain their construction objective; passing a different objective to `knn(..., bvh=indexes)` does not rebuild or retune them. Construct new indexes to change the policy.

## Build once and query repeatedly

```python
import warp as wp

# target_numpy: float32 coordinates with shape (B, M, 3).
# decoded_numpy: float32 coordinates with shape (B, N, 3).
targets = wp.array(target_numpy, dtype=wp.vec3, device="cuda:0")
decoded = wp.array(decoded_numpy, dtype=wp.vec3, device="cuda:0")

indexes = wp.build_point_index(
    targets,
    num_queries=decoded.shape[1],
    k=1,
    backend=wp.PointQueryBackend.CUBQL,
    objective=wp.PointQueryObjective.REUSE,
)

_, indices, distances_sq, counts = wp.knn(targets, decoded, 1, bvh=indexes)
```

Construction needs only targets. The example uses the decoded array to obtain its expected count; the same count can be supplied before query coordinates exist. `num_queries` is a hint for the expected queries **per cloud**, and `k` is a hint for neighbor count. Supply realistic hints when known. Omitting `num_queries` uses M; omitting `k` uses 1. Later calls can use a different N or valid k without invalidating the index, although the original tree settings may be less efficient.

The name `bvh=` accepts either backend's index object. For one cloud the builder returns a `wp.Bvh` or `wp.CubqlPointIndex`; for batches it returns a tuple of B such objects. Pass the returned object or tuple directly, rather than extracting native handles.

For construction and query in one call, omit `bvh`:

```python
indexes, indices, distances_sq, counts = wp.knn(targets, decoded, 1)
```

This uses Warp by default and returns the indexes for optional later reuse. Calling it again without `bvh=` constructs new indexes.

## Shapes and results

Inputs are contiguous float32 `wp.vec3` arrays on the same device. N and M may differ, but every cloud in a batch has the same N and M. A query in batch b searches only targets in batch b.

| Value | One cloud | B independent clouds |
| --- | --- | --- |
| Input coordinate tensor for targets | `(M, 3)` | `(B, M, 3)` |
| Warp target array shape, dtype `wp.vec3` | `(M,)` | `(B, M)` |
| Warp query array shape, dtype `wp.vec3` | `(N,)` | `(B, N)` |
| Returned indices and distances | `(N, k)` | `(B, N, k)` |
| Returned counts | `(N,)` | `(B, N)` |
| Returned index owner | One index | Tuple of B indexes |

`wp.knn()` always returns `(indexes, indices, distances, counts)`. Indices are local to each target cloud and default to int32; caller-supplied int64 outputs are supported. Counts are int32. Distances are float32 and **squared by default**; use `squared=False` for Euclidean distances. Missing neighbors have index `-1` and infinite distance, with the number of valid neighbors in `counts`. Empty clouds and B=0 are supported; empty inputs do not define a finite Chamfer mean. B=1 retains the batch dimension. Ragged batches and cross-batch broadcasting are outside this API.

Both backends perform exact nearest-neighbor selection under their float32 arithmetic. Equal-distance choices and rounding can differ between them. Warp breaks equal-distance ties by original target index; cuBQL uses upstream tie selection.

## Which arguments should users set

| Argument | Recommended use |
| --- | --- |
| `k` | Required positive Python integer on `knn`; use 1 for Chamfer. cuBQL supports 1–64. |
| `backend` | Set `wp.PointQueryBackend.CUBQL` explicitly for native cuBQL indexes; otherwise use the default `wp.PointQueryBackend.WARP` or infer from an existing index. |
| `objective` | Use `wp.PointQueryObjective.REUSE` for fixed cuBQL targets. Use `wp.PointQueryObjective.BUILD_AND_QUERY` when evaluating cuBQL construction-inclusive workloads. |
| `num_queries` | On `build_point_index` only: expected N per cloud, used for tree tuning. |
| `bvh` | Supply the retained index or tuple to avoid rebuilding targets. |
| `squared` | Keep `True` for squared Chamfer; set `False` when consumers need actual distances. |
| `max_dist` | Warp only for finite cutoffs. Always a distance in coordinate units, even when outputs are squared; the cutoff is strict. Default is infinity. |
| `exclude_indices` | Warp only: an int32 array matching query shape, containing one batch-local target ID to omit per query; `-1` omits nothing. Useful for self-kNN in normal estimation. |
| `indices`, `distances`, `counts` | Optional output buffers with the documented shapes, dtypes, and device. Reuse them for stable shapes or capture. They must not overlap inputs or one another. |
| `leaf_size` | Leave automatic. Warp-only positive integer override during construction; rejected with an existing index. |
| `query_stride` | Leave automatic. Expert override for launch spacing: 1, 2, 4, 8, 16, or 32. Supported by both backends. |
| `ordered` | Leave `None` for the backend's policy. Warp can reorder query work internally and restore original output order. Direct cuBQL rejects `True`. |

The cuBQL route requires CUDA, memory-pool support, and a Warp build with cuBQL enabled. Use finite target coordinates with representable float32 squared distances. There is no public cuBQL builder or block-size knob: the objective and shape hints select the measured settings.

`wp.point_query_config(M, N, device)` exposes Warp's suggested launch/tree settings for advanced users, including custom fused kernels. It is not the direct-cuBQL policy selector. Most callers can use the automatic settings in `knn`.

## Scalar queries inside a kernel

Use `wp.point_query_knn()` to select the final neighbors for one query point, then `wp.point_query_next()` to visit them. Each kernel thread can perform its own query. This example consumes eight neighbors and writes only their summed squared distance:

```python
import warp as wp


@wp.kernel
def sum_neighbor_distances_kernel(
    bvh_id: wp.uint64,
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    sums: wp.array[float],
):
    row = wp.tid()
    query_point = queries[row]
    search = wp.point_query_knn(bvh_id, points, query_point, wp.inf, 8)
    index = int(-1)
    total = float(0.0)
    while wp.point_query_next(search, index):
        # Replace this with your own per-neighbor computation.
        total += wp.length_sq(query_point - points[index])
    sums[row] = total


# Host setup: points and queries are contiguous 1D wp.vec3 arrays
# of shape (M,) and (N,) on the same CPU or CUDA device.
bvh = wp.build_point_index(
    points,
    num_queries=queries.shape[0],
    k=8,
    backend=wp.PointQueryBackend.WARP,
)
sums = wp.empty(queries.shape[0], dtype=float, device=points.device)
wp.launch(
    sum_neighbor_distances_kernel,
    dim=queries.shape[0],
    inputs=[bvh.id, points, queries, sums],
    device=points.device,
)
```

`search` has an inferred, kernel-local type; no public query-state class needs to be constructed or annotated. `wp.point_query_next(search, index)` advances the cursor, writes the next index, and returns `True`. It yields only valid neighbors, ordered by computed squared distance and then original target index. If fewer than k are eligible, the loop ends early without padding. Once exhausted, it returns `False` and leaves `index` unchanged.

Use the `while wp.point_query_next(search, index)` form shown above. Direct `for index in search` iteration is not implemented for this query type and fails to compile. A fixed `for` loop over k also needs to check `point_query_next` because fewer than k neighbors may be eligible.

Selection is eager: `wp.point_query_knn()` finishes searching and sorting before the loop starts. Breaking out of `point_query_next` enumeration early does not reduce search work. The scalar k must be a positive compile-time integer, such as literal `8` above or `wp.static(K)`; a runtime kernel argument cannot supply it.

These scalar functions take a `wp.Bvh.id` and the corresponding one-dimensional target array. Use `backend=wp.PointQueryBackend.WARP` when building through the helper. A `wp.CubqlPointIndex.id` is not a Warp BVH handle and must not be passed to them; direct native cuBQL queries are exposed through the host `wp.knn` route. For independent clouds, select the matching Warp BVH and target cloud for each query, or launch this kernel separately per cloud.

For one neighbor, use the simpler scalar call:

```python
@wp.kernel
def nearest_distance_kernel(
    bvh_id: wp.uint64,
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    distances_sq: wp.array[float],
):
    row = wp.tid()
    query_point = queries[row]
    index = wp.point_query_nearest(bvh_id, points, query_point, wp.inf)
    distance_sq = wp.inf
    if index >= 0:
        distance_sq = wp.length_sq(query_point - points[index])
    distances_sq[row] = distance_sq
```

Launch it with the same `bvh.id`, `points`, and `queries`, plus an `(N,)` float32 output. Guard `index >= 0` before accessing target coordinates. Both scalar selectors accept a distance cutoff in coordinate units and an optional scalar `exclude_index`; pass the corresponding target ID to exclude self. Selection and cursor updates are nondifferentiable. Gathered coordinates and distance calculations support gradients with selected IDs held fixed; keep inputs and the index unchanged through backward.

The example uses one query per thread. `wp.point_query_config()` also exposes tuned launch spacing for custom kernels; applying its `query_stride` requires matching thread-to-query indexing. See the [user guide](../../docs/user_guide/point_queries.rst) and [runnable fused example](../../warp/examples/geometry/example_point_query_batch.py) for that pattern.

## Updating data and replaying graphs

Changing query coordinates requires another query, with no target rebuild. Changing target coordinates in place requires updating every index: call `index.refit()` for Warp or `index.rebuild()` for cuBQL. For batched inputs, iterate over the returned tuple. A new target allocation, point count, or batch layout requires new indexes. Indexes retain references to target coordinates; they do not take a coordinate snapshot.

For stable query shapes, reuse outputs:

```python
decoded.assign(next_decoded_numpy)
wp.knn(
    targets,
    decoded,
    1,
    bvh=indexes,
    indices=indices,
    distances=distances_sq,
    counts=counts,
)
```

After a warmup query, the same operation can be captured:

```python
with wp.ScopedCapture(device=targets.device) as capture:
    wp.knn(
        targets,
        decoded,
        1,
        bvh=indexes,
        indices=indices,
        distances=distances_sq,
        counts=counts,
    )

decoded.assign(next_decoded_numpy)
wp.capture_launch(capture.graph)
```

Keep index owners, input arrays, and outputs alive through replay. Replays use the captured allocations and shapes; update their contents in place. A shape or allocation change requires new outputs as appropriate and a new capture. cuBQL construction, rebuilding, and destruction can synchronize; construction/rebuilding must occur outside capture. Warp supports construction/refit capture with CUDA memory pools enabled.

The measured public wrapper adds about 40 µs to an ordinary B=1 call versus standalone cuBQL. Graph replay avoids repeating that Python work. For small repeated queries, measure graphs and preallocated outputs before manually tuning launch spacing. Use the [scalar APIs](#scalar-queries-inside-a-kernel) when per-neighbor work belongs inside your own kernel.

## Chamfer loss with fixed pulmonary targets

Create decoded coordinates with `requires_grad=True`. Build fixed target indexes outside the optimization loop; within the loop, query both directions:

```python
with wp.Tape() as tape:
    to_target = wp.knn(targets, decoded, 1, bvh=indexes)
    to_decoded = wp.knn(decoded, targets, 1, backend=wp.PointQueryBackend.WARP)
    # Reduce to_target[2] and to_decoded[2] into the scalar loss here.
```

For the sum-of-means convention, reduce decoded-to-target squared distances by `B*N` and target-to-decoded squared distances by `B*M`, then add them. Both directions contribute gradients to decoded coordinates. The changing decoded cloud is the indexed target in the reverse call, so that direction needs a new or refitted index each step.

Warp's tape differentiates selected distances with neighbor IDs held fixed. Set `requires_grad=True` on whichever inputs need gradients; preallocated distance outputs also need it. Keep input coordinates and selected IDs unchanged until backward completes. Run backward before updating decoded coordinates or reusing selection buffers. Do not assume wrapping a Torch tensor automatically connects Warp's tape to Torch autograd; the measured example uses Warp's tape.

The [complete runnable example](../../warp/examples/geometry/example_point_query_chamfer.py) includes reductions, backward, and updates:

```bash
uv run --no-sync -m warp.examples.geometry.example_point_query_chamfer \
  --device cuda:0 --backend cubql --reverse-backend warp
```

## Measurements behind the recommendations

The [current comparisons](comparisons.md) are the authoritative performance report for this branch, with four colored tables comparing public settings and external libraries, plus separate pulmonary/Chamfer and retained-memory measurements. They also explain the retained standalone-integration evidence and its different timing convention. All current measurements use the 94-SM RTX PRO 6000 Blackwell MIG slice; historical 46-SM experiments are archived separately.

Reuse tuning can make construction substantially more expensive. Compare `T_build + R*T_query` for the expected R query calls, using the same allocation/capture mode and including relevant retirement costs. For symmetric Chamfer, include the changing reverse index and backward pass. Query-only speedups alone do not establish a win for a short-lived index or a complete training step.

For fixed k, index/output storage scales as `O(B*M + B*N*k)`, plus input coordinates and linear scratch, without a `B*N*M` pairwise tensor. Each batch currently uses a separate index and query operation, so large B also deserves measurement.

References: [current comparisons](comparisons.md), [public API implementation](../../warp/_src/point_query.py), and [user guide](../../docs/user_guide/point_queries.rst).

`wp.point_query_knn_batch()` remains a working deprecated alias for `wp.knn()`, with the same arguments and four return values. New callers should use `wp.knn()`.
