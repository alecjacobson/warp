# Warp kNN public API and settings guide

The `alecjacobson/knn` branch exposes two main operations: `wp.build_point_index()` builds reusable target indexes, and `wp.knn()` queries them. Both accept a single point cloud or independent batches. This guide describes the API at [cafc98ee6](https://github.com/alecjacobson/warp/commit/cafc98ee6).

**Start with Warp when targets change frequently. For repeated CUDA queries against fixed targets, start with cuBQL and the `REUSE` objective.** For symmetric Chamfer loss, use cuBQL for the fixed target cloud and Warp for the changing decoded cloud, then measure the complete training step. Leave the lower-level tuning overrides automatic initially.

## Choose a configuration

| Your workload or requirement | Starting choice | Reason |
| --- | --- | --- |
| General use, CPU execution, or one portable configuration | `backend=wp.PointQueryBackend.WARP` (default) | Supports CPU/CUDA and the full query contract. |
| Build a new target index for each query call | `backend=wp.PointQueryBackend.WARP` | Lowest build-plus-query median in 23/24 initial benchmark cases; the remaining case was effectively tied. |
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

The measured public wrapper adds about 40 µs to an ordinary B=1 call versus standalone cuBQL. Graph replay avoids repeating that Python work. For small repeated queries, measure graphs and preallocated outputs before manually tuning launch spacing. For custom fused Warp kernels, the scalar `wp.point_query_knn` and `wp.point_query_nearest` remain available; see the [kernel API examples](../../docs/user_guide/point_queries.rst).

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

Measurements use the 94-SM RTX PRO 6000 Blackwell MIG slice in this project. They guide starting settings; they do not establish a fastest configuration for every GPU or geometry.

| Measurement | Result | Consequence for a caller |
| --- | --- | --- |
| Completed API, reused queries including output allocation | cuBQL reuse wins 23/23 workloads; 1.33× geometric-mean speedup over Warp | Consider cuBQL for fixed targets. |
| Completed API, reused query graphs | cuBQL reuse wins 22/23; 1.26× geometric-mean speedup over Warp | Capture can retain the query advantage; a Warp graph still wins one case. |
| Completed API, cuBQL integration versus matched standalone | Query graphs average about 1% slower; ordinary public B=1 calls add about 40 µs | Distinguish traversal speed from Python call overhead. |
| Initial implementation, build plus query | Warp has the lowest median in 23/24 workloads; remaining case effectively tied | Start with Warp when rebuilding often. |
| Initial implementation, dense symmetric Chamfer, B=4, M=379,385, N=8,192 | Warp both ways: 4.04 ms; cuBQL fixed/Warp reverse: 3.97 ms; cuBQL both ways: 13.68 ms | Faster fixed-target queries produce a modest full-step gain; backend choice for the reverse direction matters. |

Reuse tuning can make construction substantially more expensive. In the initial dense case, cuBQL reuse construction plus query took 46.12 ms versus 1.34 ms for Warp. Decide using `T_build + R*T_query` for the expected R query calls, measured with the same allocation/capture mode and including relevant retirement costs. Query-only speedups alone do not establish a win for a short-lived index.

For fixed k, index/output storage scales as `O(B*M + B*N*k)`, plus input coordinates and linear scratch, without a `B*N*M` pairwise tensor. The initial dense case retained 98.79 MiB in cuBQL indexes/outputs plus 17.74 MiB in inputs; a float32 pairwise tensor would be 46.31 GiB. These are retained-memory figures, not peak training memory. Each batch currently uses a separate index and query operation, so large B also deserves measurement.

Sources: [completed API and standalone comparison](completion/README.md), [construction, Chamfer, and memory measurements](batched-cubql/README.md), [public API implementation](../../warp/_src/point_query.py), and [user guide](../../docs/user_guide/point_queries.rst).

`wp.point_query_knn_batch()` remains a working deprecated alias for `wp.knn()`, with the same arguments and four return values. New callers should use `wp.knn()`.
