Point-cloud nearest neighbors
=============================

Use an existing :class:`warp.Bvh` and an explicit contiguous one-dimensional
``wp.array[wp.vec3]`` for exact Euclidean nearest-neighbor queries. The BVH
owns acceleration storage; there is no separate point-cloud owner. Each
primitive must enclose the point with the same original array index.

.. code-block:: python

    points = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=wp.vec3)
    bvh = wp.Bvh(lowers=points, uppers=points, leaf_size=4)

    @wp.kernel
    def nearest(bvh_id: wp.uint64, points: wp.array[wp.vec3],
                queries: wp.array[wp.vec3], indices: wp.array[int]):
        t = wp.tid()
        indices[t] = wp.point_query_nearest(bvh_id, points, queries[t], wp.inf)

``point_query_nearest`` returns an original point index, or ``-1`` if no
eligible point exists. Read ``points[i]`` only when ``i >= 0``. A finite
``max_dist`` is a strict cutoff in coordinate units; zero, negative, and NaN
cutoffs return no neighbors, as do nonfinite query positions. Positive
infinity means an unbounded search. Pass ``exclude_index=t`` for self-query
rows only when query row ``t`` corresponds to point row ``t``. Exclusion
removes that index; coincident points with other indices remain eligible.

Automatic batch queries
-----------------------

The host wrapper defaults to Warp traversal and chooses point-tree leaf size,
query spacing, and query ordering automatically. It returns an ordinary tuple
containing the BVH and output arrays. Inputs are contiguous ``wp.vec3`` arrays
on the same device.

.. code-block:: python

    bvh, indices, distances_sq, counts = wp.point_query_knn_batch(points, queries, 8)

    # Reuse the tree for another query batch.
    _, indices2, distances_sq2, counts2 = wp.point_query_knn_batch(
        points, next_queries, 8, bvh=bvh
    )

    # Optional output reuse; shapes must match the query count and k.
    wp.point_query_knn_batch(
        points, queries, 8, bvh=bvh,
        indices=indices, distances=distances_sq, counts=counts,
    )

Output shapes are ``(Q, k)``, ``(Q, k)``, and ``(Q,)``. Indices default to
int32; caller-supplied int64 outputs are also supported. Distances default to
squared float32 values; pass ``squared=False`` for Euclidean distances.
The distance cutoff ``max_dist`` always uses coordinate units. Pass an int32
``exclude_indices`` array of shape ``(Q,)`` for per-query exclusions. Missing
neighbors have index -1 and infinite distance, with the valid count in
``counts``. Outputs must not overlap inputs or one another.

Keep the BVH, inputs, and outputs alive during asynchronous execution and
graph replay. For in-place point changes, refit the reused BVH before querying;
a changed point allocation or count requires a new BVH. The wrapper does not
silently refit an existing index. Warm up compilation before capture. Complete
CUDA LBVH construction plus query, and refit plus query, support capture with
memory pools enabled. The wrapper does not modify the global memory-pool
retention threshold. For gradients, it uses a separate differentiable gather
from the selected indices when either input requires gradients; ordinary
forward-only calls use a single selection/output kernel. Caller-provided distance
outputs need ``requires_grad=True`` to participate in backward. Keep points,
queries and selected indices unchanged between forward and backward.

The default GPU policy targets 16 scheduled warps per SM, with up to 32
threads per query. It uses leaf size 32 for tiny query batches or million-point
clouds, otherwise 4. Large dense batches additionally sort queries by Morton
keys computed entirely in Warp. Returned rows always retain the original
query order. This is a measured heuristic, not an architecture-independent
optimum. Override ``leaf_size`` when building, ``query_stride``, or ``ordered``
when appropriate; an existing BVH retains its layout.

Independent point-cloud batches
-------------------------------

Use two-dimensional ``wp.vec3`` arrays to query B independent clouds. A NumPy
float32 array with shape ``(B, M, 3)`` becomes a Warp ``vec3`` array with shape
``(B, M)``. Targets have M points per batch and queries have N points per batch;
M and N can differ. Ragged batches are not supported.

.. code-block:: python

    targets = wp.array(target_numpy, dtype=wp.vec3)  # (B, M, 3) on input
    decoded = wp.array(decoded_numpy, dtype=wp.vec3, requires_grad=True)
    indexes, indices, distances_sq, counts = wp.point_query_knn_batch(targets, decoded, 1)

    # Keep indexes while fixed targets stay unchanged; decoded may change in place.
    _, indices, distances_sq, counts = wp.point_query_knn_batch(
        targets, decoded, 1, bvh=indexes
    )

The returned index tuple has B entries. Neighbors never cross batch boundaries,
and output indices are local to the corresponding target cloud. Indices and
distances have shape ``(B, N, k)``; counts and optional exclusions have shape
``(B, N)``. Empty batches, empty query sets, and empty target sets are supported.
An empty target set produces zero counts and padded outputs. A singleton batch
retains its leading dimension. One-dimensional inputs retain their original
return shapes and single index object.

For symmetric squared Chamfer loss, average the decoded-to-target 1-NN distances
over B and N, and add the target-to-decoded mean over B and M. The fixed target
indexes can be reused across training steps. The reverse direction searches the
changing decoded points and needs rebuilt indexes or refitted Warp BVHs each
step. Both directions propagate distance gradients into the decoded coordinates;
target gradients also work when requested. Empty clouds do not define a finite
Chamfer mean, so reject them before computing that loss.

``warp/examples/geometry/example_point_query_chamfer.py`` demonstrates a small
optimization with both directions and a fixed target index. Run it with
``--backend warp`` or ``--backend cubql --device cuda:0``. Both use Warp for the
changing decoded index by default; select ``--reverse-backend cubql`` to measure
cuBQL in both directions. Acceleration and output storage
scale as ``O(B*M + B*N*k)``; the wrapper never materializes a ``B*N*M`` distance
tensor. This implementation builds one index and issues one query launch per
batch, so host launch overhead grows with B. CUDA graphs can reduce replay
overhead for reused indexes.

Optional native cuBQL queries
-----------------------------

Select :attr:`warp.PointQueryBackend.CUBQL` to call native cuBQL traversal with
tuned builder, leaf-size, and launch settings. Choose
:attr:`warp.PointQueryObjective.REUSE` for fixed targets, or
:attr:`warp.PointQueryObjective.BUILD_AND_QUERY` for construction followed by
one query batch. These policies come from measured configurations and are
heuristics; benchmark on the intended GPU and geometry.

.. code-block:: python

    indexes, indices, distances_sq, counts = wp.point_query_knn_batch(
        targets, decoded, 1,
        backend=wp.PointQueryBackend.CUBQL,
        objective=wp.PointQueryObjective.REUSE,
    )
    # The supplied indexes determine the backend and retain the build objective.
    result = wp.point_query_knn_batch(targets, decoded, 1, bvh=indexes)

The cuBQL route requires CUDA with memory-pool support and a Warp build with
cuBQL enabled. It supports k from 1 through 64, infinite cutoffs, no exclusions,
and no query ordering or leaf-size overrides. Unsupported requests raise an
error. It returns the same shapes, padding, and differentiable gathered distances
as the Warp route. Upstream cuBQL controls distance ranking and equal-distance
ties; original-index tie ordering is not promised, and rounding can differ from
Warp's scalar search. Coordinates and squared distances must be finite and
representable in float32.

Each native tree is owned by :class:`warp.CubqlPointIndex`. Construction and
rebuilding synchronize with the host and cannot be captured. Query-only capture
works after construction; keep owners and arrays alive throughout graph replay.
Destruction also synchronizes. After an in-place target update, call
``index.rebuild()`` for each affected index. Replacing the target allocation
requires new indexes. The route preserves the application's memory-pool
retention threshold.

The default Warp route remains useful for CPU execution, strict filtering and
tie semantics, construction-inclusive workloads, and capturing complete
build/refit-plus-query operations. Both backends support the independent batch
layout above.

Fast scalar calls and fused work
--------------------------------

``wp.point_query_knn()`` remains the same optimized scalar primitive used by
the batch wrapper. It introduces no batch allocation or hidden dispatch.
Use it inside a caller kernel to consume neighbors directly without storing
an entire ``(Q, k)`` result. :func:`warp.point_query_config` exposes the same
leaf-size and launch choices:

.. code-block:: python

    config = wp.point_query_config(len(points), len(queries), points.device)
    bvh = wp.Bvh(points, points, leaf_size=config.leaf_size)
    STRIDE = config.query_stride

    @wp.kernel
    def fused(bvh_id: wp.uint64, points: wp.array[wp.vec3],
              queries: wp.array[wp.vec3], sums: wp.array[float]):
        t = wp.tid()
        if t % wp.static(STRIDE) != 0:
            return
        q = t // wp.static(STRIDE)
        search = wp.point_query_knn(bvh_id, points, queries[q], wp.inf, 8)
        index = int(-1)
        total = float(0.0)
        while wp.point_query_next(search, index):
            total += wp.length_sq(queries[q] - points[index])
        sums[q] = total

    wp.launch(fused, len(queries) * STRIDE,
              inputs=[bvh.id, points, queries, sums],
              block_dim=config.block_dim, device=points.device)

This fused example uses natural query order. ``config.ordered`` indicates
when the batch wrapper would additionally sort queries; callers with custom
fusion can choose their own ordering. Sparse launches require the explicit
thread-index mapping above. A scalar call cannot rearrange the other threads
in its caller's kernel, and changing the block size alone does not reproduce
the sparse-query benefit. The ordinary one-query-per-thread pattern remains
valid and uses the same traversal, selection, and tie-breaking logic.

``wp.Bvh`` itself still defaults to leaf size **1**, for general intersection
queries. The historical benchmark label ``warp_default`` means its explicit
leaf-size-4, dense-launch baseline; it is not a different implementation or
literally every default argument of ``wp.Bvh``.

See ``warp/examples/geometry/example_point_query_batch.py`` for a runnable
batch-and-fusion example. These APIs require no PyTorch or external kNN library.

Eager k-nearest selection
-------------------------

``point_query_knn`` completes selection and sorting before returning an
inferred, kernel-local state. ``point_query_next`` only enumerates the final
neighbors. It never emits provisional candidates. Early exit from enumeration
does not shorten the search. Both operations can be called inside Warp functions.

.. code-block:: python

    K = 8

    @wp.kernel
    def neighbors(bvh_id: wp.uint64, points: wp.array[wp.vec3],
                  queries: wp.array[wp.vec3], indices: wp.array2d[int],
                  distances_sq: wp.array2d[float], counts: wp.array[int]):
        t = wp.tid()
        search = wp.point_query_knn(bvh_id, points, queries[t], wp.inf, wp.static(K))
        i = int(-1)
        count = int(0)
        while wp.point_query_next(search, i):
            indices[t, count] = i
            distances_sq[t, count] = wp.length_sq(queries[t] - points[i])
            count += 1
        counts[t] = count
        for r in range(count, wp.static(K)):
            indices[t, r] = -1
            distances_sq[t, r] = wp.inf

Allocate outputs with shapes ``(Q, K)``, ``(Q, K)``, and ``(Q,)``. Padding is a
wrapper choice: the iterator yields only valid neighbors, including for an empty
cloud or when ``K > N``. Once exhausted, subsequent calls return ``False`` and
leave the output index unchanged. Literal and ``wp.static`` positive integer
capacities are supported; runtime, zero, and negative capacities are compilation
errors. Different capacities have distinct inferred types and cache identities.

Results are ordered lexicographically by (computed squared distance, original
index). Duplicate positions remain separate neighbors; ties at the kth boundary
prefer the smaller original index. Float32 distance accumulation and box bounds
use compatible, noncontracting arithmetic. Inputs must have finite coordinates,
with differences and squared distances representable in float32. Exactness means
complete search with conservative pruning, not exact real arithmetic. Nearly tied
rankings need not match a float64 oracle. Distances recomputed in caller code can
round differently when floating-point contraction is enabled.

Lifetime and gradients
----------------------

Keep the BVH and point array alive for asynchronous launches and graph replay.
They must be on the same device with matching primitive-index mapping. Modifying
points in place requires ``bvh.refit()`` or ``bvh.rebuild()`` before another query.
Replacing an allocation or its length requires a new BVH. Aliased lower/upper
point bounds work with SAH, median, LBVH, and cuBQL constructors. Leaf size 4 is a
starting point; benchmark the builder and leaf size for your workloads. Concurrent
readers are safe; updates and refits must be ordered with readers on their streams.

CUDA LBVH construction, rebuild, refit, and point queries can be captured in CUDA
graphs. Construction and rebuild require CUDA memory-pool support to be enabled.
Warm up kernel compilation before capture, and keep the BVH, point arrays, and
outputs alive through replay. Other constructors retain their existing capture
restrictions. Construction initializes its device descriptor from graph-owned
kernel arguments, so replay does not depend on a temporary host descriptor.

Selection, sorting, and cursor updates are nondifferentiable. Ordinary gathers
and distance calculations differentiate with selected indices held fixed. For
``d2 = wp.length_sq(q - points[i])``, gradients are ``2*(q-points[i])`` for the query
and the negative for the point. Repeated point selections accumulate atomically.
Fused iterator consumption supports Warp tape replay. A separate selection kernel
followed by a differentiable gather kernel is also supported; guard padded ``-1``
slots. Do not mutate/refit/rebuild inputs between forward and backward execution.
No derivative is promised at neighbor switches, ties, or cutoff crossings.

Traversal and benchmarking
--------------------------

One thread traverses near children first and rechecks deferred bounds against a
shrinking candidate threshold. A fixed-capacity max heap retains candidates,
followed by in-place heap sorting. 1-NN has a scalar specialization. Stack overflow
clears the candidates and restarts an exhaustive scan, preserving completeness.
Scratch storage is O(tree depth + K), with no device heap allocation or hidden
host synchronization. Pathological distributions can require a linear scan.
Existing BVH sphere/AABB/ray overlap semantics are unchanged.

``warp.examples.benchmarks.benchmark_point_query`` measures query, build,
refit/rebuild, and one-shot costs separately. It includes ordinary exhaustive,
cooperative tiled, split-reference, expanding-sphere, and pinned PhysicsNeMo PR
baselines. Tiled variants use bounded partial selections and an exact merge;
they require cooperative launches and are not scalar per-thread functions.
The scan input is the unresampled Redwood fragment from Open3D's
``DemoICPPointClouds.zip``, with its normalization and checksum recorded.
Synthetic surface/volume inputs each run with and without Gaussian noise.

``warp.examples.benchmarks.profile_point_query`` collects individual native
kernel timings and, with ``--trace``, a PyTorch/CUPTI trace including CUB kernels,
allocations, and transfers. ``benchmark_point_query_optimization`` reruns the
original matrix with both the original launch configuration and explicitly tuned
configurations. It charges query ordering and construction to one-shot timings.

For tiny batches, placing many independent traversals in the same warp can
serialize divergent paths while leaving most SMs idle. The benchmark wrapper
``make_api_query(K, euclidean64=True, query_stride=32)`` can instead be launched
with ``dim=Q*32, block_dim=32`` to give each query one active lane in its own warp.
This preserves scalar query semantics; it is not a cooperative nearest-neighbor
algorithm. Large batches generally benefit from dense launches. Leaf size and
Morton ordering must also be measured: the tuning harness does not establish a
universal dispatch threshold.

For repeated one-shot calls, also measure the CUDA memory-pool release policy.
A zero release threshold permits unused pool memory to be returned to the driver
at synchronization, so warmup alone does not guarantee allocation reuse. The
comparison accepts ``--mempool-release-threshold-mib 512`` to retain memory across
samples, and ``benchmark_point_query_allocator`` isolates this effect with
alternating pool policies and paired comparisons. This is an explicit application
memory/performance tradeoff, not a change to Warp's default allocator policy.
