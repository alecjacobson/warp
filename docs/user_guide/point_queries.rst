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
