Point-cloud nearest neighbors
=============================

Kernel queries on an existing BVH
---------------------------------

Build a :class:`warp.Bvh` with coincident lower and upper bounds to index
points. The existing BVH owns acceleration storage and exposes the handle used
inside kernels. Points are a contiguous one-dimensional ``wp.vec3`` array;
each BVH primitive must enclose the point at the same original array index.

.. code-block:: python

    import warp as wp

    @wp.kernel
    def nearest_kernel(
        bvh_id: wp.uint64,
        points: wp.array[wp.vec3],
        queries: wp.array[wp.vec3],
        indices: wp.array[int],
    ):
        row = wp.tid()
        indices[row] = wp.point_query_nearest(bvh_id, points, queries[row], wp.inf)

    points = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=wp.vec3)
    queries = wp.array([[0.9, 0, 0]], dtype=wp.vec3, device=points.device)
    bvh = wp.Bvh(lowers=points, uppers=points)
    indices = wp.empty(len(queries), dtype=int, device=points.device)
    wp.launch(nearest_kernel, len(queries), [bvh.id, points, queries, indices], device=points.device)

``wp.point_query_nearest()`` returns the original target index, or -1 when no
point is eligible. Guard the index before reading target coordinates. For k
neighbors, select once and enumerate the sorted results:

.. code-block:: python

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

The scalar k is a positive compile-time integer: use a literal or ``wp.static(K)``.
Selection and sorting finish before ``point_query_knn`` returns. The iterator
visits valid neighbors in increasing squared-distance order, breaking equal-distance
ties by original target index. It stops after the available neighbors, without
padding. Once exhausted, ``point_query_next`` returns False and leaves its output
index unchanged. Breaking enumeration early does not reduce search work.
Use the ``while`` form above; direct ``for index in search`` iteration is not
implemented.

Both selectors accept a strict ``max_dist`` cutoff in coordinate units. Zero,
negative, or NaN cutoffs, and nonfinite query positions, return no neighbors;
positive infinity permits an unbounded search. Optional ``exclude_index``
removes one original target ID; -1 excludes nothing. Use it for self-queries
only when the query row corresponds to that target ID. Coincident points with
other indices remain eligible. Target coordinates must be finite, with
representable float32 squared distances.

Selection and cursor updates are nondifferentiable. Gathered coordinates and
subsequent distance calculations support gradients with selected IDs fixed.
Keep inputs and the BVH unchanged until backward completes. Call ``bvh.refit()``
or ``bvh.rebuild()`` after changing coordinates in place, and keep the BVH alive
through asynchronous work and graph replay.

Array queries and reusable targets
----------------------------------

The optional :mod:`warp.geometry` module provides an array-level convenience
API. :class:`warp.geometry.PointIndex` owns the targets and their per-cloud
indexes; :func:`warp.geometry.knn` launches queries. The default implementation
builds ordinary Warp BVHs and calls the scalar primitive shown above.

.. code-block:: python

    import warp as wp
    import warp.geometry

    # Construct without query coordinates or neighbor output allocation.
    target_index = wp.geometry.PointIndex(points, num_queries=1024, k=8)
    _, indices, distances_sq, counts = wp.geometry.knn(target_index, queries, 8)

    # Supplying an array constructs and queries in one call.
    target_index, indices, distances_sq, counts = wp.geometry.knn(points, queries, 8)

The return is always ``(PointIndex, indices, distances, counts)``. Supplying an
owner returns that same object without rebuilding. Supplying an array creates a
new owner each time. The owner retains the target allocation by reference;
construction does not copy coordinates. Query-count and k hints affect tuning,
but later calls may use different query counts and valid k values.

Inputs are contiguous float32 ``wp.vec3`` arrays on the same device. For single
clouds, output shapes are ``(N, k)``, ``(N, k)``, and ``(N,)``. Indices default to
int32; preallocated int64 outputs are supported. Distances are squared float32
values unless ``squared=False``. Missing neighbors have index -1 and infinite
distance; counts report how many entries are valid. Optional int32
``exclude_indices`` has the query shape and contains target IDs to omit, with
-1 excluding nothing. Target coordinates must be finite, with queried squared
distances representable in float32. Nonfinite queries return zero counts and
padded results. The Warp backend also returns no neighbors for nonpositive or
NaN cutoffs; native cuBQL requires positive infinity.
Output buffers must not overlap inputs or one another.

``target_index.bvhs`` exposes a tuple of ordinary Warp BVHs, one per cloud,
for applications that want to fuse kernel work. A single cloud has one entry.
There is no ambiguous aggregate ``id``. Native cuBQL owners reject ``bvhs``
access because their handles are incompatible with scalar Warp BVH queries.

Independent point-cloud batches
-------------------------------

A contiguous NumPy tensor with shape ``(B, M, 3)`` becomes a ``wp.vec3`` array
with shape ``(B, M)``. Queries have shape ``(B, N)``; M and N may differ.

.. code-block:: python

    targets = wp.array(target_numpy, dtype=wp.vec3)
    decoded = wp.array(decoded_numpy, dtype=wp.vec3, requires_grad=True)
    target_index = wp.geometry.PointIndex(targets, num_queries=decoded.shape[1])
    _, indices, distances_sq, counts = wp.geometry.knn(target_index, decoded, 1)

Each batch searches only its corresponding target cloud. Indices are local to
that cloud. Indices/distances have shape ``(B, N, k)``; counts and exclusions
have shape ``(B, N)``. B=1 preserves the leading dimension. Empty batches,
queries, and targets are supported; empty targets produce zero counts and
padded results. Ragged batches and cross-batch broadcasting are unsupported.

The owner has the same type for single clouds and batches. ``refit()`` or
``rebuild()`` updates every owned index after in-place target changes. New
allocations, point counts, or batch layouts require a new owner. Changing
query coordinates does not require target updates. For fixed k, storage is
``O(B*M + B*N*k)`` with linear scratch; no B×N×M distance tensor is materialized.
The current implementation submits one query operation per cloud.

Backend choices and tuning
--------------------------

``wp.geometry.PointQueryBackend.WARP`` supports CPU/CUDA, finite cutoffs,
exclusions, original-index ties, and CUDA construction/refit capture with memory
pools enabled. It is the default. For repeated queries against fixed CUDA
clouds, native cuBQL is an explicit alternative:

.. code-block:: python

    target_index = wp.geometry.PointIndex(
        targets,
        num_queries=decoded.shape[1],
        k=1,
        backend=wp.geometry.PointQueryBackend.CUBQL,
        objective=wp.geometry.PointQueryObjective.REUSE,
    )
    _, indices, distances_sq, counts = wp.geometry.knn(target_index, decoded, 1)

Native cuBQL requires CUDA, memory pools, a build with cuBQL enabled, and k=1-64.
It requires infinite cutoff, no exclusions, and no query reordering
(``ordered=None`` or ``False``; ``True`` is rejected);
unsupported options raise errors. Equal-distance choices follow upstream
cuBQL ordering. Construction, rebuilding, and retirement can synchronize;
construct and rebuild outside capture. Use ``rebuild()`` after modifying
native cuBQL targets, finish outstanding readers first, and recapture query
graphs afterward because native handles change.

``PointQueryObjective.REUSE`` favors reused queries;
``PointQueryObjective.BUILD_AND_QUERY`` favors construction followed by a query
set. These are measured cuBQL policies, not universal performance guarantees.
They do not select a backend or affect Warp's policy. Existing owners retain
their backend and objective; conflicting overrides raise errors. Any non-None
``leaf_size`` override is rejected for an existing owner, even if it matches the
current layout.

Leave ``leaf_size``, ``query_stride``, and ``ordered`` automatic initially.
``wp.geometry.point_query_config(M, N, device)`` exposes the default Warp policy
for custom fused kernels. Its launch spacing requires matching thread-to-query
indexing. The default array path already applies this policy; scalar calls can
use the same traversal and tuning without an array output allocation. The
ordinary ``wp.Bvh`` defaults remain unchanged.

For the complete argument and return contracts, see :func:`warp.geometry.knn`,
:class:`warp.geometry.PointIndex`, and :func:`warp.geometry.point_query_config`.
The generic ``wp.Bvh(constructor="cubql")`` builds a converted Warp BVH, which
still uses Warp traversal. It is separate from
``PointQueryBackend.CUBQL``, whose native indexes are queried through
``wp.geometry.knn()``. A raw ``wp.Bvh`` is not accepted as a ``PointIndex``; use it
directly with scalar queries or retain an earlier ``wp.knn(..., bvh=...)`` call
while migrating.

Graphs and gradients
--------------------

After warmup, reuse outputs and capture queries against a retained owner:

.. code-block:: python

    with wp.ScopedCapture(device=targets.device) as capture:
        wp.geometry.knn(
            target_index, decoded, 1,
            indices=indices, distances=distances_sq, counts=counts,
        )
    decoded.assign(next_decoded_numpy)
    wp.capture_launch(capture.graph)

Keep the owner, inputs, and outputs alive through replay. Shape/allocation
changes require new outputs and a new capture. Warp construction, refit, and
rebuild support CUDA capture with memory pools enabled; native cuBQL supports
queries against indexes constructed outside capture.

For symmetric Chamfer, query both directions. Retain the fixed target owner;
construct or refit the changing decoded index each step:

.. code-block:: python

    with wp.Tape() as tape:
        to_target = wp.geometry.knn(target_index, decoded, 1)
        to_decoded = wp.geometry.knn(decoded, targets, 1)
        # Reduce to_target[2] by B*N and to_decoded[2] by B*M, then sum.

Euclidean distances use a zero subgradient at coincident coordinates.

Set ``requires_grad=True`` on each input needing gradients, and on any
preallocated distance output. Both directions contribute to decoded gradients.
Keep coordinates and selected indices unchanged until backward completes.
Empty clouds do not define a finite Chamfer mean. Sharing Torch tensor storage
does not automatically connect Warp's Tape to Torch autograd.

Migration from the experimental top-level API
---------------------------------------------

``wp.knn()``, ``wp.build_point_index()``, ``wp.point_query_config()``, and
``wp.point_query_knn_batch()`` remain functional deprecated entry points.
Their original signatures and raw BVH/cuBQL-index/tuple returns are preserved.
The old top-level configuration types remain deprecated aliases. New code
should explicitly import ``warp.geometry`` and use its configuration enums.

Replace ``wp.build_point_index(targets, ...)`` with
``wp.geometry.PointIndex(targets, ...)``. Replace
``wp.knn(targets, queries, k, bvh=indexes)`` with
``wp.geometry.knn(target_index, queries, k)``. Both array APIs return four values,
but the geometry API's first value is always a ``PointIndex``. Do not pass a
legacy raw index as a geometry owner. Kernel calls and ordinary ``wp.Bvh``
construction are unchanged.

Runnable examples
-----------------

:github:`warp/examples/geometry/example_point_query.py` demonstrates scalar
queries and gradients through gathered coordinates.
:github:`warp/examples/geometry/example_point_query_batch.py` compares the
geometry array API with a fused caller kernel using the same Warp BVH.
:github:`warp/examples/geometry/example_point_query_chamfer.py` includes the
complete symmetric loss reduction and optimization loop for fixed targets.

From a built checkout, run:

.. code-block:: bash

    uv run --no-sync -m warp.examples.geometry.example_point_query --device cpu
    uv run --no-sync -m warp.examples.geometry.example_point_query_batch --device cuda:0
    uv run --no-sync -m warp.examples.geometry.example_point_query_chamfer --device cuda:0 --backend cubql --reverse-backend warp
