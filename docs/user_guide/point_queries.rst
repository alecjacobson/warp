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
Unordered batches submit one query kernel across all clouds, with a single
batched distance-gather kernel when gradients are enabled. CUDA graph replay
uses retained device descriptors and reads current query coordinates. Query
spacing is selected from the total ``B*N`` work rather than treating each cloud
as a separate sparse launch. ``ordered=True`` retains the per-cloud ordering
path; APIC captures retain per-cloud launches for resource tracking. Index construction, refitting, and rebuilding still process each cloud;
reuse a ``PointIndex`` when targets remain fixed.

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
They do not select a backend or affect Warp's policy. The dense-batch query
launch policy also accounts for total query count, independently of the tree's
construction objective. Existing owners retain
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


Batched Chamfer performance
---------------------------

Measure the complete workload when selecting a backend: both search directions,
distance gathers, reduction, backward, and any changing-target index construction.
An encoder's full training epoch also includes work outside Chamfer. Speedups
for loss evaluation alone do not predict the same training speedup.

A customer using an earlier branch revision (``6e7ce08ea``) reported the
following A40 results for 32 independently subsampled pulmonary point clouds.
These are externally reported measurements, not results reproduced by this
repository's benchmark. The customer's chunked baseline performs direct search
under ``torch.no_grad()`` followed by differentiable selected-pair gathers.

.. list-table:: Customer-reported symmetric Chamfer forward + backward (ms)
   :header-rows: 1

   * - Points per cloud
     - cdist
     - Chunked cdist + gathers
     - Earlier Warp implementation
   * - 1,024
     - 6.1
     - 3.2
     - 11.0
   * - 4,096
     - 90.0
     - 46.6
     - 10.6
   * - 16,384
     - Out of memory
     - 762.8
     - 17.7
   * - 65,536
     - Out of memory
     - 13,637
     - 34.4

The corresponding full training epoch times (chunked baseline versus Warp)
were 0.55 versus 0.60 seconds at 1,024 points, 1.86 versus 1.39 at 4,096,
14.17 versus 4.69 at 16,384, 49.23 versus 9.15 at 32,768, and approximately
191 (estimated) versus 17.95 at 65,536. Small clouds can favor brute force;
large clouds avoid the quadratic distance-matrix cost.

That earlier implementation submitted Python query and gradient work for each
cloud separately. The current unordered batch path fuses those launches. Tree
construction remains per cloud, so fixed-target reuse and graph capture still
matter. The runnable benchmark below separates reused queries, query replay,
reused-index Chamfer, and Chamfer with a rebuilt reverse index.

Backend rankings depend on query density, geometry, hardware, and launch policy.
The customer also reported reused Warp BVHs at 29 ms versus reused cuBQL indexes
at 46 ms for B=32 and N=M=65,536 on A40. That result contrasts with earlier
sparse-query cuBQL wins. It is evidence against choosing a universal winner,
not a prediction for the fused launch policy. Benchmark both backends on the
actual shapes and device; changing reverse targets can additionally make cuBQL
construction more expensive than Warp construction.

.. code-block:: console

    uv run --no-sync -m warp.examples.benchmarks.benchmark_point_query_chamfer \
        --batches 1 8 32 --points 1024 4096 16384 32768 65536 \
        --output chamfer.json --profile

The default inputs are synthetic Gaussian volumes, explicitly labeled in the
JSON. To subsample a supplied pulmonary archive containing ``.npz`` files with
``points`` arrays, add ``--archive /path/to/pulmonary_point_clouds.zip``. Archive
and generated-array hashes identify the measured data. Raw customer coordinates
are not stored in the report. This benchmark measures the loss, not encoder
training or an A40 reproduction on a different GPU.

Pulmonary batch validation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The supplied archive (MD5 ``ba4fee2fdce2d76e8056594c519851c3``) contains nine
clouds, each with at least 65,536 points. The following measurements use B=32
independent target/query subsamples without replacement, cycling through those
clouds. Inputs match exactly between implementations. Hardware is an RTX PRO
6000 Blackwell Server Edition MIG 1g.24gb slice with 46 SMs, rather than the
customer's A40. Both coordinate arrays have gradients enabled; the workload is
the sum of the two mean squared nearest distances, including backward and
gradient clearing. It does not include encoder training.

The reference loads the per-cloud Python implementation from ``ff39bfb`` and
uses the same native binary as the fused implementation from ``739a6c9``.
Native single-cloud traversal is unchanged. Timings are medians of seven
synchronized wall samples after two warmups, excluding compilation and upload.
The fixed target index is reused and the reverse index is rebuilt each call.

.. list-table:: Pulmonary Chamfer forward + backward with reverse construction (ms)
   :header-rows: 1

   * - Points per cloud
     - Per-cloud Warp
     - Fused Warp
     - Fused cuBQL fixed / Warp reverse
   * - 1,024
     - 9.084
     - 2.602
     - 2.573
   * - 4,096
     - 8.961
     - 2.903
     - 2.794
   * - 16,384
     - 10.610
     - 8.346
     - 7.802
   * - 32,768
     - 14.444
     - 13.575
     - 12.456
   * - 65,536
     - 26.699
     - 23.797
     - 21.243

At 1,024 points, a Python profile of Warp's reverse-build Chamfer changes from
66 recursive query calls and 196 ``wp.launch`` calls to two queries and 11
launches. Native construction launches are additional. Per-cloud construction
remains, which limits the improvement once query dispatch is inexpensive.

For reused single-direction queries at 65,536 points, the reference takes
6.500 ms with Warp and 10.994 ms with cuBQL. Fusion and total-batch launch
spacing change these to 5.182 ms and 2.715 ms respectively; graph replay takes
5.148 ms and 2.694 ms. Thus the earlier dense-query crossover is also observed
on these inputs, and the fused scheduling changes that ranking on this GPU.

Construction policy remains a separate consideration. Rebuilding a cuBQL
reverse index with ``BUILD_AND_QUERY`` reduces the 65,536-point Chamfer time
from 169.530 ms (``REUSE``) to 31.357 ms, but the mixed cuBQL/Warp path takes
21.243 ms. The query-only winner is not necessarily the construction-inclusive
winner. These policies are explicit; the API does not automatically switch
backends. To measure the mixed path, add:

.. code-block:: console

    --backends cubql --reverse-backend warp --reverse-objective build_and_query

For native cuBQL in both directions with a construction-oriented reverse tree,
use ``--reverse-backend cubql --reverse-objective build_and_query``. Without
these options the benchmark uses the forward backend and ``REUSE`` in both
directions. Every measured configuration checks all output IDs and counts and
compares up to eight queries per cloud and direction with exhaustive float32
search. These are sampled correctness checks, not a full exhaustive comparison.
