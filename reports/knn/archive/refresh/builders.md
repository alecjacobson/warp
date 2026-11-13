# Warp kNN with LBVH versus cuBQL-built trees

**Historical snapshot.** Measurements, API names, and status below describe the recorded implementation and hardware. Use the [current comparisons](../../comparisons.md) for this branch. Archived reproduction commands assume their original checkout and paths.

The colored tables add **Warp kNN + LBVH (matched control)** and **Warp kNN + cuBQL SAH**. Both use the same `wp.point_query_knn_batch()` implementation, which calls Warp's native `wp.point_query_knn()` traversal and top-k selection. Only the requested BVH constructor differs:

```python
config = wp.point_query_config(len(points), len(queries), points.device)
bvh = wp.Bvh(points, points, constructor="lbvh", leaf_size=config.leaf_size)
# Or: constructor="cubql"
_, ids, distances_sq, counts = wp.point_query_knn_batch(points, queries, k, bvh=bvh)
```

The benchmark uses int64 output indices and Euclidean distances to match the existing comparison contract; the example above keeps the public API's simpler int32/squared defaults. It explicitly constructs the BVH for both variants, including construction inside the one-shot timer. The requested leaf threshold, query stride, block size, ordering policy and compiled query kernels match. This is a controlled builder comparison, not an independently tuned cuBQL integration.

**Warp's `constructor="cubql"` enables cuBQL's SAH builder**, builds a temporary cuBQL tree, and converts it into Warp's native node layout. It also copies node information to the host to check depth against Warp's traversal-stack capacity. These steps and temporary-tree retirement are included in build-plus-query timing. This differs from the standalone `cuBQL findKNN` columns, which query cuBQL's own layout and can choose other builders such as ELH or rebin radix. No native implementation or constructor default was changed for this experiment.

The paired measurement finds cuBQL SAH **13.97x slower for build + query** than LBVH, geometrically averaged across 40 cases. Its resulting trees improve reused queries by **1.07x for ordinary calls and 1.09x for query graph replay**, with lower medians in 31/40 and 33/40 cases, respectively. This supports retaining LBVH for inexpensive construction; the query benefit of the cuBQL SAH tree depends on the workload and how often the tree is reused.

[Paired timing table and all 40 case latencies](builders-timings.md) · [Raw measurements](results/warp-builders.json) · [Summary](builders-summary.json) · [Provenance](builders-provenance.json)

## Capture and validation

Reused-index query graphs pass both normal and changed-query checks for **both** builders on all 40 cases. LBVH build-plus-query graphs additionally pass replay after changing reference points and queries. The cuBQL full-build probe fails: upstream SAH construction calls `cudaStreamSynchronize` during capture (CUDA error 900). Its error handler raises SIGINT, so the isolated probe exits with code 130. The timing sweep runs in a fresh process; that failed capture does not contaminate its measurements. This is an observed unsupported path, not a user interruption.

[Capture failure receipt](results/cubql-full-capture.json) · [Original probe log](results/cubql-full-capture.log) · [Validation](builders-verification.json) · [Existing test suite](results/warp-builders-tests.log)

Each measured phase has nine samples after two warmups, using the existing synchronized wall/event timer, comparison environment and Warp-owned blocking stream. Upload, compilation, capture and instantiation are excluded. Graph outputs are preallocated. cuBQL's memory resource is initialized before setting the shared Warp pool-retention policy to 512 MiB. The native binary and query implementation are unchanged.

All earlier result JSONs remain byte-identical. The main colored plots retain the earlier **Warp batch (LBVH, automatic tuning)** timing as their common denominator, so adding columns does not rewrite earlier ratios. The newly measured LBVH column is the matched control for the new cuBQL column; their direct comparison above uses only this fresh paired run. Small differences between the retained LBVH denominator and the new control can reflect run-to-run variation and the explicit-construction wrapper. Neither is silently selected per case.

The retained LBVH build-plus-query measurement takes 8.2% longer in aggregate than this fresh explicit-construction control; reused-query and query-graph differences are only 0.3% and 0.2%. No native algorithm changed, and these cross-run differences are not attributed solely to the constructor. The builder conclusions use the paired fresh measurements.
