# Warp array queries and fused scalar kernels

- **Status:** Implemented through `warp.geometry.PointIndex` and `warp.geometry.knn`; the original top-level wrapper is deprecated
- **Updated:** 2026-11-15
- **Related design:** [Target ownership, native cuBQL, and independent batches](point-query-backends-and-batches.md)

## Motivation

This extends the branch's scalar point-query design with a host convenience wrapper. The scalar functions, native BVH and selection implementation remain unchanged. The batch layer uses Warp Python and kernels, with no Torch or external kNN dependency.

## Design

`wp.geometry.knn(points, queries, k, ...)` returns `(PointIndex, indices, distances, counts)`. A concrete `wp.geometry.PointIndex(points, ...)` owns reusable target coordinates and indexes; supplying it as the first argument reuses them without rebuilding or refitting. The default backend owns ordinary `wp.Bvh` objects, exposed through `index.bvhs` for fused kernels. Single clouds and independent fixed-size batches use the same owner type. Callers can supply outputs, including int64 IDs. Distances are squared by default; exclusions, cutoffs, and padding follow the scalar contract. Keep the owner, inputs, and outputs alive during asynchronous work or graph replay.

The earlier `wp.point_query_knn_batch`, `wp.knn`, and `wp.build_point_index` entry points preserve their signatures and raw index/tuple returns with deprecation warnings. New callers use `import warp as wp` and explicitly `import warp.geometry`; see the [migration guide](../reports/knn/public-api.md#migrate-the-earlier-branch-api).

`wp.geometry.point_query_config(M, N, device)` exposes the shared leaf-size, query-stride, block-size and ordering policy. The batch wrapper applies it automatically; fused caller kernels can use the same stride mapping and block size while consuming `wp.point_query_knn()` directly. Neither `wp.Bvh`'s generic defaults nor `wp.launch`'s global defaults change. A scalar builtin cannot rearrange unrelated caller threads, so sparse launch mapping remains explicit in fused kernels. The configured benchmark now delegates to this public policy instead of carrying a separate copy.

GPU query ordering uses a bounded-worker bounds reduction, 30-bit Morton codes and the existing Warp radix-sort primitive. It computes bounds from queries rather than reaching into the native BVH descriptor. This avoids adding a new native getter or native snippet. Query order affects scheduling only; result rows remain in their original order. CPU batches default to dense traversal.

Selection and distance output are fused for ordinary forward execution. When input gradients are requested, a separate recorded gather computes distances from the selected indices; selection and ordering are not differentiated or replayed for gradients. This avoids mutable iterator/counter state in the adjoint while preserving the fast forward-only path. Native scalar fused-loss gradients remain supported as before.

Scratch allocated during capture receives dependency-tracked free nodes from Warp. Outside capture, an external nonblocking stream adds a device-side dependency to the default stream before temporary/output retirement; it does not synchronize the host. Normal Warp blocking streams need no additional fence. Construction/refit plus query capture is validated after modifying both inputs. A separate stress probe covers repeated calls on an external nonblocking stream.

The policy originated from tuning on one 46-SM Blackwell MIG slice. [The current comparison report](../reports/knn/comparisons.md) separates that historical tuning from later 94-SM measurements and the API/main regression check. Explicit overrides remain available. The wrapper does not change global allocator retention settings. The benchmark's historical `warp_default` label refers to leaf size 4 and dense 256-thread blocks; the generic `wp.Bvh` constructor actually defaults to leaf size 1. Documentation makes that distinction explicit.
