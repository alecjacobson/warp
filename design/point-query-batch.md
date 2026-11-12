# Warp-only batch point queries

- **Status:** Implemented on the kNN feature branch
- **Date:** 2026-11-12

## Motivation

This extends the branch's scalar point-query design with a host convenience wrapper. The scalar functions, native BVH and selection implementation remain unchanged. The batch layer uses Warp Python and kernels, with no Torch or external kNN dependency.

## Design

`wp.point_query_knn_batch(points, queries, k, ...)` returns the ordinary tuple `(bvh, indices, distances, counts)`. The existing `wp.Bvh` owns acceleration storage; there is no new point-cloud owner or public query-result record. An optional existing BVH is reused without implicit refitting. Callers can also supply outputs, including int64 indices. Default distances are squared, and exclusions/cutoffs/padding follow the scalar contract. Callers retain the BVH, input arrays and outputs for asynchronous work or graph replay.

`wp.point_query_config(N, Q, device)` exposes the shared leaf-size, query-stride, block-size and ordering policy. The batch wrapper applies it automatically; fused caller kernels can use the same stride mapping and block size while consuming `wp.point_query_knn()` directly. Neither `wp.Bvh`'s generic defaults nor `wp.launch`'s global defaults change. A scalar builtin cannot rearrange unrelated caller threads, so sparse launch mapping remains explicit in fused kernels. The configured benchmark now delegates to this public policy instead of carrying a separate copy.

GPU query ordering uses a bounded-worker bounds reduction, 30-bit Morton codes and the existing Warp radix-sort primitive. It computes bounds from queries rather than reaching into the native BVH descriptor. This avoids adding a new native getter or native snippet. Query order affects scheduling only; result rows remain in their original order. CPU batches default to dense traversal.

Selection and distance output are fused for ordinary forward execution. When input gradients are requested, a separate recorded gather computes distances from the selected indices; selection and ordering are not differentiated or replayed for gradients. This avoids mutable iterator/counter state in the adjoint while preserving the fast forward-only path. Native scalar fused-loss gradients remain supported as before.

Scratch allocated during capture receives dependency-tracked free nodes from Warp. Outside capture, an external nonblocking stream adds a device-side dependency to the default stream before temporary/output retirement; it does not synchronize the host. Normal Warp blocking streams need no additional fence. Construction/refit plus query capture is validated after modifying both inputs. A separate stress probe covers repeated calls on an external nonblocking stream.

The policy is a workload heuristic measured on one 46-SM Blackwell MIG slice. Explicit overrides remain available. The wrapper does not change global allocator retention settings. The benchmark's historical `warp_default` label refers to leaf size 4 and dense 256-thread blocks; the generic `wp.Bvh` constructor actually defaults to leaf size 1. Documentation makes that distinction explicit.
