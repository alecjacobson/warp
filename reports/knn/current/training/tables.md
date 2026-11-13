# Batched query and Chamfer timings

All ratios are method / the matching public `backend=wp.PointQueryBackend.WARP` reference. Values above 1 favor the reference. Only passing paired cases enter geometric means. Passing counts differ; geometric means over different subsets are not head-to-head rankings. See [the report](../../comparisons.md#batched-queries-and-chamfer-against-external-libraries), [every median/status](cases.csv), and [reproduction](reproduce.md).

## Reused-target query

| Configuration | Passing pairs | Method/reference geometric mean | Method wins |
| --- | ---: | ---: | ---: |
| `backend=wp.PointQueryBackend.WARP` | 24/24 | 1.000x | 0 |
| `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE` | 24/24 | 0.737x | 24 |
| `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.BUILD_AND_QUERY` | 24/24 | 1.174x | 4 |
| `torch.cdist(compute_mode="use_mm_for_euclid_dist")` | 0/24 | — | 0 |
| `torch.cdist(compute_mode="donot_use_mm_for_euclid_dist")` | 18/24 | 2948.592x | 1 |
| Public PyTorch3D automatic kernel selection | 24/24 | 84.500x | 0 |
| Public `knn_points(version=...)`, selected on independent synthetic inputs | 24/24 | 73.072x | 0 |
| Public batched `LazyTensor` reductions; selected-pair autograd for Chamfer | 24/24 | 58.706x | 0 |

## Symmetric Chamfer forward + backward

| Configuration | Passing pairs | Method/reference geometric mean | Method wins |
| --- | ---: | ---: | ---: |
| Fixed and reverse: `backend=wp.PointQueryBackend.WARP` | 21/21 | 1.000x | 0 |
| Fixed: `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE`; reverse: `backend=wp.PointQueryBackend.WARP` | 21/21 | 0.974x | 21 |
| Fixed: `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE`; reverse: `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.BUILD_AND_QUERY` | 21/21 | 2.450x | 0 |
| `torch.cdist(compute_mode="use_mm_for_euclid_dist")` | 0/21 | — | 0 |
| `torch.cdist(compute_mode="donot_use_mm_for_euclid_dist")` | 16/21 | 1639.609x | 0 |
| `torch.cdist` direct mode under `torch.no_grad()`; gather selected pairs, reduce squared distances, backward | 16/21 | 1022.752x | 0 |
| `pytorch3d.loss.chamfer_distance(norm=2, batch_reduction="mean", point_reduction="mean")` | 21/21 | 19.027x | 0 |
| Public `knn_points(version=...)`, selected on independent synthetic inputs | 21/21 | 15.833x | 0 |
| Public batched `LazyTensor` reductions; selected-pair autograd for Chamfer | 21/21 | 14.704x | 0 |
