# Current cross-library timings

See [comparisons](../../comparisons.md) for the colored tables and methodology. [cases.csv](cases.csv) contains every median, ratio, and check status. All ratios below are method time divided by the public `backend=wp.PointQueryBackend.WARP` time; values above 1 favor that reference. Geometric means include only passing paired cases.

## Build + query, ordinary calls

| Method / public settings | Passing pairs | Method / reference geometric mean | Method wins |
| --- | ---: | ---: | ---: |
| `wp.point_query_knn()`; default `wp.Bvh()` | 40/40 | 3.124x | 0 |
| `backend=wp.PointQueryBackend.WARP` | 40/40 | 1.000x | 0 |
| `backend=wp.PointQueryBackend.WARP`, `bvh=wp.Bvh(..., constructor="cubql")` | 40/40 | 14.718x | 0 |
| `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE` | 40/40 | 12.181x | 0 |
| `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.BUILD_AND_QUERY` | 40/40 | 1.871x | 2 |
| PhysicsNeMo, pinned PR implementation | 40/40 | 5.231x | 0 |
| Standalone cuBQL, custom top-k adapter | 40/40 | 6.722x | 0 |
| Standalone cuBQL `findKNN`, default settings | 40/40 | 7.766x | 0 |
| Standalone cuBQL `findKNN`, frozen tuning | 40/40 | 1.767x | 0 |
| CuPy `KDTree` | 40/40 | 49.876x | 0 |
| KeOps `Kmin_argKmin` | 40/40 | 22.202x | 0 |
| Faiss `GpuIndexFlatL2` | 5/40 | 1.535x | 3 |
| cuVS brute force, expanded squared L2 | 3/40 | 0.482x | 3 |
| cuVS brute force, direct squared L2 | 40/40 | 2.869x | 4 |
| cuVS brute force, Minkowski p=2 | 40/40 | 49.840x | 4 |
| PyTorch3D `knn_points` | 40/40 | 24.438x | 0 |

## Reused-index query, ordinary calls

| Method / public settings | Passing pairs | Method / reference geometric mean | Method wins |
| --- | ---: | ---: | ---: |
| `wp.point_query_knn()`; default `wp.Bvh()` | 40/40 | 4.225x | 0 |
| `backend=wp.PointQueryBackend.WARP` | 40/40 | 1.000x | 0 |
| `backend=wp.PointQueryBackend.WARP`, `bvh=wp.Bvh(..., constructor="cubql")` | 40/40 | 0.957x | 31 |
| `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE` | 40/40 | 0.723x | 40 |
| `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.BUILD_AND_QUERY` | 40/40 | 1.106x | 16 |
| PhysicsNeMo, pinned PR implementation | 40/40 | 11.120x | 0 |
| Standalone cuBQL, custom top-k adapter | 40/40 | 4.823x | 0 |
| Standalone cuBQL `findKNN`, default settings | 40/40 | 4.650x | 0 |
| Standalone cuBQL `findKNN`, frozen tuning | 40/40 | 0.649x | 40 |
| CuPy `KDTree` | 40/40 | 52.841x | 0 |
| KeOps `Kmin_argKmin` | 40/40 | 47.198x | 0 |
| Faiss `GpuIndexFlatL2` | 5/40 | 1.780x | 3 |
| cuVS brute force, expanded squared L2 | 3/40 | 0.553x | 3 |
| cuVS brute force, direct squared L2 | 40/40 | 6.020x | 4 |
| cuVS brute force, Minkowski p=2 | 40/40 | 105.373x | 4 |
| PyTorch3D `knn_points` | 40/40 | 51.950x | 0 |

## Build + query, CUDA graph replay

| Method / public settings | Passing pairs | Method / reference geometric mean | Method wins |
| --- | ---: | ---: | ---: |
| `wp.point_query_knn()`; default `wp.Bvh()` | 40/40 | 3.559x | 0 |
| `backend=wp.PointQueryBackend.WARP` | 40/40 | 1.000x | 0 |
| `backend=wp.PointQueryBackend.WARP`, `bvh=wp.Bvh(..., constructor="cubql")` | 0/40 | — | 0 |
| `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE` | 0/40 | — | 0 |
| `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.BUILD_AND_QUERY` | 0/40 | — | 0 |
| PhysicsNeMo, pinned PR implementation | 40/40 | 4.539x | 0 |
| Standalone cuBQL, custom top-k adapter | 0/40 | — | 0 |
| Standalone cuBQL `findKNN`, default settings | 0/40 | — | 0 |
| Standalone cuBQL `findKNN`, frozen tuning | 0/40 | — | 0 |
| CuPy `KDTree` | 0/40 | — | 0 |
| KeOps `Kmin_argKmin` | 0/40 | — | 0 |
| Faiss `GpuIndexFlatL2` | 0/40 | — | 0 |
| cuVS brute force, expanded squared L2 | 3/40 | 0.279x | 3 |
| cuVS brute force, direct squared L2 | 40/40 | 3.101x | 4 |
| cuVS brute force, Minkowski p=2 | 40/40 | 58.755x | 4 |
| PyTorch3D `knn_points` | 40/40 | 30.075x | 0 |

## Reused-index query, CUDA graph replay

| Method / public settings | Passing pairs | Method / reference geometric mean | Method wins |
| --- | ---: | ---: | ---: |
| `wp.point_query_knn()`; default `wp.Bvh()` | 40/40 | 5.652x | 0 |
| `backend=wp.PointQueryBackend.WARP` | 40/40 | 1.000x | 0 |
| `backend=wp.PointQueryBackend.WARP`, `bvh=wp.Bvh(..., constructor="cubql")` | 40/40 | 0.922x | 32 |
| `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE` | 40/40 | 0.805x | 40 |
| `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.BUILD_AND_QUERY` | 40/40 | 1.453x | 9 |
| PhysicsNeMo, pinned PR implementation | 40/40 | 12.303x | 0 |
| Standalone cuBQL, custom top-k adapter | 40/40 | 7.200x | 0 |
| Standalone cuBQL `findKNN`, default settings | 40/40 | 6.925x | 0 |
| Standalone cuBQL `findKNN`, frozen tuning | 40/40 | 0.806x | 40 |
| CuPy `KDTree` | 0/40 | — | 0 |
| KeOps `Kmin_argKmin` | 0/40 | — | 0 |
| Faiss `GpuIndexFlatL2` | 0/40 | — | 0 |
| cuVS brute force, expanded squared L2 | 3/40 | 0.371x | 3 |
| cuVS brute force, direct squared L2 | 40/40 | 8.440x | 4 |
| cuVS brute force, Minkowski p=2 | 40/40 | 159.565x | 4 |
| PyTorch3D `knn_points` | 40/40 | 81.515x | 0 |
