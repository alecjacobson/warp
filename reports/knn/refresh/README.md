# GPU kNN comparison — fresh Warp timings, November 12, 2026

**The highest-performing Warp path on this 40-case matrix is the automatically tuned batch API replayed as a CUDA graph.** It is **3.33x faster for build + query and 5.68x faster for reused-index queries** than the simple scalar ordinary-call path, geometrically averaged. The tuned scalar graph is close: the batch graph leads it by only 0.8% / 4.3% in aggregate, and does not win every case. These small differences are not a statistical significance claim.

Only Warp was remeasured for this update. The nine competing configurations retain their previous samples byte-for-byte; their implementations and dependencies did not change. The ordinary-call library comparison still gives the automatic Warp batch API **36/40 wins** in both timing modes. The four losses are the tiny N=1,000, Q=32 cases. Using Warp graph replay against those retained ordinary-call competitor measurements also gives 36/40; it is not a graph-to-graph library comparison.

## What the Warp labels mean

| Label | Actual call and setup |
| --- | --- |
| **Warp scalar (simple dense launch)** | `wp.Bvh(points, points)` with its actual default leaf size **1**, followed by a caller kernel using `wp.point_query_knn()`; one thread per query, block size 256, no query reordering. |
| **Warp batch (automatic tuning)** | `wp.point_query_knn_batch()` chooses the BVH leaf size and launch policy. The report states whether it is an ordinary call or CUDA graph replay. |
| Warp scalar (tuned launch) | Diagnostic control: the same scalar primitive with explicit leaf/launch tuning, matching the earlier winning benchmark policy. This is not the simple usage. |

The old “Warp default” measurement explicitly set leaf size 4; it was **not** literal `wp.Bvh` default usage. The new simple-scalar measurements use leaf size 1. Consequently, speedups against this new baseline should not be interpreted as additional algorithm improvements over the previous report. The optimized native implementation has not changed.

The automatic policy uses leaf size 32 for Q≤32 or N≥1M, otherwise 4; it spreads sparse queries across warps based on the GPU's SM count. All 40 cases here select the sparse, unordered path. Large dense batches can also select GPU query ordering, covered by the separate [batch holdout measurements](../batch/README.md).

## Ordinary calls versus CUDA graph replay

These are fresh measurements on the same inputs and GPU. Values are geometric-mean speedups over **Warp scalar (simple dense launch), ordinary call**; greater than one is faster.

| Warp path | Execution | Build + query | Reused-index query |
| --- | --- | ---: | ---: |
| Warp scalar (simple dense launch) | Ordinary call | 1.00x | 1.00x |
| Warp scalar (simple dense launch) | CUDA graph replay | 1.08x | 1.17x |
| Warp scalar (tuned launch) | Ordinary call | 2.85x | 4.04x |
| Warp scalar (tuned launch) | CUDA graph replay | 3.31x | 5.44x |
| **Warp batch (automatic tuning)** | Ordinary call | **2.81x** | **3.90x** |
| **Warp batch (automatic tuning)** | **CUDA graph replay** | **3.33x** | **5.68x** |

Graph replay makes the automatic batch path **1.18x / 1.45x faster than its own ordinary calls**. The earlier few-percent wrapper penalty applies to ordinary calls: here the batch API takes 1.4% / 3.6% longer than manually tuned scalar calls. Captured execution bypasses repeated Python validation and per-kernel host submission. It still executes the GPU work; no per-case choice of fastest implementation is used in these tables.

Build-plus-query graphs rebuild the BVH on every replay. Reused-index graphs query an existing BVH without rebuilding or refitting it. Graph output buffers are allocated before capture. Capture, instantiation, compilation and input upload are excluded; the first two replays are warmups. Each timed sample includes one replay, event recording and device synchronization. This is single-invocation wall latency, not amortized throughput across hundreds of queued replays. GPU event samples are also retained.

[Every case's latency](warp-timings.md) · [All samples and replay checks](results/warp-replay.json) · [CSV, including graph columns](cases.csv) · [Warp provenance and unchanged competitor hashes](warp-provenance.json)

## Comparison with other libraries: ordinary calls

The denominator here is **Warp batch (automatic tuning), ordinary call**, so execution modes match the retained competitor timings. Ratios are **competitor latency / Warp latency**; greater than one favors Warp. Numerical failures are excluded per case when choosing a competitor.

| Regime | Cases | Warp wins, build + query | Warp wins, reused index | Best competitor / Warp, build + query | Best competitor / Warp, reused index |
| --- | ---: | ---: | ---: | ---: | ---: |
| N=39K/100K, Q=1,024, k=1 | 16 | 16 | 16 | 2.29–4.19x | 1.42–2.91x |
| N=39K/100K, Q=1,024, k=32 | 16 | 16 | 16 | 1.44–2.96x | 2.25–6.22x |
| N=1M, Q=8,192, k=11 | 4 | 4 | 4 | 1.09–1.16x | 1.46–1.83x |
| N=1K, Q=32, k=32 | 4 | 0 | 0 | 0.42–0.50x | 0.32–0.61x |

All 36 ordinary-call wins exceed 5%. Tiny cases still favor direct-L2 cuVS or Faiss. The following geometric means include only cases where both methods pass the numerical checks; different denominators must not be treated as comparable coverage.

| Alternative | Warp wins, build + query | Alternative / Warp GM | Warp wins, reused index | Alternative / Warp GM |
| --- | ---: | ---: | ---: | ---: |
| Pinned PhysicsNeMo | 40/40 | 4.37x | 40/40 | 10.43x* |
| Direct cuBQL | 40/40 | 5.93x | 40/40 | 4.44x |
| CuPy KDTree | 40/40 | 46.27x | 40/40 | 50.95x |
| KeOps | 40/40 | 18.43x | 40/40 | 43.92x* |
| Faiss Flat | 2/5 | 1.77x | 2/5 | 2.19x |
| cuVS expanded L2 | 0/3 | 0.50x | 0/3 | 0.57x |
| cuVS direct L2 | 36/40 | 3.36x | 36/40 | 7.89x |
| cuVS Minkowski p=2 | 36/40 | 77.73x | 38/40 | 185.16x |
| PyTorch3D | 40/40 | 20.20x | 40/40 | 48.14x* |

*These adapters expose a complete operator, not a reusable index. Their full call is compared with Warp's reused-index query; this is not an isolated traversal-kernel ratio.

[Accuracy-filtered table](table.md) · [All timings, including numerical failures](all-timed-table.md) · [Reused-index table](query-table.md) · [Best competitor per build-plus-query case](best-one_shot.json) · [Best competitor per reused-index case](best-reused_index_search.json)

![Ordinary-call build-plus-query comparison](one-shot.svg)

![Ordinary-call reused-index comparison](reused-query.svg)

## Correctness, capture, and scope

All three fresh Warp paths pass checks on all 40 cases for ordinary calls and graph replay, both build-plus-query and reused-index search. Every graph is also replayed after changing queries; build-plus-query graphs additionally change reference points. Checks cover all-row structure and float64 exhaustive reference distances for 32 evenly spaced query rows, not every large-batch query. Native scalar and batch tests, including gradients and ordered capture, remain separate evidence. [Validation log](results/warp-replay-tests.log)

The unchanged competitor findings remain: Faiss Flat passes 5/40 cases and expanded-L2 cuVS passes 3/40 under the stricter numerical contract. All other tested exact competitors pass 40/40. Failures remain visible in raw results. CuPy KDTree and KeOps fail the earlier query-capture probe. Those probes are capture checks, not graph performance measurements. CAGRA remains excluded at the user's request and was not run.

The original equally tuned PhysicsNeMo control is retained unchanged: 36 clear Warp wins and four million-point near ties within 5%, with its own contemporaneous Warp denominator. It does not use the new batch/graph times. See [control data](results/tuned-control.json) and [environment](results/tuned-control.environment.json). The advantage over the pinned PhysicsNeMo PR should not be presented as a robust million-point advantage over that equally tuned variant.

The current Warp source revision is `4e0a854e920377817e8c528fbb28ca49afaa0b03`; the new benchmark script's exact hash is recorded separately because measurements preceded its commit. The native binary is unchanged from the original refresh. The same comparison environment, CUDA toolkit, shared blocking stream, nine-sample timer and int64/float32 Euclidean output contract are retained. Warp now allocates outputs as Warp arrays instead of wrapping Torch allocations; the API itself has no Torch dependency, although the shared benchmark timer uses Torch. Warp pool retention remains 512 MiB. This is an API/execution-mode comparison, not a pure kernel ablation.

The GPU is one 46-SM RTX PRO 6000 Blackwell MIG slice in a shared VM, with unlocked clocks. Retained competitor samples were collected earlier on this same setup, not simultaneously with these Warp timings. Small run-to-run differences should not be overinterpreted. A 5% band is descriptive, not a confidence interval. Results do not establish rankings on other GPUs, high-dimensional embeddings or arbitrary k/N/Q.

The frozen cases include scan vertices, LiDAR, RGB-D, surfaces, volumes, clusters/outliers and thin planes, with and without Gaussian noise. [Manifest](manifest.json) · [Dataset sources](../comparison/README.md#dataset-coverage) · [Reproduce](reproduce.md). The [previous full refresh report](https://github.com/alecjacobson/warp/blob/4e0a854e920377817e8c528fbb28ca49afaa0b03/reports/knn/refresh/README.md) and its original raw Warp samples remain available for historical comparison.
