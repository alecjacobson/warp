# Current kNN comparisons and recommended settings

**2026-11-13 · `alecjacobson/knn` · completed public API.** Start with `backend=wp.PointQueryBackend.WARP` when targets change often. For repeated CUDA queries against fixed targets, start with `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE`. For symmetric Chamfer, measure that fixed-target choice with `backend=wp.PointQueryBackend.WARP` in the reverse direction.

All three configurations below use **Warp's public `wp.build_point_index()` and `wp.knn()` APIs**. The settings select implementations and tuning policies; callers do not need a separate cuBQL Python library. See the [public API guide](public-api.md) for complete examples and contracts.

## Which settings should I use?

| Workload or requirement | Public arguments | Current evidence or contract |
| --- | --- | --- |
| Construct an index and query it once | `backend=wp.PointQueryBackend.WARP` | Lowest construction-plus-query median in **23/24** workloads. |
| Repeated queries against fixed CUDA targets | `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE` | Faster ordinary queries in **24/24**, with **1.37×** geometric-mean speedup; query graphs win **22/24**, averaging **1.20×**. Construction must be amortized. |
| Evaluate cheaper cuBQL construction | `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.BUILD_AND_QUERY` | Lowest construction-plus-query median in **1/24** workloads, by 2.3% in that case. Usually slower reused queries than `objective=wp.PointQueryObjective.REUSE`. |
| CPU, finite cutoff, self-exclusion, original-index tie ordering, or k > 64 | `backend=wp.PointQueryBackend.WARP` | Supports these requirements. The direct cuBQL route rejects unsupported requests. |
| Capture index construction or refit | `backend=wp.PointQueryBackend.WARP` | Construction/refit capture supported with CUDA memory pools enabled. Both backends support querying prebuilt indexes in graphs. |
| Symmetric Chamfer with fixed targets | Fixed: `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE`; reverse: `backend=wp.PointQueryBackend.WARP` | Faster forward/backward steps in **19/21** k=1 cases, averaging **1.06×** over `backend=wp.PointQueryBackend.WARP` both ways. |
| Fuse neighbor processing inside your kernel | Build with `backend=wp.PointQueryBackend.WARP`; use `wp.point_query_knn()` and `while wp.point_query_next(...)` | Scalar queries accept `wp.Bvh.id`. Direct `for index in search` iteration is not implemented. |

The backend choice is explicit. `objective` tunes cuBQL; it does not switch backends or change the default implementation's policy. Build the fixed target index once, then pass `bvh=indexes` on subsequent calls. Supplied indexes keep their original settings. Leave `leaf_size`, `query_stride`, and `ordered` automatic initially. The cuBQL route requires CUDA, memory-pool support, and a build with cuBQL enabled; it supports k=1–64.

## Latest workload measurements

This fresh run uses the completed API and native binary at [93e506474](https://github.com/alecjacobson/warp/commit/93e506474cdc0e10e57b875a60e986f6f5fbf7f9). It supersedes the earlier construction/Chamfer report, which predates the API cleanup. Measurements cover **24 workloads**, each with all three configurations, on the **94-SM RTX PRO 6000 Blackwell MIG 2g.48gb** slice. [All timings](current/timings.md), [raw samples](current/batched.json), and [aggregates](current/summary.json) are retained.

![Per-workload speedup for fixed targets](current/query-speedups.svg)

The chart compares `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE` against `backend=wp.PointQueryBackend.WARP`; values above 1 favor the former. Query graphs lose on `0116_H_PULM_ALGS_full_q8192` (0.04609 versus 0.04262 ms) and `0117_H_PULM_ALGS_full_q8192` (0.04751 versus 0.04466 ms). These are observed medians, not statistical guarantees of a winner.

### Dense fixed-target example

**B=4, M=379,385, N=8,192, k=1.** Each batch uses the largest original pulmonary target cloud and independently sampled queries. All times are median milliseconds; lower is better.

| Public arguments | Construct + query | Reused ordinary query | Reused query graph |
| --- | ---: | ---: | ---: |
| `backend=wp.PointQueryBackend.WARP` | **1.34463** | 0.30096 | 0.21059 |
| `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE` | 46.37292 | **0.20496** | **0.14676** |
| `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.BUILD_AND_QUERY` | 2.83211 | 0.51076 | 0.45455 |

Reuse tuning buys query speed with substantially more expensive construction. Compare total time over the expected number of calls, including index retirement and the chosen output-allocation/capture mode. A query-only advantage does not establish a benefit for short-lived indexes.

The construction exception across the full suite is the synthetic M=1,000,000, N=8,192, k=11 case: `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.BUILD_AND_QUERY` takes **0.9313 ms**, versus **0.9535 ms** for `backend=wp.PointQueryBackend.WARP`. This narrow measured win is workload-specific.

### Symmetric Chamfer: measure both directions and backward

For the same dense example, fixed target construction is paid before timing. Every step constructs the reverse index on the changing decoded points, queries both directions, reduces squared distances using the sum-of-means convention, and runs Warp's backward pass.

| Fixed target arguments | Changing reverse target arguments | Forward + backward (ms) |
| --- | --- | ---: |
| `backend=wp.PointQueryBackend.WARP` | `backend=wp.PointQueryBackend.WARP` | 4.05286 |
| `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE` | `backend=wp.PointQueryBackend.WARP` | **3.96831** |
| `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE` | `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.BUILD_AND_QUERY` | 13.73926 |

The mixed choice improves this dense step by about **2%**, despite the larger fixed-target query improvement. Across all 21 k=1 workloads it wins 19, with a 1.06× geometric-mean speedup. This measures the Chamfer computation and its gradients, not an encoder, decoder, optimizer, or complete training iteration. The [runnable example](../../warp/examples/geometry/example_point_query_chamfer.py) uses Warp's tape; sharing Torch storage does not automatically connect the two autograd systems.

### Retained memory

For the dense example, live index/output allocations retain **121.94 MiB** with `backend=wp.PointQueryBackend.WARP`, or **98.79 MiB** with `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE`. Input coordinates add **17.74 MiB** in either case. A float32 B×N×M distance tensor would require **46.31 GiB**.

These are live CUDA-pool bytes retained after construction and query. They exclude inputs from the index/output count and exclude transient construction scratch, tape storage, and peak training memory. The pairwise figure is arithmetic, not a measured `torch.cdist` allocation. For fixed k, index/output storage scales as `O(B*M + B*N*k)`; the API does not materialize a B×N×M tensor.

## Does the public cuBQL route retain standalone speed?

Yes for GPU query graphs, within about **1% in geometric mean** of the matched standalone implementation. Ordinary calls have additional public-wrapper work. The retained [integration comparison](completion/README.md) uses exactly the current recorded source files and native binary; [provenance](current/provenance.json) verifies every saved source/binary hash. It remains applicable and was not rerun for this report cleanup.

| `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE` | Public time / matched standalone time | Median added time, B=1 |
| --- | ---: | ---: |
| Ordinary query, preallocated outputs | 2.040× | 40.35 µs |
| Ordinary query, output allocation included | 1.724× | 42.01 µs |
| Query graph replay | 1.010× | 0.29 µs |

The raw native preallocated call averages 1.005× standalone time. The public wrapper validates inputs/options, creates batch views, and dispatches each cloud. Capturing the query avoids repeating that Python work. These B=1 measurements retain the leading batch dimension; they do not measure the single-cloud layout's wrapper overhead.

This is a **separate 23-workload experiment** with 31 interleaved samples, int64 indices, Euclidean distances, and 32 replays per graph sample. The fresh 24-workload results above use int32 indices, squared distances, nine sequential samples, and one replay per graph sample. Do not pool their timings or geometric means. The integration experiment times queries only, including when `objective=wp.PointQueryObjective.BUILD_AND_QUERY` selects the tree policy.

## Method, correctness, and remaining limits

- Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb, **94 SMs**, approximately 47.4 GiB available; CUDA 12.9.1 toolkit, driver supporting CUDA 13.2. The existing quick-build native binary was reused. No other test or benchmark job was launched by this agent during the run.
- Inputs: nine supplied original pulmonary clouds with **82,734–379,385** float32 points, original centimeter scale. Each runs with N=1,024 and 8,192. Two cases repeat the largest cloud with B=2/4; one explicitly samples four different clouds to M=65,536 for a non-ragged B=4 batch. Three synthetic `(M,N,k)` cases are `(1000,32,32)`, `(100000,1024,32)`, and `(1000000,8192,11)`. Different-size original clouds are measured separately, with no implicit padding.
- Query generation: seed 20261113, target samples plus independent Gaussian noise of sigma 0.01 in source units. Changed-query graph checks shift coordinates by 0.003.
- Timing: median of nine synchronized wall-clock samples after two warmups, with variants measured sequentially. Construction-plus-query includes previous result/index retirement. Ordinary reuse includes new output allocation and prior output retirement. Graph timing includes one replay and synchronization, using preallocated outputs. Uploads and compilation are excluded. The application retains 512 MiB in the default CUDA pool. Small differences may reflect scheduling, allocation, or run-order effects.
- Fresh numerical checks: all **72 workload/configuration combinations** pass, covering **5,952 sampled rows** against exhaustive reference searches before and after query changes. Tie-aware recall is 1.0; duplicate, ordering, rank, and reported-distance mismatches are zero. Chamfer losses and decoded gradients are finite, and mixed-route losses agree within the harness tolerance. These checks do not exhaustively compare every output or every gradient.
- Retained completion evidence: **89 focused CPU/CUDA tests passed**, including finite-difference gradients and explicit route checks. The recorded source hashes still match. Saved build, strict documentation, and conditional translation-unit checks are in the [completion report](completion/README.md); those compile checks do not establish full alternate-platform runtime support.
- Current scope: fixed-size independent batches, reusable target indexes, both query graph routes, and scalar kernel queries. No ragged batches. Larger-batch fusion, wider hardware tuning, peak training memory, and application integration remain follow-ups.

There is no refreshed claim here against cuVS, Faiss, PhysicsNeMo, or other external libraries. Those comparisons used earlier implementations and the **46-SM** slice; they are available in the [historical archive](archive/README.md). Their absolute timings must not be compared directly with this run.

## Reproduce and inspect

From the repository root, with the supplied dataset and an already built native library matching the source:

```bash
uv run --no-sync -m warp.examples.benchmarks.benchmark_point_query_batched \
  --archive pulmonary_point_clouds.zip \
  --output reports/knn/current/batched.json
uv run --no-sync python reports/knn/current/summarize.py
```

The summarizer needs Matplotlib and regenerates the aggregate JSON, per-case tables, and SVG from saved samples without using the GPU. When taking a new benchmark snapshot, update the report conclusions and provenance as well. Dataset SHA256: `8a16347cc942777083a0c763e049a693af452a2c7bbb07ad7e8511a1457889a4`; the archive remains outside Git.

Evidence: [benchmark source](../../warp/examples/benchmarks/benchmark_point_query_batched.py), [raw results](current/batched.json), [run log](current/benchmark.log.gz), [summary](current/summary.json), [source-hash audit](current/provenance.json), and [standalone reproduction](completion/README.md#reproduce).
