# cuBQL: custom adapter, upstream defaults, and tuned settings

**Historical snapshot.** Measurements, API names, and status below describe the recorded implementation and hardware. Use the [current comparisons](../../comparisons.md) for this branch. Archived reproduction commands assume their original checkout and paths.

The colored tables retain the original cuBQL measurement and add two independent columns:

| Column | Implementation and configuration |
| --- | --- |
| **cuBQL custom top-k (leaf 4, dense)** | Original retained results. cuBQL's spatial-median builder and shrinking-radius traversal, with our selection callback matching Warp's distance arithmetic, exclusions and deterministic ties. Leaf size 4; one thread per query; 256 threads per block. |
| **cuBQL findKNN (defaults)** | New measurements calling unmodified `cuBQL::points::findKNN()` and `gpuBuilder()` with `BuildConfig()`. For this vendored version the default builder is spatial median and resolves the leaf threshold to 1. |
| **cuBQL findKNN (best tested settings)** | The same upstream `findKNN()`, with builder, leaf threshold and query launch settings selected on separate training inputs. Fixed policies depend on the workload regime and whether construction is included. |

cuBQL supplies a device query function, not a host batch operator with prescribed launch dimensions. The defaults column therefore uses an explicit conventional dense launch of 256 threads per block. Both new wrappers write int64 indices and float32 Euclidean distances directly to GPU output arrays. That conversion is included in the kernel. The original adapter used separate Torch conversions from int32 indices and squared distances; it also compiled with `--fmad=false` for its custom distance contract. The new wrapper uses the normal CUDA compiler defaults and upstream distance calculations. Differences from the old column therefore include adapter and arithmetic changes, not just tuning.

The exact vendored cuBQL headers are unchanged. We do not replace `findKNN()`'s candidate list, distance function, tie behavior or traversal. Equal-distance indices may differ from Warp; the existing comparison oracle is tie-aware and does not require identical IDs for equivalent distances. Numerical failures, if any, are retained and excluded from accuracy-filtered speed rankings.

[Native wrapper](../../../../warp/examples/benchmarks/point_query_cubql_upstream.cu) · [Benchmark and tuning search](../../../../warp/examples/benchmarks/benchmark_point_query_cubql.py) · [Upstream query implementation](../../../../warp/native/cuBQL/queries/pointData/knn.h) · [Upstream candidate management](../../../../warp/native/cuBQL/queries/common/knn.h)

## Tuning method

“Best tested settings” means the winner of this documented hierarchical search, not a claim of a global optimum. Training uses seed **20261112**, independent of the original 40 evaluation cases. Each of four regimes has clean/noisy sphere and uniform-volume samples: N=1K/Q=32/k=32; N=100K/Q=1,024/k=1; N=100K/Q=1,024/k=32; N=1M/Q=8,192/k=11. The same deterministic dataset generator and normalization/noise definitions are used. No scan-specific settings are fitted to the evaluation data.

1. Screen five upstream builders—spatial median, radix, rebin radix, SAH and ELH—at leaf thresholds 0, 1, 2, 4, 8, 16, 32 and 64. Zero retains the builder's own choice. The initial launch uses the existing GPU-aware spacing heuristic.
2. Keep the two best tree configurations for build-plus-query and the two best for reused queries, plus the default tree. For each, measure all combinations of query stride 1/2/4/8/16/32, block size 32/64/128/256, and ordering on/off. Ordering uses GPU bounds, Morton keys and radix sorting; its complete cost is timed.
3. Recheck the three best candidates per objective, plus defaults, using nine samples and the full 32-row oracle. Measure actual build-plus-query calls for final selection, including retirement of previous trees. Select a single configuration per regime and objective by geometric-mean latency across its four training inputs.
4. Freeze those settings before evaluating the original 40 cases. No per-evaluation-case minimum or retrospective setting changes are used.

Screening uses five samples after two warmups and two exhaustive reference rows per input. Finalist selection and evaluation use nine samples, all-row structure checks and 32 exhaustive reference rows. The two-row screens are preliminary filters, not the final correctness evidence. The finalist query and build-plus-query objectives may select different trees; both choices are reported in [selected settings](cubql-settings.md).

The search is not exhaustive over the full Cartesian product: launch tuning follows tree screening, and only finalists receive end-to-end construction timing. Wide BVHs, other precision/types, alternate selection algorithms and other hardware are outside the search. Capture-specific launch settings are not tuned independently.

## Capture, memory, and timing

Build-only samples describe construction of the reused-query configuration; one-shot timings independently use the build-plus-query configuration.

The two new variants use the same comparison environment, shared Warp-owned blocking stream, event/wall timer, output contract and input data as the existing report. Host-to-device input upload, compilation and tuning are excluded from evaluation timings. Output allocation, wrapper submission, query ordering and result conversion are included in ordinary calls.

cuBQL's vendored builders read construction state back to the host and synchronize internally; destruction also synchronizes. Those costs remain in build-plus-query timing. We do not claim capture of upstream construction. Reused-index queries are captured and checked after changing query values, with separate replay samples and preallocated output buffers. The colored plots show ordinary-call timings, not replay timings.

Both new variants retain cuBQL's own default asynchronous memory resource, which sets the CUDA default pool's release threshold to its maximum. Warp output arrays use the existing Warp pool setup. This is not an equal-memory-budget comparison. The old cuBQL adapter also used the default cuBQL memory resource.

The original cuBQL column and all existing Warp/other-library samples are preserved unchanged. [Provenance](cubql-provenance.json) records the vendored tree, compiler command, library/source hashes and retained-result hashes. [Full tuning samples](results/cubql-tuning.json.gz) · [Defaults results](results/cubql_defaults.json) · [Tuned results](results/cubql_tuned.json) · [Reproduction](reproduce.md)

[Ordinary and graph timing summary](cubql-timings.md) compares the execution modes explicitly.
