# Batched point queries and direct cuBQL: pulmonary validation

**Historical snapshot.** Measurements, API names, and status below describe the recorded implementation and hardware. Use the [current comparisons](../../comparisons.md) for this branch. Archived reproduction commands assume their original checkout and paths.

2026-11-13. Initial implementation measurements for [direct cuBQL (#20)](https://github.com/alecjacobson/warp/issues/20) and [fixed-size batches (#21)](https://github.com/alecjacobson/warp/issues/21). See the [completion report](../../completion/README.md) for the final public API, standalone integration comparison, and resolved initial scope. Measurements and source hashes below describe the initial implementation.

## Results

- Direct cuBQL with reuse tuning wins ordinary reused queries in 24/24 workloads: **1.37×** geometric-mean speedup over the Warp batch route. Query-only graph replay wins 22/24: **1.21×**.
- Warp has the lowest build-plus-query median in 23/24 cases. The remaining million-point case is effectively tied (0.9378 ms Warp, 0.9376 ms construction-tuned cuBQL). Retain Warp for inexpensive construction and complete build/refit capture.
- Using cuBQL only for fixed targets and Warp for the changing decoded index wins 20/21 pulmonary Chamfer cases, with **1.05×** geometric-mean full forward/backward speedup. This modest end-to-end gain is smaller than the query-only gain.
- All 72 backend/workload combinations pass structural and sampled exhaustive-reference checks, including changed-query graph replay. Across original and changed queries, 5,952 rows were exhaustively checked; tie-aware recall is 1.0 and rank/distance mismatch counts are zero. This is sampled numerical verification, not exhaustive validation of every output row.
- The final native build and all **85** tests in the point-query, BVH, and comparison-harness modules pass on CPU/CUDA. New tests cover batch isolation, empty/partial outputs, native dispatch, changed-input capture, nondefault streams, rebuilding, and analytic/finite-difference gradients for both squared and Euclidean distances.

### Dense fixed targets

For B=4, M=379,385 full-resolution targets per batch, N=8,192 decoded queries per batch, k=1:

| Configuration | Build + query (ms) | Reused query (ms) | Query graph (ms) | Symmetric forward + backward (ms) |
| --- | ---: | ---: | ---: | ---: |
| Warp in both directions | 1.3352 | 0.2993 | 0.2119 | 4.0412 |
| cuBQL reuse tuning; cuBQL reverse | 46.1210 | 0.2025 | 0.1475 | 13.6849 |
| cuBQL construction tuning; cuBQL reverse | 2.7890 | 0.5090 | 0.4511 | 13.7622 |
| cuBQL fixed targets; Warp reverse | — | — | — | 3.9689 |

Both Chamfer directions are included: decoded-to-target uses retained fixed indexes; target-to-decoded constructs a fresh decoded index each step. Each direction is averaged by its own point count and B, followed by backward into decoded coordinates. The cuBQL reverse uses construction tuning. Switching just the reverse direction to Warp cuts the dense case from 13.68 ms to 3.97 ms, locating the main cost in the changing-index direction. Upstream cuBQL construction and index retirement synchronize, and the frozen policy has not been retuned for these dense reverse queries. Pure cuBQL is therefore not the default training recipe.

### Retained memory

| Workload | Warp index + outputs (MiB) | cuBQL reuse index + outputs (MiB) | Input coordinates (MiB) | Pairwise float32 tensor alone (GiB, theoretical) |
| --- | ---: | ---: | ---: | ---: |
| largest_full_b2 | 60.97 | 49.39 | 8.87 | 23.16 |
| largest_full_b4 | 121.94 | 98.79 | 17.74 | 46.31 |
| mixed_subset_b4_m65536 | 21.38 | 17.38 | 3.38 | 8.00 |

Measured memory is the increase in CUDA default-pool **used** bytes after construction/query and synchronization, retaining one index set and output set. It excludes input coordinates, transient construction scratch, gradients, other training buffers, and driver/JIT allocations. It is **not peak memory** and does not measure `torch.cdist`. The theoretical comparison is exactly `4*B*N*M` bytes. Doubling B from two to four doubles retained index/output storage; the implementation uses O(BM + BNk), with linear construction/query-order scratch and no pairwise matrix.

## Inputs and method

- Device: NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb, **94 SMs**, 47.4 GiB reported device memory; CUDA toolkit 12.9.1, driver CUDA 13.2. Historical reports used a 46-SM slice; their absolute timings are not combined with this run.
- Warp 1.20.0.dev0 built with `uv run --no-sync build_lib.py --quick --cuda-path /home/horde/cuda/12.9.1 --jobs 4`. Source and native-binary SHA256 values are recorded in `results.json`.
- Supplied archive: nine float32 pulmonary clouds with 82,734–379,385 points. Archive metadata identifies surface-triangle centroids, original scale in centimeters, no downsampling, vessel walls/outlet caps included, inflow excluded. Raw data stays outside Git.
- Archive SHA256: `8a16347cc942777083a0c763e049a693af452a2c7bbb07ad7e8511a1457889a4`.
- Eighteen singleton cases use every supplied target point, with N=1,024 and 8,192. Full-cloud B=2/4 cases repeat the largest target geometry with independently generated queries. The mixed B=4 case explicitly samples 65,536 points from each of four different clouds to form a non-ragged batch. These distinct-patient original clouds are ragged and are not silently padded or presented as a full-resolution uniform batch.
- Queries sample target coordinates and add independent Gaussian noise with sigma 0.01 in source units, seed 20261113. Three synthetic uniform-volume cases cover (M,N,k)=(1,000,32,32), (100,000,1,024,32), and (1,000,000,8,192,11).
- Each timing is the median of nine synchronized wall-clock samples after two warmups. Compilation and uploads are excluded. Ordinary calls include Python dispatch and output allocation; repeated build timings include retirement of the prior returned index. Graph replay reuses preallocated outputs. Each configuration runs sequentially; no simultaneous benchmark jobs were launched.
- The application explicitly retains up to 512 MiB in the default CUDA memory pool. The library route does not change that threshold.
- All rows receive count/index-range/finite/deduplication/order checks. Per batch, 32 evenly spaced query rows receive exhaustive float64 distance-rank and reported-distance checks, with the existing comparison harness tolerances. Replayed graphs are checked after shifting queries by 0.003 cm.
- Small synthetic optimization examples reduce the same symmetric loss from 0.310435 to 0.157861 using Warp, pure cuBQL, and cuBQL-fixed/Warp-reverse combinations.

## Reproduce

```bash
uv run --no-sync -m warp.examples.benchmarks.benchmark_point_query_batched \
  --archive pulmonary_point_clouds.zip \
  --output reports/knn/batched-cubql/results.json
uv run --no-sync -m warp.examples.geometry.example_point_query_chamfer \
  --device cuda:0 --backend cubql
uv run --no-sync -m unittest warp.tests.geometry.test_point_query_batched \
  warp.tests.geometry.test_point_query_batch warp.tests.geometry.test_point_query \
  warp.tests.geometry.test_bvh warp.examples.benchmarks.test_point_query_comparison
```

## Follow-up identified in the initial run

- The same-device standalone comparison is now recorded in the [completion report](../../completion/README.md). The frozen settings match the prior independently selected builder/leaf/launch policies; current data does not establish universal fastest settings.
- Broaden shape/device coverage and investigate dense-query tuning. Two graph cases remain slower with cuBQL:

| Case | Warp graph (ms) | cuBQL reuse graph (ms) |
| --- | ---: | ---: |
| 0116_H_PULM_ALGS_full_q8192 | 0.0425 | 0.0460 |
| 0117_H_PULM_ALGS_full_q8192 | 0.0429 | 0.0471 |

- Measure transient/peak memory and compare an actual training workload, including its Torch interop if needed. The current retained-memory results should not be substituted for peak training memory.
- Review cross-platform compilation and index lifetime/capture contracts before upstream submission. Native cuBQL supports CUDA, k<=64, unbounded queries without exclusions, and implementation-dependent ties. Unsupported requests fail explicitly.
- Consider fusing launches across B if larger batch counts justify it. The first implementation uses one tree/query operation per batch; query graphs reduce replay overhead without changing that design.

## Artifacts

- [Raw timings, checks, and source hashes](results.json)
- [Final 85-test log](validation.log.gz)
- [Final native build log](build.log.gz)
- [Documentation build with warnings as errors](docs.log.gz)
- [Benchmark execution log](benchmark.log.gz)
- [Validation environment and additional source hashes](validation.json)

Authored by Codex.
