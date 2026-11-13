# Fixed-size kNN and direct cuBQL completion

2026-11-13. Completion evidence for [optional native cuBQL (#20)](https://github.com/alecjacobson/warp/issues/20) and [fixed-size batches (#21)](https://github.com/alecjacobson/warp/issues/21). The initial scope is implemented. Wider hardware tuning, fused batches, peak training memory, and application integration remain follow-up work.

## Public API

```python
# targets: contiguous wp.vec3 array of shape (B, M), or (M,) for one cloud.
# decoded: contiguous wp.vec3 array of shape (B, N), or (N,) for one cloud.
indexes = wp.build_point_index(targets, num_queries=N, k=1, backend=wp.PointQueryBackend.CUBQL)

# Repeat while targets stay fixed; decoded coordinates may change each step.
_, indices, distances_sq, counts = wp.knn(targets, decoded, 1, bvh=indexes)
```

Construction needs no query coordinates or neighbor outputs. `num_queries` and `k` are optional tuning hints; later calls may use different valid values. Batched outputs have shapes `(B, N, k)`, `(B, N, k)`, and `(B, N)`, with batch-local indices. The builder returns a tuple of B indexes; single-cloud inputs return one index and preserve single-cloud output shapes.

Warp remains the default backend. Direct cuBQL is optional and retains its documented CUDA, capacity, filtering, and tie contracts. Existing `wp.point_query_knn_batch` calls work through a deprecated alias with the same signature and returns. The scalar `wp.point_query_knn` primitive is unchanged. The [user guide](../../../docs/user_guide/point_queries.rst) covers graph capture, output reuse, distance gradients, and index lifetime. The Chamfer example builds fixed targets once and defaults to Warp for the changing reverse-direction index.

## Integration comparison

All **230 route/objective/workload combinations** pass structural checks and sampled exhaustive float64 reference comparisons, including replay after changing queries. The checks cover **16,640 sampled rows**, with tie-aware recall 1.0 and zero rank or reported-distance mismatches. Numerical checks are sampled; they do not exhaustively validate every output distance.

The integrated native route preserves standalone traversal performance closely. Its public Python wrapper has measurable ordinary-call overhead:

| Tuning objective | Call mode | Public integrated / matched standalone time, geometric mean | Median B=1 added time |
| --- | --- | ---: | ---: |
| Reuse | Preallocated ordinary call | 2.040× | 40.35 µs |
| Reuse | Allocation-inclusive ordinary call | 1.724× | 42.01 µs |
| Reuse | Query graph replay | 1.010× | 0.29 µs |
| Build + query | Preallocated ordinary call | 1.477× | 41.32 µs |
| Build + query | Allocation-inclusive ordinary call | 1.414× | 42.99 µs |
| Build + query | Query graph replay | 1.014× | 1.39 µs |

These timings exclude construction even for the **build + query tuning objective**: that label selects the construction-oriented tree/launch policy. Construction-inclusive measurements are in the [initial pulmonary report](../batched-cubql/README.md).

The preallocated raw native C API is only 1.005× standalone time with reuse tuning, versus 2.040× through the public wrapper. Public and raw native graph replay differ by 0.15% in geometric mean. This isolates the material ordinary-call difference to the Python wrapper, which validates arrays/options, constructs zero-copy batch views, and dispatches each cloud. It is visible for tiny queries, and graph replay avoids repeating that work. This comparison uses the batched layout even when B=1; it does not measure the single-cloud layout's wrapper cost.

Public graph timings range from 0.998× to 1.026× matched standalone with reuse tuning, and 0.996× to 1.028× with construction tuning. Small residual GPU differences are consistent with the integrated output handling: it validates nonfinite queries, supports int32/int64 indices and squared/Euclidean outputs, and recomputes selected distances from coordinates to match the differentiable gather. Standalone writes the distances retained by upstream selection. This is an explanation from code inspection, not a separately isolated measurement of each added instruction. The traversal itself remains vendored upstream `findKNN`.

Compiling standalone for `sm_120` instead of Warp's quick-build `compute_75` PTX improves reuse graph time by 1.4% in geometric mean, but is 2.3% slower with the construction-oriented policy. Neither build target wins universally; architecture selection does not explain the ordinary public-wrapper gap.

Against the Warp public route, reuse-tuned cuBQL wins 23/23 preallocated ordinary cases (1.40× geometric-mean speedup), 23/23 allocation-inclusive cases (1.33×), and 22/23 graph cases (1.26×). The graph exception is `0115_H_PULM_ALGS_q1024`: 0.02889 ms cuBQL versus 0.02432 ms Warp. Different query seeds and paired sampling mean this run need not reproduce the exact winning cases in the earlier report. The construction-oriented cuBQL policy usually has slower reused queries, reinforcing the distinction between objectives. Warp retains inexpensive construction and full construction/refit capture.

### Dense fixed-target example

B=4, M=379,385, N=8,192, k=1, reuse tuning. All four clouds use the largest original target geometry, with independently sampled queries.

| Route | Preallocated ordinary (ms) | Allocation-inclusive ordinary (ms) | Query graph (ms) |
| --- | ---: | ---: | ---: |
| Public cuBQL | 0.23675 | 0.26626 | 0.15696 |
| Raw integrated native C API | 0.20221 | — | 0.15701 |
| Standalone, matched PTX target | 0.20169 | 0.22027 | 0.15431 |
| Standalone, sm_120 | 0.19989 | 0.21634 | 0.15330 |
| Public Warp | 0.39675 | 0.41735 | 0.23463 |

The [initial report](../batched-cubql/README.md) separately measures construction, symmetric Chamfer forward/backward, and retained memory. Its 98.79 MiB cuBQL index/output footprint plus 17.74 MiB input coordinates compares with a theoretical 46.31 GiB pairwise tensor for this shape. Those are retained-memory measurements, not peak training memory or a measured `torch.cdist` run.

## Method and evidence

- Same NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb, **94 SMs**, CUDA 12.9.1 toolkit and CUDA 13.2 driver. The historical tuning study used a 46-SM slice; no absolute historical timings are mixed into this comparison.
- 23 workloads: four synthetic regimes, all nine original pulmonary target clouds with N=1,024/8,192, and the dense B=4 case. Original clouds contain 82,734–379,385 targets. Each workload runs both tuning objectives and five routes. B=1 retains the leading batch dimension. No hidden padding or ragged batching is used.
- Synthetic `(M,N,k)` settings are `(1000,32,32)`, `(100000,1024,1)`, `(100000,1024,32)`, and `(1000000,8192,11)`. Queries sample target coordinates plus Gaussian noise of sigma 0.01, seed 20261114. Graph validation shifts queries by 0.003.
- All routes use identical float32 coordinates, int64 indices, float32 Euclidean distances, int32 counts, and matched frozen builder/leaf/stride/block settings. The standalone CUDA adapter and vendored cuBQL are unmodified. This benchmark loads that adapter with ctypes and Warp arrays; Torch is unnecessary.
- Median of 31 synchronized wall-clock samples after three warmups. Route order is deterministically randomized within each sampling round. Graph samples average 32 replays. Allocation-inclusive timings allocate outputs inside the timed operation; output retirement occurs after timing. Construction, uploads, compilation, and validation are excluded. No other benchmark or test job was launched concurrently.
- The application retains 512 MiB in the CUDA default pool. Standalone upstream construction changes the pool threshold, so the benchmark restores it before timing queries. The integrated allocation resource does not alter the application's policy.
- [integration.json](integration.json) retains every sample, configuration, check, archive checksum, and source/binary SHA256. [summary.json](summary.json) contains aggregates. The raw archive stays outside Git; SHA256 is `8a16347cc942777083a0c763e049a693af452a2c7bbb07ad7e8511a1457889a4`.

## Validation and native review

- **89 tests pass** across the point-query, batch, BVH, and benchmark-reference modules on CPU/CUDA. Coverage includes native-route assertions, legacy compatibility, query-free construction, reuse without rebuilding, changed query counts/k, isolation, empty/partial outputs, nondefault streams, squared/Euclidean gradients to both inputs, and finite differences.
- The capture test additionally passes after explicitly constructing batched Warp indexes with `build_point_index` inside capture, changing targets, and replaying. It verifies that direct cuBQL build-only construction rejects capture. Reused cuBQL query capture remains supported.
- Full native CUDA build, changed-file pre-commit, and Sphinx build with warnings as errors pass. The updated cuBQL-fixed/Warp-reverse Chamfer example reduces loss from 0.310435 to 0.157861.
- CPU-only `bvh.cpp`, disabled-cuBQL CPU glue, and disabled-cuBQL CUDA translation units compile; symbol inspection confirms the three new native entry points in the relevant objects. These are conditional translation-unit compile checks, **not full alternate-configuration runtime builds**, and do not establish Windows/ARM portability.
- Native construction catches host allocation failure within the C ABI, checks the point-box launch, avoids launch-size integer overflow, and cleans up partial allocations without replacing the original diagnostic. Queries reject missing coordinates for a nonempty native tree before launch. A failed Python-side rebuild preserves the old index. CPU-only creation reports why the route is unavailable.
- Owners retain coordinate storage. Destruction synchronizes before freeing native tree buffers; construction and destruction are outside reused-query timings. Graph users must retain indexes, inputs, and outputs until replay finishes. No claim is made that cuBQL construction or retirement is capturable or synchronization-free.

Compressed logs and `validation.json` preserve the build/test/check evidence. Wider hardware validation, dense reverse-query tuning, fused larger batches, peak training memory, and actual encoder integration are follow-ups, not unresolved initial requirements. Ragged batches remain outside scope.

## Reproduce

From this checkout, with the supplied archive and existing CUDA toolkit:

```bash
uv run --no-sync build_lib.py --quick --cuda-path /home/horde/cuda/12.9.1 --jobs 4
mkdir -p /tmp/warp-knn-integration
/home/horde/cuda/12.9.1/bin/nvcc -std=c++17 -O3 -DNDEBUG --extended-lambda -t 4 \
  -gencode=arch=compute_75,code=compute_75 -Xcompiler=-fPIC -shared -Iwarp/native \
  warp/examples/benchmarks/point_query_cubql_upstream.cu \
  -o /tmp/warp-knn-integration/upstream_matched.so
/home/horde/cuda/12.9.1/bin/nvcc -std=c++17 -O3 -DNDEBUG --extended-lambda -t 4 \
  -arch=sm_120 -Xcompiler=-fPIC -shared -Iwarp/native \
  warp/examples/benchmarks/point_query_cubql_upstream.cu \
  -o /tmp/warp-knn-integration/upstream_sm120.so
uv run --no-sync -m warp.examples.benchmarks.benchmark_point_query_integration \
  --archive pulmonary_point_clouds.zip \
  --matched-library /tmp/warp-knn-integration/upstream_matched.so \
  --native-library /tmp/warp-knn-integration/upstream_sm120.so \
  --output reports/knn/completion/integration.json
uv run --no-sync -m unittest warp.tests.geometry.test_point_query_batched \
  warp.tests.geometry.test_point_query_batch warp.tests.geometry.test_point_query \
  warp.tests.geometry.test_bvh warp.examples.benchmarks.test_point_query_comparison -v
uv run --no-sync build_docs.py --warnings-as-errors
```

Conditional translation-unit checks (create the scratch directory above first):

```bash
g++ -std=c++17 -O3 -fPIC -fno-rtti -DWP_ENABLE_CUDA=0 -DWP_BUILD_DLL \
  -DWP_ENABLE_MATHDX=0 -Iwarp/native -c warp/native/bvh.cpp \
  -o /tmp/warp-knn-integration/bvh_cpu.o
g++ -std=c++17 -O3 -fPIC -fno-rtti -DWP_ENABLE_CUDA=0 -DWP_BUILD_DLL \
  -DWP_DISABLE_CUBQL -DWP_ENABLE_MATHDX=0 -Iwarp/native \
  -c warp/native/bvh_cubql.cpp -o /tmp/warp-knn-integration/cubql_disabled_cpu.o
/home/horde/cuda/12.9.1/bin/nvcc -std=c++17 -O3 -DNDEBUG --extended-lambda -t 4 \
  -gencode=arch=compute_75,code=compute_75 -Xcompiler=-fPIC -DWP_ENABLE_CUDA=1 \
  -DWP_DISABLE_CUBQL -DWP_BUILD_DLL -DWP_ENABLE_MATHDX=0 -Iwarp/native \
  -c warp/native/bvh_cubql.cu -o /tmp/warp-knn-integration/cubql_disabled_cuda.o
nm -g --defined-only /tmp/warp-knn-integration/bvh_cpu.o \
  /tmp/warp-knn-integration/cubql_disabled_cuda.o | rg wp_point_query_cubql
```
