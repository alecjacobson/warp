# Reproducing the Warp / PhysicsNeMo investigation

Use the [comparison environment and dataset preparation instructions](../comparison/reproduce.md). The hardware, dependency pins, raw sources and license notes are unchanged. Run GPU jobs sequentially. Commands below run from the repository root; paths assume the retained workspace layout used for the measurements.

The earlier configured baseline is commit `25cd0218bd3937d826fb85690a09fba948047d9a`. `results/before.json` remeasures its native implementation and configuration on a Warp-owned blocking stream. Final timing JSONs name that parent revision because measurements preceded the commit; their source hashes and [final source hashes](source-sha256.json) identify the changed code. Native intermediate patches apply to the parent and are **rejected experiments**, not required changes. Use a separate checkout to explore them; these context-free patches require `git apply --unidiff-zero`.

The native library uses the quick build (compute_75 PTX, driver-JIT on the sm_120 GPU), matching the previous baseline; Warp query kernels JIT for the device. Other architecture/build targets need their own measurements. [Retained build log](build-final.log.gz)

## Build and tests

```sh
uv run --no-sync build_lib.py --quick --cuda-path /tmp/warp-intersection-cuda-12.9 -j 2
uv run --no-sync -m unittest warp.tests.geometry.test_point_query warp.tests.geometry.test_bvh warp.tests.geometry.test_grouped_bvh warp.tests.test_intersect
uv run --no-sync -m unittest warp.tests.geometry.test_point_query warp.examples.benchmarks.test_point_query_comparison
```

The final native change only adjusts point-tree Morton normalization; no new native query code or third-party native dependency is introduced. The query-spacing policy is in `point_query_common.point_query_configuration` and the configured provider; ordinary dense launches remain available.

## Final 40-case comparison

```sh
uv run --no-sync -m warp.examples.benchmarks.benchmark_point_query_hybrid \
  --manifest ../data/knn-comparison/pilot/manifest.json \
  --baseline-only --physics-isotropic --physics-tuned --repeats 9 \
  --output ../artifacts/hybrid-reproduction/final.json
```

`--baseline-only` excludes the experimental native-snippet combinations; it includes configured Warp and the named PhysicsNeMo variants. The pinned PhysicsNeMo loader uses `../artifacts/comparison/pilot/physicsnemo`, downloading its pinned source if absent. Generated variants retain the standalone shim and receive distinct source files/custom-op names. The baseline is unchanged.

```sh
uv run --no-sync -m warp.examples.benchmarks.point_query_comparison_environment \
  --python .venv-knn-compare/bin/python --cuda /tmp/warp-intersection-cuda-12.9 -- \
  --manifest ../data/knn-comparison/pilot/manifest.json \
  --output ../artifacts/hybrid-reproduction/cuvs --methods cuvs_unexpanded --repeats 9 --timeout 3600
uv run --no-sync -m warp.examples.benchmarks.benchmark_point_query_comparison \
  --worker warp_default --manifest ../data/knn-comparison/pilot/manifest.json \
  --output ../artifacts/hybrid-reproduction/default --repeats 9
```

## Holdouts and a controlled Q sweep

The generator needs `plyfile`, present in the comparison environment. It generates 56 cases from new seeds, reusing a maximum-sized query array through nested prefixes for the million-point sweep. NumPy 2.5.4 generated the retained holdouts; use that version to reproduce their checksums.

```sh
uv run --no-sync -m warp.examples.benchmarks.point_query_comparison_environment \
  --python .venv-knn-compare/bin/python --cuda /tmp/warp-intersection-cuda-12.9 \
  --module warp.examples.benchmarks.point_query_hybrid_holdout -- \
  --source-dir ../data/knn-comparison --output ../data/knn-comparison/hybrid-holdout
uv run --no-sync -m warp.examples.benchmarks.benchmark_point_query_hybrid \
  --manifest ../data/knn-comparison/hybrid-holdout/manifest.json \
  --baseline-only --physics-isotropic --physics-tuned --repeats 9 \
  --output ../artifacts/hybrid-reproduction/holdout.json
```

## Diagnostics

```sh
uv run --no-sync -m warp.examples.benchmarks.probe_point_query_hybrid_capture \
  --output ../artifacts/hybrid-reproduction/capture.json
uv run --no-sync -m warp.examples.benchmarks.profile_point_query_hybrid \
  --output ../artifacts/hybrid-reproduction/profile
uv run --no-sync -m warp.examples.benchmarks.probe_point_tree \
  --case thin_plane-n100000-q1024-sigma0-k32 \
  --output ../artifacts/hybrid-reproduction/counts-thin.json
uv run --no-sync -m warp.examples.benchmarks.benchmark_point_query_hybrid \
  --blocks --physics-tuned \
  --filter volume-n100000-q1024-sigma0-k32,sphere-n1000000-q8192-sigma0-k11,volume-n1000000-q8192-sigma0-k11 \
  --output ../artifacts/hybrid-reproduction/launches-controlled.json
```

The counter probe compares identical point coordinates in aliased versus separate lower/upper arrays, enabling or disabling the native isotropic point-tree path; its kernels count traversal iterations, primitive distances and candidate insert calls. Do not use instrumented timings as performance measurements. CUPTI traces likewise diagnose phases; unprofiled wall medians establish latency claims. Decompress the `.json.gz` traces before opening in Perfetto/Chrome.

`--sweep` evaluates leaf size, ordering, packed reference positions, seed bounds and selector choices. `--isotropic` is an exploratory sentinel-padding proof with sentinels excluded from results. `--morton-seed`, `--carry` and `--cached` examine Morton initialization and candidate retention. They are benchmark-only native snippets, have a narrower input contract than the public API, and are not selected production paths. Intermediate snapshots and measurements are retained to explain rejected approaches; final-source reruns do not recreate every intermediate compiler state bit for bit.

After copying fresh results into the report's `results/` directory:

```sh
uv run --no-sync reports/knn/hybrid/summarize.py
uv run --no-sync reports/knn/hybrid/plot.py
```

The summarizer validates the recorded correctness/capture fields and regenerates the CSV, tables and summary. The plot requires Matplotlib. All ratios are reference latency divided by configured Warp latency. The 5% band is a descriptive near-tie band, not a confidence interval. Primary timings are synchronized wall times; CUDA-event samples are retained separately.
