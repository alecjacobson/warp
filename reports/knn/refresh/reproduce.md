# Reproducing the fresh full-library comparison

This reruns the [earlier broad comparison](../comparison/README.md) after the [geometry and scheduling improvements](../hybrid/README.md), using the same 40 immutable cases and pinned dependencies. Run commands from the repository root. Downloaded sources, virtual environments, datasets and native baseline builds remain outside Git.

The measured implementation is commit `197ffeeea97ed94352a01ee67c3c852accb92731`. [Provenance](provenance.json) records the source and native library SHA256 hashes. The existing native quick build uses compute_75 PTX, driver-JIT on the sm_120 device; query kernels JIT for sm_120. No native code or adapter algorithms were changed for this rerun. The report itself is committed afterward.

Follow the [original environment, source-build and dataset instructions](../comparison/reproduce.md) if starting from a fresh checkout. The retained `.venv-knn-compare` contains the original versions; [environment freeze](environment-freeze.txt) lists its packages, while each result JSON also identifies inherited Torch and imported source builds. [Hardware](hardware.txt) records the same 46-SM RTX PRO 6000 Blackwell MIG slice and driver. This rerun holds library versions fixed; it is not an upgrade to newer releases.

```sh
uv run --no-sync -m warp.examples.benchmarks.point_query_comparison_environment \
  --python .venv-knn-compare/bin/python --cuda /tmp/warp-intersection-cuda-12.9 -- \
  --manifest ../data/knn-comparison/pilot/manifest.json \
  --output ../artifacts/comparison/refresh-2026-11-12 --repeats 9 --timeout 3600 \
  --methods warp_default warp_configured physicsnemo cubql cupy_kdtree keops \
    faiss_flat cuvs_brute_force cuvs_unexpanded cuvs_minkowski pytorch3d \
  --cubql-library ../artifacts/point_query_cubql.so \
  --pytorch3d-path ../third_party/pytorch3d \
  --faiss-path ../third_party/faiss/build-knn/faiss/python
```

The runner launches all 11 exact configurations sequentially in isolated processes. Every configuration has a separate query-capture probe followed by its timing worker. It writes process outcomes, complete samples, accuracy measurements, provider metadata and error traces. It checks every input file hash against the [frozen manifest](manifest.json). Never run GPU benchmarks concurrently with builds or other GPU jobs.

Two warmups precede nine samples per phase. Primary times are device-synchronized wall medians, including host submission, output allocation/conversion, sorting and retirement of prior outputs. One-shot measurements include index construction; reused-index measurements exclude construction. PhysicsNeMo, KeOps and PyTorch3D expose only complete operators in these adapters, so their full calls remain in the reused-index comparison. Upload and JIT compilation are excluded; CUDA-event samples are supplemental.

All providers share a Warp-owned blocking stream where their API permits; this preserves Warp temporary-allocation lifetime dependencies. This is the same stream policy as the final hybrid report, replacing the original broad report's Torch-created nonblocking stream. Warp retains 512 MiB in its memory pool; cuVS uses a matching RMM release threshold, while other provider allocator policies remain unchanged. These are explicit policies, not equal-memory-budget measurements.

The configured Warp path uses the existing GPU-aware query spacing and leaf policy from `point_query_common.point_query_configuration`. The default path remains dense with leaf size 4. No per-dataset tuning is performed during this run. PhysicsNeMo is the same pinned PR, and the main matrix does not silently substitute the tuned experimental variant.

Validate the harness and regenerate the report after copying the result JSONs into `results/`:

```sh
uv run --no-sync -m unittest warp.examples.benchmarks.test_point_query_comparison -q
uv run --no-sync reports/knn/refresh/summarize.py
uv run --no-sync reports/knn/refresh/plot.py
```

The scripts require all 11 methods, all 40 case IDs, and nine repetitions. They preserve numerical failures, exclude them from accuracy-filtered rankings. The 5% band is descriptive, not a statistical confidence interval. The SVG plots group clean/noisy pairs; gray cells mean at least one case failed sampled checks. PNG previews are written under `/tmp`.

Validation checks all output rows for shape, bounds, finiteness, duplicates and sorted distances, plus 32 evenly spaced query rows against an exhaustive float64 distance oracle. Passing these checks is not exhaustive validation of every large-batch query, nor a mathematical real-arithmetic exactness claim. Graph probes use a separate small input and changed queries; they do not establish capture of every library's index construction.

CAGRA was removed at the user's request before its worker started. Stopping the original coordinator also interrupted the active Minkowski worker after 35 completed cases. Those samples were preserved; the remaining five cases were run with an explicit subset manifest and identical settings. The original and resumed JSONs, subset manifest, scope-change record and logs are retained under `results/segments/` and `results/resume/`. The aggregate Minkowski report merges completed cases without averaging or replacing samples; it records both segments. PyTorch3D ran afterward in its own sequential coordinator. A fresh reproduction can use the uninterrupted 11-method command above.

## Supplemental tuned-PhysicsNeMo control

After the main sweep, the same comparison environment runs a separate nine-sample control with configured Warp, pinned PhysicsNeMo and the explicitly labeled tuned PhysicsNeMo variant. Ratios for this control use its own contemporaneous Warp measurements; they are not mixed into the main matrix.

```sh
uv run --no-sync -m warp.examples.benchmarks.point_query_comparison_environment \
  --python .venv-knn-compare/bin/python --cuda /tmp/warp-intersection-cuda-12.9 \
  --module warp.examples.benchmarks.benchmark_point_query_hybrid -- \
  --manifest ../data/knn-comparison/pilot/manifest.json \
  --baseline-only --physics-tuned --repeats 9 \
  --output ../artifacts/comparison/refresh-2026-11-12/tuned-control.json
uv run --no-sync -m warp.examples.benchmarks.point_query_comparison_environment \
  --python .venv-knn-compare/bin/python --cuda /tmp/warp-intersection-cuda-12.9 \
  --module warp.examples.benchmarks.probe_point_query_hybrid_capture -- \
  --output ../artifacts/comparison/refresh-2026-11-12/full-capture.json
```

The latter probe captures complete build-plus-query calls for Warp and three PhysicsNeMo variants, replaying after both reference points and queries change. This is stronger than the main matrix's query-only capture probe, but is still a separate small-case check. The generated PhysicsNeMo source hashes identify the experimental variants; see the [hybrid report](../hybrid/README.md) for their exact changes and patches.
