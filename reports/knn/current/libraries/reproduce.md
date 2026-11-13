# Reproduce the current cross-library comparison

Run from the Warp checkout. These are optional benchmark dependencies; Warp's public API does not require them. The prepared workspace is `.benchmarks/knn/`, excluded from Git, with one project `.venv`. The existing CUDA toolkit is `/home/horde/cuda/12.9.1`.

## Run the prepared experiment

```bash
# One workload per method, including isolated capture checks.
uv run --no-sync python reports/knn/current/libraries/run.py \
  --workspace .benchmarks/knn --cuda /home/horde/cuda/12.9.1

# All 40 workloads, nine timed samples after warmup, one process at a time.
uv run --no-sync python reports/knn/current/libraries/run.py \
  --workspace .benchmarks/knn --cuda /home/horde/cuda/12.9.1 --full-matrix
```

The first command writes `.benchmarks/knn/smoke/`; the second writes `.benchmarks/knn/run/`. The runner records each command, start/end time, return code, and timeout in `processes.json`. Ordinary calls and each provider's query/full graph workers run in separate processes. Capture failure stops that graph worker; its context is never reused for another timing job. `--only` selects named jobs for diagnosis. A zero worker exit code alone does not establish numerical correctness: inspect every saved validation result.

Do not run timed measurements concurrently with compilation, another GPU benchmark, or a GPU test suite. Restore the same input arrays, output contract, allocator settings, and timing convention when comparing runs.

## Restore optional dependencies

The measured Python package versions are in [environment-freeze.txt](environment-freeze.txt). Installation used the existing project environment:

```bash
uv pip install --python .venv/bin/python 'torch==2.7.1' \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python \
  'cupy-cuda12x==14.2.0' 'cuvs-cu12==26.10.0' 'rmm-cu12==26.10.0' \
  'pykeops==2.3' 'plyfile==1.1.5' ninja iopath 'swig==4.5.1' 'cmake>=3.30,<4'
```

The cuVS environment supplies CUDA 12.9 runtime-library wheels alongside Torch's CUDA 12.8 build. The environment launcher preloads the selected cuBLAS pair and supplies library/toolkit paths to the child process. This combination was exercised by the saved installation checks and complete benchmark. NumPy 2.5.3 generated the datasets and ran this experiment.

The external source builds use these unchanged upstream revisions:

| Dependency | Revision / build |
| --- | --- |
| PhysicsNeMo | `bc1c912528f0409c6853dbd33e00346a9184185c`; the existing adapter substitutes framework imports, preserving the kNN algorithm. |
| PyTorch3D | `e4ad6e5687a360ccdc6ebd91348e8dbbac97bf31`; CUDA build for `sm_120`. |
| Faiss | `5622e93733b64b2e033362dbdfda019b2ab33ef0`; CUDA build for `sm_120`, cuVS integration off, float32 search. |
| cuBQL | Vendored source from this branch; standalone adapters compiled for `sm_120`. |

Reuse existing checkouts before cloning. The prepared source paths are `.benchmarks/knn/third_party/pytorch3d` and `.benchmarks/knn/third_party/faiss`. PyTorch3D is imported from its source build; Faiss is imported from `build-knn/faiss/python`. The result metadata records the actual imported paths and source revisions, rather than relying on installed package names.

Both source builds use the shared CUDA 12.9.1 toolkit, CUDA wheel include directories in `CPATH`, and four build workers. The toolkit includes NVIDIA's SHA256-verified Profiler API 12.9.79 headers. PyTorch3D uses `FORCE_CUDA=1`, `TORCH_CUDA_ARCH_LIST=12.0`, `MAX_JOBS=4`, and `CUB_HOME=/home/horde/cuda/12.9.1/include` with `setup.py build_ext --inplace`.

Faiss uses CMake Release, `CMAKE_CUDA_ARCHITECTURES=120`, `FAISS_ENABLE_GPU=ON`, `FAISS_ENABLE_CUVS=OFF`, `FAISS_ENABLE_MKL=OFF`, `FAISS_OPT_LEVEL=generic`, Python bindings enabled, and extras/tests disabled. Build the `swigfaiss` target with `-j 4`, then run the generated Python package's `setup.py build`. This machine uses Ubuntu OpenBLAS 0.3.26 libraries extracted under `.benchmarks/knn/openblas`; their package checksums and the complete compiler/configuration commands are recorded in [provenance.json](provenance.json) and compressed build logs. cuBLAS libraries and headers come from the environment's CUDA wheels.

The source/native hashes and build logs identify the measured binaries. Rebuild Warp only if its native source changed; this experiment reused its existing quick-build library, whose cuBQL integration uses `compute_75` PTX. The separate [integration experiment](../../completion/README.md) includes a matched-PTX standalone control to isolate that build-target difference.

## Restore data and standalone adapters

```bash
uv run --no-sync -m warp.examples.benchmarks.point_query_datasets \
  --source-dir .benchmarks/knn/data/sources \
  --output .benchmarks/knn/data/pilot --download --suite pilot

/home/horde/cuda/12.9.1/bin/nvcc -std=c++17 -O3 -DNDEBUG \
  --extended-lambda -t 4 -arch=sm_120 -Xcompiler=-fPIC -shared -Iwarp/native \
  warp/examples/benchmarks/point_query_cubql.cu \
  -o .benchmarks/knn/libraries/cubql_custom.so
/home/horde/cuda/12.9.1/bin/nvcc -std=c++17 -O3 -DNDEBUG \
  --extended-lambda -t 4 -arch=sm_120 -Xcompiler=-fPIC -shared -Iwarp/native \
  warp/examples/benchmarks/point_query_cubql_upstream.cu \
  -o .benchmarks/knn/libraries/cubql_upstream.so
```

The generator reuses existing source archives and verifies their checksums. Original Stanford, KITTI, and Open3D/Redwood source URLs and attribution are in [manifest.json](manifest.json). Source archives and derived point arrays remain outside Git. All 40 case IDs, shapes, and coordinate hashes in this run match the historical matrix; compressed-file hashes are checked against the newly generated manifest at runtime.

The standalone tuned route uses the existing independent training selections, with separate construction-inclusive and reused-query settings. Decompress [the preserved tuning record](../../archive/refresh/results/cubql-tuning.json.gz) to `.benchmarks/knn/frozen-cubql-tuning.json`. The policy was trained on the earlier 46-SM slice and is frozen for this 94-SM evaluation. There is no per-case choice of the fastest measured setting. The standalone default route remains a separate column.

## Regenerate the presentation

The committed `results/` directory contains the completed raw measurement JSON files and compressed logs; the root manifest is the dataset manifest used by every worker. No historical timing samples are loaded by the current generator.

```bash
uv run --no-sync python reports/knn/current/libraries/summarize.py
```

This generates four SVG matrices, [tables.md](tables.md), [cases.csv](cases.csv), and [summary.json](summary.json). Each colored cell combines the clean/noisy pair only if both measurements pass the relevant checks. Graph cells require both original-input and changed-input replay validation. Failed checks, unsupported construction capture, failed capture attempts, and unmeasured cases remain explicit gray cells; ordinary times never fill a missing graph cell.

After a new run, review and copy the completed evidence into `results/`, refresh the provenance and written conclusions, then regenerate the presentation. Preserve process failures and numerical failures alongside successful timings. The 24-workload pulmonary/batched experiment uses a different output/timing convention and is reported separately.
