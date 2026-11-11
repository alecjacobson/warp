# Reproducing the broad comparison

Run from the Warp repository root. The comparison adds optional benchmark dependencies only; Warp itself does not depend on these packages. Keep downloaded datasets, external checkouts, builds, and virtual environments outside Git.

## Environment

This run uses Python 3.12.15, Torch 2.7.1+cu128, Warp's CUDA 12.9 native build, and an RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb slice (46 SMs, sm120, driver 595.91.07). Start with the native build in [the original reproduction instructions](../reproduce.rst).

The isolated `.venv-knn-compare` reuses Torch from the original `.venv`. A fresh environment with the same Torch wheel is also suitable. On this machine:

```sh
uv venv --python .venv/bin/python --system-site-packages .venv-knn-compare
uv run --no-sync python - <<'PY'
from pathlib import Path
base = Path('.venv/lib/python3.12/site-packages').resolve()
site = Path('.venv-knn-compare/lib/python3.12/site-packages')
(site / 'base-warp-env.pth').write_text(f'import site; site.addsitedir({str(base)!r})\n')
PY
uv pip install --python .venv-knn-compare/bin/python \
  numpy==2.5.4 cupy-cuda12x==14.2.0 cuvs-cu12==26.10.0 \
  pykeops==2.3 plyfile==1.1.5 ninja iopath swig==4.5.1 'cmake>=3.30,<4'
```

[The supplemental environment freeze](environment-freeze.txt) records the comparison venv packages; Torch and other inherited base-environment packages are identified separately in the result metadata.

The environment launcher resolves the comparison environment's CUDA libraries and preloads its matching cuBLAS/cuBLASLt pair. This matters when reusing Torch from a venv with older CUDA libraries. It also supplies the toolkit's unversioned NVRTC library for KeOps. It changes only the launched process environment.

The PyPI `faiss-gpu-cu12==1.14.1.post1` wheel failed with CUDA error 209 on sm120. The measured Faiss baseline uses the official source build below, not that wheel. The metadata may still list that installed distribution; `provider.imported_path`, `provider.imported_version`, and `faiss_revision` identify the code actually imported.

## Native baseline builds

Install OpenBLAS development headers/libraries through your system package manager. Use a complete CUDA 12.9 toolkit, including cuBLAS, cuRAND and CUDA Profiler API headers. The minimal toolkit previously used for Warp lacked the latter headers; this run supplemented it with NVIDIA's official profiler API 12.9.79 archive (SHA256 `8c50636bfb97e9420905aa795b9fa6e3ad0b30ec6a6c8b0b8db519beb9241ce6`) and CUDA wheel include directories.

```sh
git clone https://github.com/facebookresearch/pytorch3d.git ../third_party/pytorch3d
git -C ../third_party/pytorch3d checkout e4ad6e5687a360ccdc6ebd91348e8dbbac97bf31
git clone https://github.com/facebookresearch/faiss.git ../third_party/faiss
git -C ../third_party/faiss checkout 5622e93733b64b2e033362dbdfda019b2ab33ef0
```

Set `CUDA_HOME` to your toolkit and include CUDA wheel headers in `CPATH` when the toolkit is minimal. Build PyTorch3D with `FORCE_CUDA=1`, `TORCH_CUDA_ARCH_LIST=12.0`, `MAX_JOBS=2`, and `CUB_HOME=$CUDA_HOME/include`, running the original Torch environment's Python `setup.py build_ext --inplace` from its checkout. The benchmark imports this checkout via `--pytorch3d-path`; no upstream code was modified.

Faiss configuration (substitute your absolute CUDA toolkit and comparison Python paths):

```sh
.venv-knn-compare/bin/cmake -S ../third_party/faiss -B ../third_party/faiss/build-knn -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_CUVS=OFF -DFAISS_ENABLE_MKL=OFF \
  -DFAISS_ENABLE_PYTHON=ON -DFAISS_ENABLE_EXTRAS=OFF -DBUILD_TESTING=OFF -DFAISS_OPT_LEVEL=generic \
  -DSWIG_EXECUTABLE="$PWD/.venv-knn-compare/bin/swig" \
  -DPython_EXECUTABLE="$PWD/.venv-knn-compare/bin/python"
.venv-knn-compare/bin/cmake --build ../third_party/faiss/build-knn --target swigfaiss -j 2
```

With cuBLAS supplied by wheels, additionally set `CUDA_cublas_LIBRARY`, `CUDA_cublasLt_LIBRARY`, and the cuBLAS include path in `CMAKE_CUDA_FLAGS`; retain those includes in `CPATH` for host compilation. Run `setup.py build` in `../third_party/faiss/build-knn/faiss/python` to assemble its importable `faiss/` package; pass that directory through `--faiss-path`. See the checked-in build logs for the actual toolchain configuration.

Build direct cuBQL using the existing [original instructions](../reproduce.rst). The comparison records its loaded library SHA256. PhysicsNeMo is fetched at the existing pinned revision and verified by its existing adapter.

## Data and benchmark

```sh
uv run --no-sync --with numpy==2.5.3 --with plyfile==1.1.5 \
  -m warp.examples.benchmarks.point_query_datasets \
  --source-dir ../data/knn-comparison --output ../data/knn-comparison/pilot --download --suite pilot

uv run --no-sync -m warp.examples.benchmarks.point_query_comparison_environment \
  --python .venv-knn-compare/bin/python --cuda "$CUDA_HOME" -- \
  --manifest ../data/knn-comparison/pilot/manifest.json \
  --output ../artifacts/comparison/pilot --repeats 5 --timeout 3600 \
  --cubql-library ../artifacts/point_query_cubql.so \
  --pytorch3d-path ../third_party/pytorch3d \
  --faiss-path ../third_party/faiss/build-knn/faiss/python

uv run --no-sync -m unittest warp.examples.benchmarks.test_point_query_comparison -q
```

`--suite smoke` prepares 32 smaller cases. `--methods` selects adapters; `--limit 1 --repeats 2` is a quick installation check. Every method runs sequentially in a fresh process. Graph probes use separate processes to isolate capture failures. Do not run final benchmarks concurrently with compilation or other GPU work.

The original pilot arrays were generated with NumPy 2.5.3; the manifest records this separately from benchmark NumPy 2.5.4. Use the generator version in the manifest for byte-identical reproduction. Array and archive hashes provide semantic identity; compressed NPZ file hashes are also checked before each run.

No source archives or derived point arrays are committed. Stanford, KITTI, and Open3D/Redwood sources and usage notes are recorded in `manifest.json`. KITTI is CC BY-NC-SA 3.0; this is a research benchmark. Download from the original hosts using the provided script.
