.. note:: Historical snapshot. Use the current comparison report in ``reports/knn/comparisons.md``. Commands below assume the original checkout and paths.

Reproduce the point-query measurements
=====================================

This temporary report bundle records measurements of implementation commit
``5e2f400d063b14f7034b7802746547e8a5fda748``, based on NVIDIA Warp commit
``18c936d4390e3ff1a91eb7a7ecc9668e425e2c52``. Read ``README.md`` for the full
report, or open ``report.html`` locally. All commands below run from the
repository root. Write fresh measurements outside this saved report directory.

Build and validate
------------------

Replace ``CUDA_TOOLKIT`` with your CUDA toolkit directory::

    uv run --no-project --with numpy build_lib.py --quick --cuda-path CUDA_TOOLKIT
    uv sync --extra dev
    uv run warp/tests/geometry/test_point_query.py
    uv run -m unittest warp.tests.geometry.test_bvh warp.tests.geometry.test_mesh_query_point warp.tests.geometry.test_grouped_bvh -q
    uv run -m warp.examples.geometry.example_point_query --device cuda:0

The recorded build used CUDA 12.9 and driver 595.91.07 / CUDA driver API 13.2
on an NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb slice (46 SMs).
The first build uses ``--no-project`` because editable installation requires
native binaries. The measured implementation passes 78 CPU/CUDA tests and
60 additional direct-cuBQL comparisons; the saved logs describe those runs.

Data and dependencies
---------------------

Create output directories and download the Open3D scan fragment archive::

    mkdir -p ../knn-results ../knn-data
    curl -L --fail https://github.com/isl-org/open3d_downloads/releases/download/20220301-data/DemoICPPointClouds.zip -o ../knn-data/DemoICPPointClouds.zip
    uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

Use ``uv run --no-sync`` below to retain the optional PyTorch installation.
The harness downloads the PhysicsNeMo source at commit
``bc1c912528f0409c6853dbd33e00346a9184185c`` and replaces only two framework
imports with a minimal stream/validation adapter. It records source hashes.

The scan consists of preprocessed RGB-D XYZ rows, without resampling, centered
and scaled by the longest bounding-box edge. Synthetic data includes unit-sphere
surface samples and uniform volume samples. Each family runs with and without
additional Gaussian noise (sigma=0.01). The report documents query generation,
sampling, output representations, and timing boundaries.

Direct cuBQL baseline
--------------------

Compile the separate benchmark library against Warp's vendored cuBQL headers.
Change ``sm_120`` for your GPU::

    CUDA_TOOLKIT/bin/nvcc -std=c++17 -O3 -DNDEBUG --fmad=false --extended-lambda -arch=sm_120 -Xcompiler=-fPIC -shared -Iwarp/native warp/examples/benchmarks/point_query_cubql.cu -o ../knn-results/point_query_cubql.so

The selection callback implements strict cutoffs, exclusion, deterministic ties,
and matching float32 accumulation. ``--fmad=false`` keeps cuBQL bounds compatible
with the noncontracting distance helper. The callback widens its pruning bound
by one ULP to preserve equal-distance candidates in cuBQL's strict traversal.

Benchmark runs
--------------

The full GPU comparison includes the pinned PhysicsNeMo and direct cuBQL
baselines, with seven repeated timing samples::

    uv run --no-sync -m warp.examples.benchmarks.benchmark_point_query --output ../knn-results/results.json --scan ../knn-data/DemoICPPointClouds.zip --physicsnemo --cubql-library ../knn-results/point_query_cubql.so --tune --repeats 7

Supplemental tile, motion, and memory measurements and the CPU matrix::

    uv run --no-sync -m warp.examples.benchmarks.benchmark_point_query_tuning --output ../knn-results/tuning.json
    uv run --no-sync -m warp.examples.benchmarks.benchmark_point_query --device cpu --quick --repeats 3 --output ../knn-results/results-cpu.json

The saved ``results.json``, ``results-cpu.json``, and ``tuning.json`` contain
individual timing samples and validation counts. Query-only GPU measurements
use batches of 20 captured executions. The main comparison uses int64 indices
and Euclidean float32 distances for both implementations. One-shot measurements
include construction, planning, allocations, host submission, and query work;
host-to-device input transfers are separate. The benchmark validates up to 32
evenly spaced query rows per timed variant against exhaustive float32 and
float64 oracles. Unit tests validate all rows on smaller adversarial inputs.

The recorded 30/42 one-shot wins and 1.45x geometric mean apply only to this
matrix and GPU. Million-point one-shot cases remain slower than the PR.
The report includes those losses and the limits of the comparison.
