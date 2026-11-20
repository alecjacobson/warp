# Tiled brute-force kNN and Chamfer

On the supplied pulmonary point clouds, batch 32 with 1,024 points per cloud,
**Warp brute-force Chamfer takes 0.295 ms versus 1.306 ms for tuned chunked
PyTorch: 4.4× faster**, including forward and backward. Both pass the independent
loss/gradient check. These are fresh measurements on the 46-SM RTX PRO 6000
Blackwell MIG slice, **not A40 measurements**.

The implementation is explicit: `wp.geometry.PointQueryBackend.BRUTE_FORCE`.
The default remains the Warp BVH backend. No native code or Torch dependency was
added to the library. The existing `wp.knn()` compatibility entry point also
accepts this backend.

```python
import warp as wp
import warp.geometry

brute = wp.geometry.PointQueryBackend.BRUTE_FORCE
forward = wp.geometry.knn(targets, decoded, 1, backend=brute)
reverse = wp.geometry.knn(decoded, targets, 1, backend=brute)
# Reduce forward[2] and reverse[2] to obtain symmetric squared Chamfer loss.
# Set requires_grad=True on the coordinates to differentiate the gathered distances.
```

The [runnable Chamfer example](../../../warp/examples/geometry/example_point_query_chamfer.py)
accepts `--backend brute_force --reverse-backend brute_force`.

## Matched end-to-end comparison

Batch 32, N=M=1,024, float32, squared symmetric Chamfer, gradients to decoded
queries only. Medians of 15 synchronized wall-clock samples after two warmups;
no CUDA graph replay. Uploads, compilation, and correctness-oracle work are
outside timing. Neighbor/distance/count output buffers are reused by Warp.

| Method | Forward + backward (ms) |
|---|---:|
| **Warp tiled brute force; both one-shot searches** | **0.295** |
| Warp brute-force search + Torch gathers/backward | 0.334 |
| PyTorch chunked MM cdist + gathers/backward, 128 rows | 1.306 |
| Warp BVH, fixed target index reused, reverse built each call | 1.990 |
| Warp BVH, both indexes built each call | 3.635 |
| PyTorch full cdist with autograd | 8.414 |

The chunked baseline visits each pair once and updates both directional minima
under `torch.no_grad()`, then gathers selected coordinates for backward.
128 rows was the best 1K setting in the earlier 32–2,048-row sweep; it was rerun
here. This is the fast default matrix-multiplication cdist route, with TF32
disabled. We did not substitute PyTorch's much slower forced-direct diagnostic.
The [previous cdist report](../cdist-batch/README.md) retains the full chunk sweep
and numerical diagnostics for larger sizes.

At 1K, the new backend is 12.3× faster than constructing both Warp BVHs, and
28.5× faster than full autograd cdist. These are Chamfer timings, not complete
training-epoch speedups.

## Crossover with BVH search

Same batch size, data sampling, gradient convention, 15-sample measurement,
and fresh one-shot searches in both directions. Neither method reuses trees.
All ten measurements pass the loss/gradient checks.

| Points per cloud | Warp brute force (ms) | Warp BVH build + query (ms) | Faster method |
|---:|---:|---:|---|
| 1,024 | **0.295** | 3.635 | Brute force, 12.3× |
| 4,096 | **2.025** | 4.802 | Brute force, 2.4× |
| 16,384 | 24.206 | **12.490** | BVH, 1.9× |
| 32,768 | 93.420 | **18.231** | BVH, 5.1× |
| 65,536 | 363.332 | **27.922** | BVH, 13.0× |

Exhaustive work grows with B×N×M even though memory remains bounded. These
measurements support an explicit small-cloud option, not a universal replacement
or a hardware-independent automatic crossover.

## What changed and what was tuned

* No spatial indexes are built, and no Python loop processes individual clouds.
  A lightweight owner retains the target array. In-place coordinate changes are
  immediately visible; `refit()` and `rebuild()` are no-ops for this backend.
* One CUDA launch handles all clouds. Each thread maintains its query's best k
  candidates while a block cooperatively loads bounded shared target tiles.
  The k=1 path specializes to a single minimum. CPU uses a scalar target stream.
* The implementation never allocates a B×N×M distance tensor. Target tiles have
  256 points; the shipped CUDA block has 256 threads. Target chunks are internal,
  so callers need only select the backend. General k, cutoffs, exclusions, and
  original-index tie breaking retain Warp's semantics.
* Distance selection uses noncontracted coordinate-difference arithmetic to match
  native Warp ranking and strict cutoff behavior. Differentiable gathers use the
  existing Warp kernels, including zero Euclidean subgradients at coincidence.

The [tuning script](tune.py) swept scalar target reads and shared tiles of
32/64/128/256/512 points, with 64/128/256 threads: **162 validated configurations**.
Inputs include uniform volume, sphere surface, pulmonary scans, and each with
added Gaussian noise (sigma 0.02 in the input coordinate units), plus smaller
batch, larger-cloud, and general-k cases. Every candidate's entire ID array was
checked against the Warp BVH result. Candidate order was shuffled deterministically.

At B=32, N=1K, k=1, the chosen 256/256 setting takes about 0.051 ms per search
across all six geometries, within 1% of the best tested setting. Scalar target
reads take about 0.096–0.097 ms. Shared target reuse provides about 1.9× of the
kernel-level improvement; removing the two BVH builds provides the larger
end-to-end improvement.

The single fixed launch policy is intentionally conservative: the sweep found
better settings for B=1 (0.017 versus 0.021 ms), B=32/N=4K (0.709 versus
0.733 ms), and the small k=11 case (0.080 versus 0.095 ms). The shipped policy
is tuned for dense small-cloud Chamfer; it is not claimed optimal for every k
and query density. No automatic backend selection or per-device autotuning was added.

## Capture and profiling

The backend has no host data readback or host-device synchronization in its
query path. Tests cover CUDA and APIC capture of one-shot queries, changed
coordinates on replay, and captured Chamfer backward into both inputs. A separate
full-workload benchmark measures gradients to **both** coordinate arrays:

| Batch / points | Eager Chamfer (ms) | Captured Chamfer replay (ms) |
|---|---:|---:|
| 1 / 1,024 | 0.234 | 0.073 |
| 1 / 4,096 | 0.243 | 0.195 |
| 32 / 1,024 | 0.294 | 0.239 |
| 32 / 4,096 | 2.001 | 1.954 |

These replay numbers are supplemental, not compared against eager Torch to
claim an extra speedup. The main table uses the matched decoded-only gradient
workload, while this existing benchmark differentiates both inputs.

CUDA event instrumentation at B=32/N=1K attributes roughly 0.110 ms to the two
searches, 0.010 ms to forward distance gathers, 0.020 ms to backward gathers,
and 0.103 ms to the example's two atomic loss reductions. The reductions are
now a substantial remaining cost. Instrumented times include event overhead;
use the uninstrumented tables for total latency. CPU profiling shows a fixed
number of query/gather/reduction launches, with no per-cloud tree work.

## Validation and reproduction

64 point-query tests pass on CPU/CUDA on both branch checkouts, including 10 new
brute-force tests. Coverage includes tile tails crossing batch boundaries,
k=1/3/11, k larger than the target count, empty layouts, int64 IDs, cutoff
underflow, exclusions, nonfinite queries, tied distances, owner reuse, both-input
gradients, CUDA/APIC capture, and gradient replay after coordinate updates.
The legacy `wp.knn()` construction/reuse smoke and runnable Chamfer example pass.
Pre-commit checks and the documentation build pass.

The input archive SHA256 is
`8a16347cc942777083a0c763e049a693af452a2c7bbb07ad7e8511a1457889a4`.
Raw coordinates are not committed. Sampling uses NumPy
`SeedSequence([2348, B, N])`, cycles through eligible clouds, and independently
subsamples each target/query cloud without replacement. No normalization or
noise is applied to the headline Chamfer comparisons. Torch is 2.7.1+cu128;
Warp uses CUDA toolkit 12.9 on the 24 GiB, 46-SM Blackwell slice.

Library measurements use commit `d64a681`. The extended cdist harness is committed
with this report; every end-to-end record includes its exact source hash, library
source hashes, native-library hash, sample hashes, and all timing samples. All
14 end-to-end records pass the existing independent NumPy loss/full-gradient
check and sampled exhaustive neighbor checks. The tuning records retain every
candidate's samples and correctness result.

Run from the repository root, with a built Warp library and Torch installed for
the cdist comparison:

```bash
uv run --no-sync -m warp.examples.benchmarks.benchmark_point_query_cdist \
  --archive ../pulmonary_point_clouds.zip --output chamfer.json \
  --points 1024 4096 16384 32768 65536 \
  --methods warp_brute_force warp_build_both --repeats 15
uv run --no-sync -m warp.examples.benchmarks.benchmark_point_query_cdist \
  --archive ../pulmonary_point_clouds.zip --output torch-1k.json --points 1024 \
  --methods chunked_mm cdist warp_brute_force_torch warp --chunk-rows 128 --repeats 15
uv run --no-sync reports/knn/brute-force/tune.py \
  --archive ../pulmonary_point_clouds.zip --output tuning.json
uv run --no-sync -m warp.examples.benchmarks.benchmark_point_query_chamfer \
  --archive ../pulmonary_point_clouds.zip --output graphs.json \
  --batches 1 32 --points 1024 4096 --backends brute_force --repeats 15 --profile
```

Raw results: [Chamfer scaling](chamfer.json), [fresh 1K controls](torch-1k.json),
[tuning sweep](tuning.json), [graph timings and CUDA profile](graphs.json),
[worker logs](workers.log.gz), and [test, lint, and documentation logs](validation.log.gz).
