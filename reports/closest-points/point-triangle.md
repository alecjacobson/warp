# Point–triangle closest point: performance and feature accuracy

**The API and correctness checks are implemented; the strict per-workload speed target remains open.**

Implementation: [`0f440012b`](https://github.com/alecjacobson/warp/commit/0f440012b864b3b820112d1546e2763a9bd7fcf2) on `alecjacobson/point-tri`, stacked on `994126406` of the pending intersection-next branch.

`wp.geometry.closest_point_point_tri(p, a, b, c)` returns `(bary, distance_squared, feature)` for float32 or float64. `TriangleFeature` uses Newton-compatible IDs 0–6. The barycentrics and distance support automatic differentiation.

## Method

- NVIDIA RTX PRO 6000 Blackwell Server Edition **MIG 1g.24gb**, 46 SMs; driver 595.91.07, embedded CUDA 12.9. These are not full-GPU timings.
- Float32, 262,144 independent queries; 30 launches per CUDA graph, median of 21 samples, alternating implementation order. Table entries are medians across three complete runs. Block size 128.
- Both timing events are captured inside the graph, excluding host scheduling gaps. Compilation, transfers, accuracy checks, BVH traversal, and mesh-query face filtering are excluded. Input arrays and output layouts are identical.
- Every final measurement ran under the machine-wide `gpu_run exclusive` lock, with an idle process table checked inside the exclusive window before initialization.
- CPU quota: `400000 / 100000`, equivalent to four cores. All three main runs recorded zero additional throttled periods and zero additional throttled microseconds. The raw JSON includes before/after cgroup counters.
- Native baseline calls the actual `wp::closest_point_to_triangle` header. Newton is imported from its original checkout at `f39d8199f18d5117f9dd8073c1eb71c9a196f020`; its function AST is checked against the maintained benchmark copy. The helper is unchanged at `a1903cdc48f349d0de7d5e19f1838e736c4d1cf3`.
- The source was uncommitted during measurement, so raw `warp_commit` identifies the base HEAD. The recorded implementation SHA-256 matches the committed implementation linked above. Earlier tuning runs and runs contending with Blender are excluded.

## Timings

| Workload | Native (µs) | Newton (µs) | New Warp (µs) | Native / new | Newton / new |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 9.683 | 9.553 | 9.548 | 1.014× | 1.001× |
| near | 9.493 | 9.418 | 9.390 | 1.011× | 1.003× |
| face | 9.477 | 9.413 | 9.343 | 1.014× | 1.008× |
| edge | 9.457 | 9.346 | 9.407 | 1.005× | 0.994× |
| vertex | 9.321 | 9.212 | 9.342 | 0.998× | 0.986× |
| thin | 9.549 | 9.465 | 9.866 | 0.968× | 0.959× |
| degenerate | 9.488 | 9.367 | 9.638 | 0.984× | 0.972× |
| small | 9.611 | 9.476 | 9.518 | 1.010× | 0.996× |
| large | 9.730 | 9.626 | 9.548 | 1.019× | 1.008× |
| boundary | 9.479 | 9.412 | 9.371 | 1.012× | 1.004× |

Ratios below one mean the new routine is slower. Ordinary workloads are near parity. The remaining regressions versus Newton include approximately **4.2% on thin triangles**, **2.9% on degenerate triangles**, and **1.4% on vertex queries**. No universal speedup is claimed.

The compiled kernels use 34 registers/thread (native), 38 (Newton), and 46 (new Warp), with zero local-memory spill allocation reported for all three.

A separate run at **1,048,576 queries** puts every measured workload within **0.15%** of both baselines, consistent with memory bandwidth dominating. A separate run omitting feature output also retains small timing differences, so feature inference on the native baseline is not the sole source of the performance gap. See the raw results for both controls.

## Accuracy

Each workload checks 256 uniformly sampled queries against exact rational active-set enumeration on the actual float32 inputs. This is independent of the region-test implementations. Unit tests additionally check both precisions on CPU and CUDA.

**Native has no feature-ID return.** Its labels below are inferred by strict zero/nonzero tests on the returned barycentric weights. Rounding in the reconstructed third weight affects that inference. Interpret these separately from the explicit feature identifiers returned by Newton and the new function.

| Workload | Native inferred feature mismatches | Newton feature mismatches | New feature mismatches | Native / Newton nonfinite results | New nonfinite results |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 30 | 0 | 0 | 0 / 0 | 0 |
| near | 0 | 0 | 0 | 0 / 0 | 0 |
| face | 0 | 0 | 0 | 0 / 0 | 0 |
| edge | 59 | 0 | 0 | 0 / 0 | 0 |
| vertex | 0 | 0 | 0 | 0 / 0 | 0 |
| thin | 140 | 177 | 0 | 0 / 0 | 0 |
| small | 138 | 126 | 0 | 0 / 0 | 0 |
| large | 135 | 120 | 0 | 81 / 81 | 0 |
| boundary | 105 | 19 | 19 | 0 / 0 | 0 |

For rotated thin triangles, the maximum normalized closest-point error drops from about **1.66e-5** for the references to **6.91e-7** for the new function. At small/large scales, reference errors reach order one and the large workload produces 81 nonfinite outputs; the new routine stays at ordinary float32 rounding levels in this sample.

On ordinary workloads all three operate near float32 rounding error; the new function does not beat both references on every individual sample or maximum-error statistic. The raw results retain those differences.

The noisy boundary workload is nominally on an edge before quantization. Quantizing its vertices and query separately creates tiny true offsets. Newton and the new routine both misclassify 19/256 against the rational oracle. **This is not an exact-predicate API.** The separate exactly constructed boundary fixtures pass on CPU and CUDA. Degenerate inputs have nonunique barycentric representations, so benchmark feature mismatch counts exclude them; unit tests instead check the returned feature/support, distance, and witnesses.

Witness error is Euclidean error divided by `sqrt(max_v |p-v|²)`. Squared-distance error is divided by `max_v |p-v|²`. Maximum errors in JSON cover finite outputs; nonfinite results are counted separately.

## Validation and reproduction

- 83 selected regression tests passed on CPU and CUDA, including the 22 new point–triangle tests. Full test suite not run.
- New coverage includes exact rational oracle comparisons, all seven feature regions and vertex permutations, exact oblique boundaries, one-ULP offsets, repeated/collinear vertices, rotated slivers, scales 1e-15 through 1e15, translation, finite differences in all seven regions, and squared-distance envelope gradients across scales.
- Documentation built without warnings; pre-commit passed.

Run from the feature branch, using `gpu_run` as defined in the machine AGENTS.md:

```bash
gpu_run shared uv run --extra dev -m warp.tests -s autodetect \
  -k TestClosest -k TestTriangleClosestPoint -k TestMeshQueryPoint

gpu_run exclusive bash -c '
  activity=$(nvidia-smi) || exit 75
  printf "%s\n" "$activity"
  printf "%s\n" "$activity" | rg -q "No running processes found" || exit 75
  exec uv run asv/benchmarks/closest_point_tri.py \
    --newton-source /path/to/newton --output point-triangle.json
'
```

Raw measurements: [run 1](point-triangle-run1.json), [run 2](point-triangle-run2.json), [run 3](point-triangle-run3.json), [without feature output](point-triangle-no-features.json), [1,048,576 queries](point-triangle-million.json).

The remaining work toward the original strict speed target is reducing instruction/register cost for thin, degenerate, and vertex cases while retaining the gradient and feature checks. Numerical shortcuts that preserve only forward values are not sufficient.
