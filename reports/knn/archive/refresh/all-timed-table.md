**Historical snapshot.** Measurements, API names, and status below describe the recorded implementation and hardware. Use the [current comparisons](../../comparisons.md) for this branch. Archived reproduction commands assume their original checkout and paths.

Ordinary calls; denominator: **retained Warp batch (LBVH, automatic tuning)**. New matched Warp builder controls are included; earlier Warp and competitor samples are retained. CUDA graph replay is reported separately in [warp-timings.md](warp-timings.md).

| Method | Warp faster / timed cases | Geomean other / Warp | Cases passing sampled checks |
| --- | ---: | ---: | ---: |
| Warp scalar (simple dense launch) | 40/40 | 2.81x | 40/40 |
| Warp scalar (tuned launch) | 7/40 | 0.99x | 40/40 |
| Warp kNN + LBVH (matched control) | 4/40 | 0.92x | 40/40 |
| Warp kNN + cuBQL SAH | 40/40 | 12.91x | 40/40 |
| Pinned PhysicsNeMo | 40/40 | 4.37x | 40/40 |
| cuBQL custom top-k (leaf 4, dense) | 40/40 | 5.93x | 40/40 |
| cuBQL findKNN (defaults) | 40/40 | 6.93x | 40/40 |
| cuBQL findKNN (best tested settings) | 36/40 | 1.52x | 40/40 |
| CuPy KDTree | 40/40 | 46.27x | 40/40 |
| KeOps | 40/40 | 18.43x | 40/40 |
| Faiss Flat | 36/40 | 10.87x | 5/40 |
| cuVS expanded L2 | 36/40 | 3.83x | 3/40 |
| cuVS direct L2 | 36/40 | 3.36x | 40/40 |
| cuVS Minkowski p=2 | 36/40 | 77.73x | 40/40 |
| PyTorch3D | 40/40 | 20.20x | 40/40 |
