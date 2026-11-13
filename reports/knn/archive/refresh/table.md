**Historical snapshot.** Measurements, API names, and status below describe the recorded implementation and hardware. Use the [current comparisons](../../comparisons.md) for this branch. Archived reproduction commands assume their original checkout and paths.

Ordinary calls; denominator: **retained Warp batch (LBVH, automatic tuning)**. New matched Warp builder controls are included; earlier Warp and competitor samples are retained. CUDA graph replay is reported separately in [warp-timings.md](warp-timings.md).

| Method | Completed | Pass sampled checks | Min recall | Warp wins / comparable | Geomean other / Warp | Query graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Warp scalar (simple dense launch) | 40 | 40 | 1.00000 | 40/40 | 2.81x | pass |
| Warp scalar (tuned launch) | 40 | 40 | 1.00000 | 7/40 | 0.99x | pass |
| Warp kNN + LBVH (matched control) | 40 | 40 | 1.00000 | 4/40 | 0.92x | pass |
| Warp kNN + cuBQL SAH | 40 | 40 | 1.00000 | 40/40 | 12.91x | pass |
| Pinned PhysicsNeMo | 40 | 40 | 1.00000 | 40/40 | 4.37x | pass |
| cuBQL custom top-k (leaf 4, dense) | 40 | 40 | 1.00000 | 40/40 | 5.93x | pass |
| cuBQL findKNN (defaults) | 40 | 40 | 1.00000 | 40/40 | 6.93x | pass |
| cuBQL findKNN (best tested settings) | 40 | 40 | 1.00000 | 36/40 | 1.52x | pass |
| CuPy KDTree | 40 | 40 | 1.00000 | 40/40 | 46.27x | failed/unsupported |
| KeOps | 40 | 40 | 1.00000 | 40/40 | 18.43x | failed/unsupported |
| Faiss Flat | 40 | 5 | 0.84375 | 2/5 | 1.77x | pass |
| cuVS expanded L2 | 40 | 3 | 0.00284 | 0/3 | 0.50x | pass |
| cuVS direct L2 | 40 | 40 | 1.00000 | 36/40 | 3.36x | pass |
| cuVS Minkowski p=2 | 40 | 40 | 1.00000 | 36/40 | 77.73x | pass |
| PyTorch3D | 40 | 40 | 1.00000 | 40/40 | 20.20x | pass |
