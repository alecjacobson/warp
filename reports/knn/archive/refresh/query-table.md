**Historical snapshot.** Measurements, API names, and status below describe the recorded implementation and hardware. Use the [current comparisons](../../comparisons.md) for this branch. Archived reproduction commands assume their original checkout and paths.

Ordinary calls; denominator: **retained Warp batch (LBVH, automatic tuning)**. New matched Warp builder controls are included; earlier Warp and competitor samples are retained. CUDA graph replay is reported separately in [warp-timings.md](warp-timings.md).

| Method | Warp query wins / comparable | Geomean other / Warp |
| --- | ---: | ---: |
| Warp scalar (simple dense launch) | 40/40 | 3.90x |
| Warp scalar (tuned launch) | 4/40 | 0.97x |
| Warp kNN + LBVH (matched control) | 18/40 | 1.00x |
| Warp kNN + cuBQL SAH | 8/40 | 0.93x |
| Pinned PhysicsNeMo | 40/40 | 10.43x |
| cuBQL custom top-k (leaf 4, dense) | 40/40 | 4.44x |
| cuBQL findKNN (defaults) | 40/40 | 4.33x |
| cuBQL findKNN (best tested settings) | 1/40 | 0.70x |
| CuPy KDTree | 40/40 | 50.95x |
| KeOps | 40/40 | 43.92x |
| Faiss Flat | 2/5 | 2.19x |
| cuVS expanded L2 | 0/3 | 0.57x |
| cuVS direct L2 | 36/40 | 7.89x |
| cuVS Minkowski p=2 | 38/40 | 185.16x |
| PyTorch3D | 40/40 | 48.14x |
