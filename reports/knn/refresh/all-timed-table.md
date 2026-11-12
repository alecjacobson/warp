Ordinary calls; denominator: **Warp batch (automatic tuning)**. Fresh Warp timings; unchanged competitor timings retained. CUDA graph replay is reported separately in [warp-timings.md](warp-timings.md).

| Method | Warp faster / timed cases | Geomean other / Warp | Cases passing sampled checks |
| --- | ---: | ---: | ---: |
| Warp scalar (simple dense launch) | 40/40 | 2.81x | 40/40 |
| Warp scalar (tuned launch) | 7/40 | 0.99x | 40/40 |
| Pinned PhysicsNeMo | 40/40 | 4.37x | 40/40 |
| cuBQL | 40/40 | 5.93x | 40/40 |
| CuPy KDTree | 40/40 | 46.27x | 40/40 |
| KeOps | 40/40 | 18.43x | 40/40 |
| Faiss Flat | 36/40 | 10.87x | 5/40 |
| cuVS expanded L2 | 36/40 | 3.83x | 3/40 |
| cuVS direct L2 | 36/40 | 3.36x | 40/40 |
| cuVS Minkowski p=2 | 36/40 | 77.73x | 40/40 |
| PyTorch3D | 40/40 | 20.20x | 40/40 |
