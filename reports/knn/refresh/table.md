Ordinary calls; denominator: **Warp batch (automatic tuning)**. Fresh Warp timings; unchanged competitor timings retained. CUDA graph replay is reported separately in [warp-timings.md](warp-timings.md).

| Method | Completed | Pass sampled checks | Min recall | Warp wins / comparable | Geomean other / Warp | Query graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Warp scalar (simple dense launch) | 40 | 40 | 1.00000 | 40/40 | 2.81x | pass |
| Warp scalar (tuned launch) | 40 | 40 | 1.00000 | 7/40 | 0.99x | pass |
| Pinned PhysicsNeMo | 40 | 40 | 1.00000 | 40/40 | 4.37x | pass |
| cuBQL | 40 | 40 | 1.00000 | 40/40 | 5.93x | pass |
| CuPy KDTree | 40 | 40 | 1.00000 | 40/40 | 46.27x | failed/unsupported |
| KeOps | 40 | 40 | 1.00000 | 40/40 | 18.43x | failed/unsupported |
| Faiss Flat | 40 | 5 | 0.84375 | 2/5 | 1.77x | pass |
| cuVS expanded L2 | 40 | 3 | 0.00284 | 0/3 | 0.50x | pass |
| cuVS direct L2 | 40 | 40 | 1.00000 | 36/40 | 3.36x | pass |
| cuVS Minkowski p=2 | 40 | 40 | 1.00000 | 36/40 | 77.73x | pass |
| PyTorch3D | 40 | 40 | 1.00000 | 40/40 | 20.20x | pass |
