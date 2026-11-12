Ordinary calls; denominator: **Warp batch (automatic tuning)**. Fresh Warp timings; unchanged competitor timings retained. CUDA graph replay is reported separately in [warp-timings.md](warp-timings.md).

| Method | Warp query wins / comparable | Geomean other / Warp |
| --- | ---: | ---: |
| Warp scalar (simple dense launch) | 40/40 | 3.90x |
| Warp scalar (tuned launch) | 4/40 | 0.97x |
| Pinned PhysicsNeMo | 40/40 | 10.43x |
| cuBQL | 40/40 | 4.44x |
| CuPy KDTree | 40/40 | 50.95x |
| KeOps | 40/40 | 43.92x |
| Faiss Flat | 2/5 | 2.19x |
| cuVS expanded L2 | 0/3 | 0.57x |
| cuVS direct L2 | 36/40 | 7.89x |
| cuVS Minkowski p=2 | 38/40 | 185.16x |
| PyTorch3D | 40/40 | 48.14x |
