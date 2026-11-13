**Historical snapshot.** Measurements, API names, and status below describe the recorded implementation and hardware. Use the [current comparisons](../../comparisons.md) for this branch. Archived reproduction commands assume their original checkout and paths.

| Method | Completed | Pass sampled checks | Min recall | Warp wins / comparable | Geomean other / Warp | Query graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| warp_default | 40 | 40 | 1.00000 | 21/40 | 1.18x | pass |
| warp_configured | 40 | 40 | 1.00000 | 0/40 | 1.00x | pass |
| physicsnemo | 40 | 40 | 1.00000 | 36/40 | 1.89x | pass |
| cubql | 40 | 40 | 1.00000 | 37/40 | 2.56x | pass |
| cupy_kdtree | 40 | 40 | 1.00000 | 40/40 | 19.98x | failed/unsupported |
| keops | 40 | 40 | 1.00000 | 40/40 | 7.97x | failed/unsupported |
| faiss_flat | 40 | 5 | 0.84375 | 2/5 | 1.23x | pass |
| cuvs_brute_force | 40 | 3 | 0.00284 | 0/3 | 0.54x | pass |
| cuvs_unexpanded | 40 | 40 | 1.00000 | 20/40 | 1.47x | pass |
| cuvs_minkowski | 40 | 40 | 1.00000 | 36/40 | 33.70x | pass |
| pytorch3d | 40 | 40 | 1.00000 | 40/40 | 8.74x | pass |
| cuvs_cagra | 40 | 38 | 0.93750 | ANN; separate | — | pass |
