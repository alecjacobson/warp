# warp.geometry sampling benchmarks

Standalone scripts (not part of the test suite) for measuring
`warp.geometry.PoissonDiskSampler`.

## Poisson-disk vs. farthest-point sampling

`poisson_vs_fps.py` compares our parallel Poisson-disk sampler against
farthest-point sampling (FPS). Both select points from the **same** dense
candidate pool (generated once with `warp.geometry.UniformSampler`), and FPS is
asked for exactly as many points as the Poisson sampler produced, so it is an
apples-to-apples comparison of two ways to thin the same pool.

`warp_fps.py` is a pure-Warp FPS (the block-aware, radix-sort + Tile-API
algorithm from [NVIDIA Kaolin](https://github.com/NVIDIAGameWorks/kaolin),
adapted to a NumPy interface so no PyTorch is needed).

Run it:

```sh
uv run --with usd-core --with matplotlib tools/benchmarks/poisson_vs_fps.py
```

### Results (Stanford bunny, NVIDIA L40, best of 3)

| radius | output pts | candidates | Poisson-disk | FPS       | speedup | min-dist / r (PDS) | min-dist / r (FPS) |
| ------ | ---------- | ---------- | ------------ | --------- | ------- | ------------------ | ------------------ |
| 0.020  | 9,758      | 219,936    | 4.3 ms       | 24.1 ms   | 6x      | 1.000              | 0.959              |
| 0.010  | 39,087     | 879,745    | 9.6 ms       | 127.9 ms  | 13x     | 1.000              | 0.960              |
| 0.005  | 156,796    | 3,518,980  | 34.4 ms      | 1288.9 ms | 38x     | 1.000              | 0.960              |

Throughput: Poisson-disk sustains ~4.5M samples/s; FPS runs at ~0.1-0.4M
samples/s. The gap widens with the output count because FPS is iterative (each
"round" radix-sorts the whole pool and accepts a head-chunk of points), whereas
the Poisson-disk sampler is a fixed 27-pass parallel sweep.

### Spectral quality

![pair correlation](poisson_vs_fps_pcf.png)

The pair-correlation function `g(r)` (the differential-domain blue-noise
measure) for both methods on the same output count:

- **Poisson-disk** has a hard gap that ends exactly at the radius (`min-dist =
  1.000 r`, guaranteed) and a **sharp** first-neighbor peak (~2.7).
- **FPS** has a **softer, broader** peak (~1.8) and lets a few pairs fall
  slightly inside the radius (`min-dist ~ 0.96 r`), since it only greedily
  maximizes distance over a fixed candidate set with no hard radius.

Both are blue noise (no low-frequency power, a peak near the mean spacing,
decaying to 1), but the Poisson-disk sampler gives a stronger, cleaner
blue-noise signature and a guaranteed minimum distance -- at a fraction of the
cost.
