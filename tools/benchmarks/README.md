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

### Fairness

FPS is timed at full speed, with no host stalls: the vendored implementation
uploads the pool *before* starting its timer, and its main loop never
synchronizes with the host (it tracks a min/max estimate of progress and reads
back the exact count only when it might be done). As a check, the benchmark
reproduces the FPS author's reference point, N=10^6, k=1024:

```
[validation] FPS N=1e6 k=1024: 20.4 ms   (author: 26.5 ms on RTX 3090 Ti)
```

(The L40 lands a bit under the 3090 Ti figure, as expected.)

The comparison is also conservative toward FPS: FPS is handed the candidate pool
for free, while the Poisson sampler is shown both `solve` (thinning the same
pool -- the apples-to-apples number) and `total` (including generating the pool
that FPS got gratis).

### Results (Stanford bunny, NVIDIA L40, best of 3)

| radius | output pts | candidates | PDS solve | PDS total | FPS       | FPS / PDS-solve | min-dist/r (PDS / FPS) |
| ------ | ---------- | ---------- | --------- | --------- | --------- | --------------- | ---------------------- |
| 0.020  | 9,758      | 219,936    | 3.8 ms    | 4.4 ms    | 24.9 ms   | 7x              | 1.000 / 0.959          |
| 0.010  | 39,087     | 879,745    | 9.9 ms    | 10.5 ms   | 122.1 ms  | 12x             | 1.000 / 0.960          |
| 0.005  | 156,796    | 3,518,980  | 34.3 ms   | 35.4 ms   | 1271.5 ms | 37x             | 1.000 / 0.960          |

Candidate generation is cheap (`total - solve` < 1 ms), so the two PDS columns
nearly coincide. The Poisson sampler sustains ~4.5M samples/s.

FPS cost scales with the **output** count `k`: each round radix-sorts the whole
pool and accepts a head-chunk of up to 512 points, so it needs ~`k/512` sorts.
That is cheap for small `k` (the author's k=1024 is ~2 rounds), but a
blue-noise *surface* sampling typically wants tens of thousands of points, where
FPS does hundreds of full-pool sorts. The Poisson sampler is instead a fixed
27-pass parallel sweep whose cost scales with the candidate pool, not with `k`.

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
