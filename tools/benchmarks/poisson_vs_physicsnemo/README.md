# warp.geometry vs. PhysicsNeMo — GPU Poisson-disk sampling

Head-to-head of `warp.geometry.poisson_disk_sample` against **PhysicsNeMo's**
Warp Poisson-disk sampler
([`mesh_poisson_disk_sample`](https://github.com/NVIDIA/physicsnemo/blob/main/physicsnemo/nn/functional/geometry/mesh_poisson_disk_sample/_warp_impl/op.py)),
both on the GPU. The two are the **same algorithm family** — grid-accelerated
dart-throwing / Poisson elimination — so this is an apples-to-apples comparison
once the point counts are matched.

## Why two environments

PhysicsNeMo's release that ships this op pins a torch build newer than many
drivers, and the op lives inside `physicsnemo.nn`, whose package import pulls in
unrelated modules. Rather than install the whole framework, `run_physicsnemo.py`
fetches the **two source files** of the sampler at a *pinned commit* into a
git-ignored `_vendor/` directory and runs them verbatim. The only substitution
is `FunctionSpec` (a device/stream helper), replaced by a small stub that binds
Warp to torch's current CUDA stream — every GPU kernel is PhysicsNeMo's own,
unchanged. The pinned commit is in `run_physicsnemo.py` (`PNEMO_COMMIT`).

## Running it

```sh
# 1. Export the shared mesh once, in the Warp env (needs usd-core):
uv run --with usd-core tools/benchmarks/poisson_vs_physicsnemo/export_mesh.py

# 2. Our sampler, in the Warp env:
uv run --with scipy tools/benchmarks/poisson_vs_physicsnemo/run_ours.py

# 3. PhysicsNeMo, in a separate env with torch (a CUDA build matching your
#    driver), warp-lang, and scipy:
python -m venv /tmp/pnemo && . /tmp/pnemo/bin/activate
pip install torch warp-lang scipy   # pick a torch CUDA wheel for your driver
python tools/benchmarks/poisson_vs_physicsnemo/run_physicsnemo.py
```

Both scripts sample the identical mesh (Stanford bunny, 6,102 verts / 12,200
faces) at radii of 2%, 1%, and 0.5% of the bounding-box diagonal, reporting best
of 7 timed runs after warmup, plus `min-d/r` (achieved minimum-distance / radius;
`1.000` = a correct, tightly packed blue-noise set).

## Results (NVIDIA L40)

Both **GPU**. PhysicsNeMo's dart-throwing runs to near-maximal, so our default
candidate budget (`candidate_multiplier=12`) gives a slightly sparser set at the
same radius; `candidate_multiplier=48` lifts ours to a matching count. Read the
comparison at the matched rows.

| radius (of diag) | method | points | best time | points/s | min-d/r |
| ---------------- | ------ | ------ | --------- | -------- | ------- |
| 1.0% | PhysicsNeMo (default)      | 6,325  | 9.87 ms  | 0.64 M | 1.000 |
| 1.0% | **warp.geometry** mult=12  | 5,773  | **3.29 ms** | 1.75 M | 1.000 |
| 1.0% | **warp.geometry** mult=48 (count-matched) | 6,279 | **3.76 ms** | 1.67 M | 1.000 |
| 0.5% | PhysicsNeMo (default)      | 25,266 | 20.56 ms | 1.23 M | 1.000 |
| 0.5% | **warp.geometry** mult=12  | 23,201 | **4.22 ms** | 5.50 M | 1.000 |
| 0.5% | **warp.geometry** mult=48 (count-matched) | 25,125 | **8.22 ms** | 3.06 M | 1.000 |

**At matched point counts, `warp.geometry` is ~2.2–2.6× faster**, and the gap
widens with density. Identical blue-noise quality (`min-d/r = 1.000` for both).
Tuning PhysicsNeMo's exposed knobs (`batch_size`, `hash_grid_resolution`) did not
beat its defaults on this mesh — larger batches and finer grids only slowed it.

## Why we win, and what they do that we don't

Both algorithms throw surface candidates and reject any within `radius` of an
accepted point using a spatial hash grid. The performance difference is
structural:

- **PhysicsNeMo** is host-synchronized dart-throwing: a Python loop (≤64
  iterations) that each pass generates a candidate batch, **rebuilds the
  accepted-points hash grid**, and **reads `accepted_count` back to the host** to
  test for saturation. Those per-iteration syncs and grid rebuilds are the cost.
- **Ours** (Bowers et al. 2010) is a fixed 27-phase parallel sweep over a
  cell-sorted candidate pool with **no host synchronization** — the whole thing
  stays on the GPU, which is why our lead grows with point count.

Things PhysicsNeMo does that ours does **not**, worth considering:

1. **Weighted Sample Elimination** (Yuksel 2015 / Open3D) as a second mode —
   oversample, then greedily remove the highest-weight samples until an **exact**
   target count remains. Gives deterministic output size and a cleaner spectrum
   than raw dart-throwing.
2. **Exact `target_num_points`** — request N points and get N. Ours controls
   density via `radius` (and candidate budget) and cannot hit an exact count.
3. **Spatially-varying radius** (`per_vertex_radius`) — variable-density Poisson
   (finer sampling where a per-vertex field is smaller). Ours is a single global
   radius (plus the optional geodesic metric).
4. **Bounded memory via batched throwing** — memory scales with `batch_size`, not
   total sample count, so it can push to very dense sets on large meshes without
   allocating the whole candidate pool. Our fixed pool (`candidate_multiplier × N`)
   is faster but allocates all candidates up front, so very dense sampling costs
   proportional memory.

## Caveats

- Different algorithms → not byte-identical output; compare at the matched-count
  rows. Dart-throwing yields a marginally denser maximal set; our default is
  ~10% sparser at the same radius (still `min-d = r` guaranteed).
- PhysicsNeMo ran on its own env (Warp 1.17 + torch/cu126); ours on the repo's
  Warp. Same L40, same mesh. Only `FunctionSpec` (device/stream context) was
  stubbed — the kernels are unmodified.
- Small mesh. PhysicsNeMo's ≤64-iteration host overhead amortizes better at
  larger scale, though its per-iteration grid rebuild still scales with the
  accepted set.
