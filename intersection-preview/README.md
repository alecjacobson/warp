# Spatial-hash mesh intersection preview

**Temporary branch report — remove this entire `intersection-preview/` directory before opening the MR.**

The [example](../warp/examples/geometry/example_mesh_intersect_hash.py) detects
triangle intersections between two generated box meshes on the GPU. Mesh B is
rotated 0.31 radians about Z and translated by `(0.7, 0.3, 0.2)`.

![Spatial-hash candidates and confirmed triangle intersections on two box meshes](candidates.png)

Left: the original overlapping configuration. Middle: mesh A alone. Right: mesh B
alone. The latter two are translated display copies; all classification uses the
original coordinates.

- **Gray:** face participates in no candidate pair.
- **Amber:** face participates in at least one candidate pair, but no confirmed intersection.
- **Red:** face participates in at least one confirmed triangle intersection.

Colors summarize pair membership per face. Red marks whole triangles, not
intersection curves, and a red triangle can also participate in rejected candidate
pairs. The separated copies expose faces obscured by the other model in the overlap
view.

## Results

Measured from the run that produced this screenshot; machine-readable values are
in [stats.json](stats.json).

| Metric | Value |
| --- | ---: |
| Triangles in mesh A / B | 432 / 432 |
| Subdivisions per box edge | 6 |
| Exhaustive cross-mesh pairs | 186,624 |
| Broad-phase candidate pairs | 3,439 |
| Confirmed intersecting pairs | 120 |
| Candidates rejected by narrow phase | 3,319 |
| Reduction in narrow-phase calls versus all pairs | 98.16% |
| Faces participating in candidates, A / B | 306 / 291 |
| Faces participating in intersections, A / B | 60 / 60 |
| Hash cell width | 0.5 |
| Hash table dimensions | 16 × 16 × 16 |
| Screenshot resolution | 1800 × 800 |

Warp 1.19.0.dev1, CUDA Toolkit 13.0, driver 13.2, NVIDIA RTX PRO 6000 Blackwell
Server Edition MIG 1g.24gb. Rendering uses Polyscope 2.6.1 and its
[`openGL3_egl` headless backend](https://polyscope.run/py/features/headless_rendering/).
The screenshot was generated with `DISPLAY` unset; the EGL renderer reported
NVIDIA OpenGL 3.3.0, driver 595.58.03.

The reduction above is a pair-count comparison, not a measured timing speedup.

## Segment–triangle mode

Pass `--segments` to replace model A with all its unique geometric mesh edges,
including triangulation diagonals. Exactly coincident vertices are merged before
deduplicating undirected edges, so shared edges appear only once.
Model B remains a triangle mesh.

![Spatial-hash candidate segments and confirmed segment–triangle intersections](segments.png)

Left: the original configuration. Middle: the segments of A alone. Right: B alone.
Gray means no candidate, amber means candidate but no confirmed hit, and red means
at least one confirmed hit. Colors apply per segment on A and per face on B; they
do not mark the precise intersection points. A slightly oblique camera reveals
the wireframe's depth.

The renderer uses Polyscope's
[curve networks with per-edge colors](https://polyscope.run/py/structures/curve_network/color_quantities/).
The GPU result is checked against an exhaustive, independent NumPy
Möller–Trumbore reference for every segment–triangle pair in this scene.
Raw results: [segments-stats.json](segments-stats.json).

| Metric | Value |
| --- | ---: |
| Unique segments in A / triangles in B | 648 / 432 |
| Exhaustive cross-model pairs | 279,936 |
| Broad-phase candidate pairs | 3,648 |
| Confirmed intersecting pairs | 60 |
| Candidates rejected by narrow phase | 3,588 |
| Reduction in narrow-phase calls versus all pairs | 98.70% |
| Segments participating in candidates / intersections | 423 / 60 |
| Faces participating in candidates / intersections | 285 / 36 |

```bash
env -u DISPLAY uv run --with polyscope==2.6.1 \
  -m warp.examples.geometry.example_mesh_intersect_hash \
  --device cuda:0 --segments \
  --screenshot intersection-preview/segments.png \
  --stats intersection-preview/segments-stats.json
```

The new pure Warp `warp.geometry.intersect_seg_tri(p0, p1, v0, v1, v2)` returns
`bool` for `vec3` and `vec3d`. It clips the segment's parameter interval against
the triangle plane and projected edge half-planes. Endpoint, edge, vertex,
coplanar, and collinear overlap count as hits; zero-length segments are points,
and zero-area triangles return `False`.
It uses the existing tri–tri predicate's absolute `1e-6` tolerance on unnormalized
plane equations; projected edge comparisons have no additional tolerance.
This is not an exact predicate or a uniform world-space tolerance.

**29 tests passed on CPU/GPU** in `warp.tests.test_intersect` and
`warp.tests.geometry.test_mesh_intersect_hash` after adding this mode. They cover
both precisions, segment reversal, all triangle vertex permutations, all three
projection planes, multiple scales, boundary and degenerate cases, plane-tolerance
boundaries, random 3D pairs against an independent NumPy reference, and 1,000
coplanar segment fixtures against an exact rational reference. Example tests
also verify edge deduplication and exhaustive pair agreement under hash collisions.
Pre-commit checks passed, and the API documentation built without warnings.

## Reproduce

From the repository root, with Warp already built:

```bash
env -u DISPLAY uv run --with polyscope==2.6.1 \
  -m warp.examples.geometry.example_mesh_intersect_hash \
  --device cuda:0 \
  --screenshot intersection-preview/candidates.png \
  --stats intersection-preview/stats.json
```

Polyscope is optional and loaded only by the renderer. `--device cpu` runs the
same intersection calculation on the CPU. The example also accepts
`--subdivisions`, `--cell-size`, and `--grid-dim`.

## Method and validation

The broad phase stores triangle bounding-sphere centers from mesh B in
`wp.HashGrid`. Each triangle of A queries with its radius plus the largest radius
in B, followed by a per-pair sphere-overlap filter. This accounts for triangles
spanning multiple cells. A small `1e-5` padding handles roundoff in these unit-scale
fixtures.

The narrow phase now calls [`warp.geometry.intersect_tri_tri()`](../warp/_src/geometry/intersect.py),
implemented entirely with Warp functions. It returns `bool`, supports `vec3` and `vec3d`, retains
the native builtin's coplanar handling and plane tolerance, and makes no call to
native `intersect_tri_tri`. The existing `wp.intersect_tri_tri()` remains unchanged
and serves as the independent reference.

The [tests](../warp/tests/geometry/test_mesh_intersect_hash.py) compare every
intersection pair with an independent exhaustive kernel calling the existing
builtin. They cover this screenshot's scene, smaller overlapping and separated
boxes, containment, unequal triangle sizes, hash collisions, negative coordinates,
coplanar overlap, and vertex contact. They also reject duplicate candidate visits.

```bash
uv run -m unittest -v \
  warp.tests.geometry.test_mesh_intersect_hash \
  warp.tests.geometry.test_hash_grid \
  warp.tests.test_intersect \
  warp.tests.test_closest_point_edge_edge
```

**92 tests passed across CPU and GPU**, including randomized pairs, all three
coplanar projection planes, vertex permutations, swapped triangles, contact cases,
and plane-tolerance boundaries in both precisions. Pre-commit checks passed.
The API documentation build completed without warnings.

This small testbed intentionally uses dense pair masks, so its output memory is
quadratic in mesh triangle counts. It detects surface intersections; a fully
contained mesh can have no intersecting triangles.


## Pure Warp versus native performance

Measured on the GPU above with
[`benchmark_intersect_tri_tri.py`](../warp/examples/benchmarks/benchmark_intersect_tri_tri.py).
These are per-launch GPU times for 262,144 pairs, using the median of 21 samples.
Each sample times a CUDA graph of 30 repeated launches; native/pure measurement
order alternates. Compilation, data preparation, transfers, and the hash broad
phase are excluded. All benchmark outputs are checked against the native builtin
before timing. Timings apply to this machine and workload, not every GPU or input.

| Precision | Workload | Native (µs) | Pure Warp (µs) | Speedup |
| --- | --- | ---: | ---: | ---: |
| float32 | random | 31.59 | 19.46 | 1.623× |
| float32 | separated | 30.92 | 19.16 | 1.613× |
| float32 | coplanar | 32.79 | 21.17 | 1.549× |
| float32 | mesh candidates | 31.25 | 19.36 | 1.614× |
| float64 | random | 213.64 | 181.32 | 1.178× |
| float64 | separated | 137.97 | 132.28 | 1.043× |
| float64 | coplanar | 497.45 | 483.06 | 1.030× |
| float64 | mesh candidates | 203.92 | 176.87 | 1.153× |

The random workload samples both triangles in the same unit box; separated pairs
translate the second triangle by 4 along X; coplanar pairs share Z=0; mesh
candidates repeat shuffled candidate pairs from the screenshot scene. The latter
is a throughput workload with repeated geometry, not an end-to-end broad-phase
measurement.

Additional batch sizes also showed parity or improvement in all measured cases:

| Pairs | float32 speedup range | float64 speedup range | Raw samples |
| --- | --- | --- | --- |
| 8,192 | 1.001–1.024× | 1.032–1.156× | [small](benchmark-small.json) |
| 1,048,576 | 1.025–1.027× | 1.014–1.169× | [large](benchmark-large.json) |

The smallest float32 cases are effectively tied at about 3 µs per launch; differences
near 1× should be treated as timing noise. Larger batches can be limited by memory
bandwidth. No timing-based assertions are added to the test suite.

Main-run samples: [benchmark.json](benchmark.json). Reproduce with:

```bash
uv run -m warp.examples.benchmarks.benchmark_intersect_tri_tri \
  --samples 21 --repeats 30 --output intersection-preview/benchmark.json
```
