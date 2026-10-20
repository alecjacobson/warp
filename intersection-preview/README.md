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

The narrow phase still uses the existing native `wp.intersect_tri_tri()`, through
an example-local `@wp.func` replacement point. No new geometry API or pure Warp
intersection predicate has been added.

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

**86 tests passed across CPU and GPU.** Pre-commit checks passed.

This small testbed intentionally uses dense pair masks, so its output memory is
quadratic in mesh triangle counts. It detects surface intersections; a fully
contained mesh can have no intersecting triangles.
