# Comparison with Newton's analytic capsule queries

The analytic capsule path does **not** offer the large speedup measured against
Newton's iterative SDF optimizer. On this machine, the extracted axis solve is
within 2%, while replacing the nonparallel solve inside complete contact
generation is generally 1–3% slower. The existing segment-witness helper has
mixed performance and a significant small-scale accuracy limitation.

No public Warp API or Newton implementation was changed for this comparison.
The hybrid is a benchmark adapter that preserves Newton's contact policy.

## Sources and method

The feature branch was pulled through `c2c7a667e`. Its incoming changes moved
research reports to this separate branch. Newton was fetched through
[`03b36585233ef0ae5dcdb7ebdd9a40a5e127c330`](https://github.com/newton-physics/newton/tree/03b36585233ef0ae5dcdb7ebdd9a40a5e127c330).
The compared routines in `collision_primitive.py` are unchanged from our earlier
`8a656068` reference.

Three comparisons are kept separate:

1. **Axis closest pair:** Newton's extracted nonparallel formulas versus
   `warp.geometry.closest_points_seg_seg`, including conversion from the capsule
   centers/directions/half-lengths to segment endpoints. Both return two parameters
   in [0,1] and distance. This measures the geometric core, not a complete contact.
2. **Complete capsule contact:** the unmodified imported
   [`collide_capsule_capsule`](https://github.com/newton-physics/newton/blob/03b36585233ef0ae5dcdb7ebdd9a40a5e127c330/newton/_src/geometry/collision_primitive.py#L229)
   versus an adapter replacing only its nonparallel solve. Both write two distance
   slots, two positions, and one shared normal. The determinant threshold, parallel
   endpoint projections, and sphere-contact postprocessing are retained.
3. **Segment witnesses:** the unmodified imported
   [`closest_segment_to_segment_points`](https://github.com/newton-physics/newton/blob/03b36585233ef0ae5dcdb7ebdd9a40a5e127c330/newton/_src/geometry/collision_primitive.py#L113)
   versus our parameters reconstructed into two witnesses. Both write the two
   witnesses and their distance. This helper was missed in the original audit;
   no callers were found in the inspected Newton checkout. The capsule routine,
   in contrast, is used by the primitive narrow phase and Kamino.

Hardware: **NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb**, driver
595.91.07, CUDA toolkit 12.9, Warp 1.20.0.dev0. Float32, **262,144 pairs**, seed
2360, block size 128. Each run takes the median of 21 CUDA-event samples of
30-launch graphs, alternating routine order. Tables give the median of three
complete runs. Compilation, allocation, transfers, and reference evaluation
are excluded. Broad phase, transforms from world to capsule coordinates, contact
filtering/compaction/atomics, and solver execution are not measured.

Contact and witness components use structure-of-arrays output for coalesced
stores; the three axis outputs use `vec3`. Output layout materially affects
absolute timings, so these are kernel benchmarks with specified inputs/outputs,
not isolated arithmetic costs or end-to-end solver predictions. A packed-record
pilot was output-layout sensitive; the tables below consistently use the
coalesced layout on both sides.

## Complete capsule contact generation

Time change is relative to the original Newton routine. Positive means slower.

| Workload | Original Newton (µs) | Hybrid (µs) | Time change |
| --- | ---: | ---: | ---: |
| random | 17.392 | 17.687 | +1.70% |
| near | 17.395 | 17.667 | +1.56% |
| endcap | 17.408 | 17.875 | +2.68% |
| parallel, equal lengths | 17.204 | 17.437 | +1.35% |
| parallel, varied lengths | 17.668 | 17.458 | −1.19% |
| contained parallel axes | 17.988 | 18.023 | +0.20% |
| nearly parallel | 17.457 | 17.887 | +2.46% |
| small (random scaled by 1e-4) | 17.327 | 17.614 | +1.66% |
| one axis collapsed | 17.939 | 18.007 | +0.38% |

The self-contained ASV reproduction was also measured (`copy` in the JSON).
Its records matched the original source exactly on the measured GPU batches,
but its timings are not substituted for original-source timings.

Across every batch, original and hybrid selected the same one-/two-candidate
branch, and records in their shared parallel branch matched exactly. Differences
in the nonparallel branch can include the selected location of a nonunique
minimum, even when distance agrees; this is not a solver-compatibility proof.

## Axis closest-pair calculation

These rows use the extracted original formulas, not a separately exported Newton
closest-pair API. Both sides consume the same capsule representation. Rows with
any pair routed through the parallel branch are omitted, because the extracted
nonparallel solve would not be a complete equivalent query there.

| Workload | Newton extracted (µs) | Warp (µs) |
| --- | ---: | ---: |
| random | 11.469 | 11.539 |
| near | 11.465 | 11.531 |
| endcap | 11.397 | 11.530 |
| parallel, varied lengths, classified nonparallel | 11.265 | 11.453 |

The last row is intentional: floating-point residual in the Gram determinant
can exceed Newton's fixed `1e-15` threshold even for geometrically parallel axes.
All pairs in this varied-length batch followed its nonparallel branch. For the
equal-length and contained-axis fixtures we choose exactly representable lengths,
which reliably exercise the two-candidate branch instead. This illustrates why
the two-contact policy should not be assumed to activate for every parallel pair.

## Existing pure Warp segment-witness helper

| Workload | Newton helper (µs) | Warp (µs) | Warp time change |
| --- | ---: | ---: | ---: |
| random | 12.272 | 12.961 | +5.61% |
| near | 12.233 | 12.905 | +5.49% |
| endcap | 12.218 | 12.896 | +5.55% |
| parallel, equal lengths | 13.859 | 12.891 | −6.98% |
| parallel, varied lengths | 13.850 | 12.819 | −7.44% |
| contained parallel axes | 13.923 | 12.828 | −7.87% |
| nearly parallel | 12.341 | 12.917 | +4.67% |
| small | 12.292 | 12.964 | +5.47% |
| one axis collapsed | 13.521 | 12.684 | −6.19% |

This comparison does not establish a uniform speed advantage. Unlike our query,
the Newton helper regularizes both its angular denominator and point–segment
projection denominators with fixed `1e-6` terms.

## Accuracy and contact-policy limitations

Each workload checks 256 reproducibly selected pairs against an independent
active-set convex quadratic-program oracle with rational arithmetic. Oracle
geometry is reconstructed in float64 from the quantized float32 capsule inputs;
measured error includes rounding in float32 endpoint construction and witness
reconstruction. Errors below are maximum absolute distance errors on that subset,
not error bounds or witness/normal accuracy guarantees.

| Workload | Newton witness helper | Warp reconstructed witnesses |
| --- | ---: | ---: |
| random | 1.88e-7 | 1.68e-7 |
| near | 1.20e-7 | 9.59e-8 |
| endcap | 2.72e-7 | 1.63e-7 |
| nearly parallel | 9.08e-8 | 9.09e-8 |
| small (scale 1e-4) | **7.80e-5** | **1.71e-11** |

Two separate limitations affect the capsule contact routine:

- **Contained parallel axes can miss overlap.** Take axis A from `(0,0,-2)` to
  `(0,0,2)`, axis B from `(0,0.15,-0.5)` to `(0,0.15,0.5)`, and both radii 0.1.
  The global signed separation is `0.15 - 0.2 = -0.05`. Newton tests A's endpoints
  against B and returns approximately **+1.30748** in both slots. Our closest-pair
  query gives −0.05; the policy-preserving hybrid retains Newton's wrong candidates.
  All 256 oracle samples in this workload confirm missed overlaps. This is a
  standalone-routine result; a full simulation was not run.
- **The fixed determinant threshold depends on scale.** Scaling the random
  geometry by 1e-4 routes every pair through the parallel branch, although their
  directions remain random. The maximum error of the minimum contact-candidate
  distance is 3.44e-5, while the Warp closest-distance error is 1.71e-11. One
  missed overlap in the 256-pair subset is confirmed beyond a 3e-6 exclusion band.

Contact-candidate minimum error is not being presented as the error of Newton's
separate segment-witness helper. These are different routines with different
policies. Correcting the capsule manifold requires changing that policy, not
just substituting a closest-pair formula inside its nonparallel branch.

## Validation and reproduction

CPU/CUDA checks validate the copied contact routine against the original, the
hybrid's exact-parallel/contained/collapsed behavior against the original, and
Warp closest distances against the rational oracle. Extracted nonparallel
distances are checked against original capsule distances after radius subtraction.
ASV discovery and setup/time smoke checks pass, as does pre-commit. No library
implementation changes or timing assertions were added.

From the feature checkout, with Warp built:

```bash
git clone https://github.com/newton-physics/newton.git /path/to/newton-reference
git -C /path/to/newton-reference checkout 03b36585233ef0ae5dcdb7ebdd9a40a5e127c330
uv run asv/benchmarks/closest_capsule.py --newton-source /path/to/newton-reference --output capsule-analytic.json
```

Repeat three times for the aggregate tables above. The baseline is imported
from the pinned checkout; ASV's `closest_capsule.CapsuleContacts` uses the
self-contained reference so Newton is not a required ASV dependency.

Raw runs include source checksums and individual timings:
[run 1](capsule-analytic.json), [run 2](capsule-analytic-repeat2.json), and
[run 3](capsule-analytic-repeat3.json).

The remaining optimization opportunity established by our earlier measurements
is Newton's iterative triangle/segment–capsule SDF path. This analytic capsule
comparison supplies no basis for claiming the same large speedups elsewhere.
