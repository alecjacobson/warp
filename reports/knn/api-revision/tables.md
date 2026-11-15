# Matched before/after measurements

Positive percentages mean longer runtime. Ratios are geometric means of per-case medians. Times are synchronized wall measurements, including Python and allocation work.

| Phase | Comparisons | New API / before | Engine / before | New API / engine | Median added time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Build + query | 72 | -0.05% | -0.39% | +0.34% | +0.53 µs |
| Reused query | 72 | +0.58% | +0.10% | +0.48% | +1.01 µs |
| Query graph | 72 | +0.28% | +0.34% | -0.05% | +0.14 µs |
| Chamfer, same backend | 63 | +0.41% | +0.13% | +0.28% | +4.28 µs |
| Chamfer, mixed backends | 21 | +0.58% | +0.26% | +0.32% | +2.52 µs |

[Every median and ratio](cases.csv). These are separate sequential runs; small differences are not confidence bounds or universal guarantees.
