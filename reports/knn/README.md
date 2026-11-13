# kNN branch reports

**Start here: [Current comparisons and recommended settings](comparisons.md).**

Updated **2026-11-13** for `alecjacobson/knn`. **Six colored tables** cover single-cloud queries, native batched queries, and fixed-target Chamfer with backward on the 94-SM GPU. The four single-cloud tables compare 16 configurations across 40 workloads. Two new tables compare eight query configurations across 24 workloads and nine Chamfer configurations across 21 workloads, including unchanged PyTorch, PyTorch3D, and KeOps APIs with supported tuning. Failed checks and runtime errors stay visible. Retained public-only measurements also cover construction, query graphs, and live allocations.

| Read | Purpose |
| --- | --- |
| **[Current comparisons](comparisons.md)** | Colored cross-library tables, public settings to use, pulmonary/Chamfer results, memory, methodology, and limits. |
| **[Public API guide](public-api.md)** | `wp.build_point_index()`, `wp.knn()`, batches, fixed targets, gradients, graphs, and scalar queries inside kernels. |
| [Cross-library timing tables](current/libraries/tables.md) | Aggregates for all 16 configurations; [CSV](current/libraries/cases.csv) contains every median and status. |
| [Cross-library reproduction](current/libraries/reproduce.md) | Dependency/build pins, worker commands, source/data provenance, and chart generation. |
| **[Batched query and Chamfer comparisons](comparisons.md#batched-queries-and-chamfer-against-external-libraries)** | Two colored tables, native external batching, independent tuning, and full decoded-gradient checks. |
| [Batched cross-library reproduction](current/training/reproduce.md) | Settings, timing contract, failure isolation, raw evidence, and [every median/status](current/training/cases.csv). |
| [Pulmonary/batched timings](current/timings.md) | The separate 24-workload, three-configuration experiment. |
| [Standalone integration and validation](completion/README.md) | Public-wrapper overhead versus standalone cuBQL, native review, and the completed test evidence. |
| [Historical archive](archive/README.md) | Earlier implementations, tuning studies, and external-library comparisons, with their original evidence preserved. |

The optional direct-cuBQL route and fixed-size independent batches are implemented; [#20](https://github.com/alecjacobson/warp/issues/20) and [#21](https://github.com/alecjacobson/warp/issues/21) are closed. All supported routes are exposed through Warp. `wp.knn()` is the current array API; `wp.point_query_knn_batch()` remains a deprecated compatibility alias. Scalar kernel queries use `while wp.point_query_next(...)`; direct `for index in search` iteration is not implemented.

The current evidence covers one **94-SM Blackwell MIG slice**. Earlier **46-SM** measurements are historical and are not pooled with current results; the standalone cuBQL policy retains its independently trained settings. The source/native hashes also match the retained integration evidence. Broader hardware validation, peak training memory, larger-batch optimization, and encoder integration remain follow-up work. Ragged batches remain outside this API.
