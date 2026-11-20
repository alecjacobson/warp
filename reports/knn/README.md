# kNN branch reports

**Matched PyTorch comparison (2026-11-20): [B=32 pulmonary Chamfer versus `torch.cdist`](cdist-batch/README.md).** Includes chunk-size sensitivity, fixed-target reuse versus building both indexes, and loss/gradient checks on the 46-SM GPU.

**Code sync (2026-11-20):** The kNN implementation, tests, and user guide on `alecjacobson/knn` now match `alecjacobson/knn-review` at [`25c78fe`](https://github.com/alecjacobson/warp/commit/25c78fe). This includes fused batch queries and gradient gathers, dense-batch cuBQL scheduling, and the review fixes. Existing users can stay on this branch; rebuild the native library after pulling. The reports below retain their original measured revisions and timings.

**Latest review (2026-11-16): [PR validation and regression report](review-2348/README.md).** The reviewed implementation is on [`alecjacobson/knn-review`](https://github.com/alecjacobson/warp/tree/alecjacobson/knn-review); this research branch retains its experiment history.

**Start here: [Current comparisons and recommended settings](comparisons.md).**

Updated **2026-11-15** for `alecjacobson/knn`. The branch is rebased onto NVIDIA main `3afdd69addbf`. The recommended array API is now `warp.geometry.PointIndex` plus `warp.geometry.knn()`; core scalar queries continue to use `wp.Bvh`. **[API/main regression validation](api-revision/README.md)** records the matched before/after measurements and current tests.

**Six colored tables**, measured November 13, cover single-cloud queries, native batched queries, and fixed-target Chamfer with backward on the 94-SM GPU. The four single-cloud tables compare 16 configurations across 40 workloads. Two new tables compare eight query configurations across 24 workloads and nine Chamfer configurations across 21 workloads, including unchanged PyTorch, PyTorch3D, and KeOps APIs with supported tuning. Failed checks and runtime errors stay visible. Retained public-only measurements also cover construction, query graphs, and live allocations.

| Read | Purpose |
| --- | --- |
| **[Current comparisons](comparisons.md)** | Colored cross-library tables, public settings to use, pulmonary/Chamfer results, memory, methodology, and limits. |
| **[API/main regression report](api-revision/README.md)** | Matched before/after timings, main-rebase provenance, and current validation. |
| **[Public API guide](public-api.md)** | `wp.Bvh` kernel queries, `wp.geometry.PointIndex`, `wp.geometry.knn()`, fixed targets, gradients, graphs, and migration. |
| [Cross-library timing tables](current/libraries/tables.md) | Aggregates for all 16 configurations; [CSV](current/libraries/cases.csv) contains every median and status. |
| [Cross-library reproduction](current/libraries/reproduce.md) | Dependency/build pins, worker commands, source/data provenance, and chart generation. |
| **[Batched query and Chamfer comparisons](comparisons.md#batched-queries-and-chamfer-against-external-libraries)** | Two colored tables, native external batching, independent tuning, and full decoded-gradient checks. |
| [Batched cross-library reproduction](current/training/reproduce.md) | Settings, timing contract, failure isolation, raw evidence, and [every median/status](current/training/cases.csv). |
| [Pulmonary/batched timings](current/timings.md) | The separate 24-workload, three-configuration experiment. |
| [Standalone integration and validation](completion/README.md) | Public-wrapper overhead versus standalone cuBQL, native review, and the completed test evidence. |
| [Historical archive](archive/README.md) | Earlier implementations, tuning studies, and external-library comparisons, with their original evidence preserved. |

The optional direct-cuBQL route and fixed-size independent batches are implemented; [#20](https://github.com/alecjacobson/warp/issues/20) and [#21](https://github.com/alecjacobson/warp/issues/21) are closed. All supported routes are exposed through Warp. `wp.geometry.knn()` is the current array API and always returns a `PointIndex` owner first. The earlier top-level functions and types remain deprecated compatibility entry points with their original signatures and returns. [#22](https://github.com/alecjacobson/warp/issues/22) tracks the API revision and main update. Scalar kernel queries use `while wp.point_query_next(...)`; direct `for index in search` iteration is not implemented.

The current evidence covers one **94-SM Blackwell MIG slice**. Earlier **46-SM** measurements are historical and are not pooled with current results; the standalone cuBQL policy retains its independently trained settings. The external-library and integration evidence retains its original source/native hashes; the API move and main rebase are validated separately against a new pre-change baseline. Broader hardware validation, peak training memory, larger-batch optimization, and encoder integration remain follow-up work. Ragged batches remain outside this API.
