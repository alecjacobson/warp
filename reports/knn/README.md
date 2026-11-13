# kNN branch reports

**Start here: [Current comparisons and recommended settings](comparisons.md).**

Updated **2026-11-13** for `alecjacobson/knn`. Construction, reused queries, query graphs, Chamfer forward/backward, and retained memory were measured again against the completed public API. The source and native binary also match the retained standalone-integration evidence. [Provenance](current/provenance.json) records that check.

| Read | Purpose |
| --- | --- |
| **[Current comparisons](comparisons.md)** | Which public settings to use, latest timings, memory, methodology, and limits. |
| **[Public API guide](public-api.md)** | `wp.build_point_index()`, `wp.knn()`, batches, fixed targets, gradients, graphs, and scalar queries inside kernels. |
| [All current timings](current/timings.md) | Every workload and configuration, generated from the saved samples. |
| [Standalone integration and validation](completion/README.md) | Public-wrapper overhead versus standalone cuBQL, native review, and the completed test evidence. |
| [Historical archive](archive/README.md) | Earlier implementations, tuning studies, and external-library comparisons, with their original evidence preserved. |

The optional direct-cuBQL route and fixed-size independent batches are implemented; [#20](https://github.com/alecjacobson/warp/issues/20) and [#21](https://github.com/alecjacobson/warp/issues/21) are closed. All supported routes are exposed through Warp. `wp.knn()` is the current array API; `wp.point_query_knn_batch()` remains a deprecated compatibility alias. Scalar kernel queries use `while wp.point_query_next(...)`; direct `for index in search` iteration is not implemented.

The current evidence covers one **94-SM Blackwell MIG slice**. Earlier **46-SM** measurements are historical and are not pooled with current results. Broader hardware validation, peak training memory, larger-batch optimization, and encoder integration remain follow-up work. Ragged batches remain outside this API.
