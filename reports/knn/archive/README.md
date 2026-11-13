# Historical kNN reports

**Use [current comparisons](../comparisons.md) and the [public API guide](../public-api.md) for this branch.** This archive preserves earlier measurements, decisions, scripts, and logs. A statement such as “current,” “latest,” or “implemented” inside a snapshot refers to that snapshot's recorded state.

| Snapshot | What it preserves | Why it is historical |
| --- | --- | --- |
| [Original baseline](baseline/README.md) | Original top-level report, raw outputs, HTML, charts, and profiling. | Predates the completed public API; includes 46-SM results and the old batch API name. |
| [Optimization](optimization/README.md) | Earlier selection, traversal, and tuning experiments. | Development evidence for earlier implementations. |
| [External-library comparison](comparison/README.md) | Earlier comparisons with standalone implementations and external libraries. | Uses the earlier implementation and 46-SM hardware; not a current cross-library ranking. |
| [Hybrid tuning](hybrid/README.md) | Tree/launch policy exploration and cuBQL candidate measurements. | Supports policy history, not a new measurement of the public integration. |
| [Initial array API](batch/README.md) | Early array-query work and validation. | Uses the older API naming and scope. |
| [Historical refresh](refresh/README.md) | An earlier comparison refresh and its provenance. | “Refresh” is the original snapshot name, not the latest report. |
| [Initial batching and native integration](batched-cubql/README.md) | First supplied-pulmonary measurements of construction, queries, Chamfer, and memory on 94 SMs. | Predates final API/native cleanup; superseded by [fresh measurements](../current/timings.md). |

Reports now carry historical notices and repaired local navigation. Raw result files, logs, scripts, and images retain their original bytes. [manifest.json](manifest.json) records their original paths and SHA256 values before the move, from branch commit [93e506474](https://github.com/alecjacobson/warp/commit/93e506474cdc0e10e57b875a60e986f6f5fbf7f9).

Archived reproduction commands, scripts, and provenance strings retain their original checkout paths and assumptions. Reproduce a historical experiment at its recorded revision, using the original directory layout; do not treat these commands as the current workflow. Some experiments require external libraries or local data not stored in Git. Use the [current reproduction instructions](../comparisons.md#reproduce-and-inspect) for the completed API.

The [standalone integration and validation evidence](../completion/README.md) remains outside this archive because its recorded source and native binary hashes still match the current implementation. Its measurement methodology differs from the fresh workload run, as explained in the current comparison report.
