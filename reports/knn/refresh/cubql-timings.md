# cuBQL execution-mode comparison

Geometric means across all 40 passing cases. Warp means the automatic batch API. Each row compares the same execution mode. No full-construction cuBQL graph is claimed.

| Execution and objective | Defaults / tuned cuBQL | Tuned cuBQL faster than defaults | Warp / tuned cuBQL | Tuned cuBQL faster than Warp |
| --- | ---: | ---: | ---: | ---: |
| Ordinary build + query | 4.56x | 40/40 | 0.66x | 4/40 |
| Ordinary reused query | 6.18x | 40/40 | 1.43x | 39/40 |
| CUDA graph reused query | 7.11x | 40/40 | 1.19x | 34/40 |

A ratio above one favors tuned cuBQL. Build-plus-query and query-only use separately selected, frozen settings. Graphs use the query settings chosen from ordinary-call training; capture-specific tuning was not performed. Warp samples are retained from the previous update on the same GPU and environment, so small differences remain subject to run-to-run variation.
