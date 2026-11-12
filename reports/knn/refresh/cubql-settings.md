# Selected cuBQL findKNN settings

Settings were frozen on separate training inputs before the 40-case evaluation. Build-plus-query and reused-index objectives select independent trees. These are the best candidates tested in the documented hierarchical search.

| Regime | Objective | Builder | Leaf threshold | Thread stride | Block size | Query ordering |
| --- | --- | --- | ---: | ---: | ---: | --- |
| N=1K, Q=32, k=32 | Build + query | radix | 32 | 32 | 256 | False |
| N=1K, Q=32, k=32 | Reused index | elh | 8 | 32 | 128 | False |
| N=100K, Q=1,024, k=1 | Build + query | rebin_radix | 64 | 32 | 32 | False |
| N=100K, Q=1,024, k=1 | Reused index | elh | 0 | 32 | 64 | False |
| N=100K, Q=1,024, k=32 | Build + query | radix | 16 | 32 | 256 | False |
| N=100K, Q=1,024, k=32 | Reused index | elh | 8 | 32 | 32 | False |
| N=1M, Q=8,192, k=11 | Build + query | rebin_radix | 32 | 8 | 256 | False |
| N=1M, Q=8,192, k=11 | Reused index | elh | 8 | 8 | 256 | False |

Leaf threshold 0 leaves the choice to the upstream builder. Defaults use spatial median with resolved leaf threshold 1, stride 1, block size 256 and no ordering. Every selected setting disables ordering after its full cost was measured.

| Regime | Builder/leaf candidates | Launch candidates | Remeasured finalists |
| --- | ---: | ---: | ---: |
| N=1K, Q=32, k=32 | 40 | 240 | 7 |
| N=100K, Q=1,024, k=1 | 40 | 240 | 7 |
| N=100K, Q=1,024, k=32 | 40 | 240 | 7 |
| N=1M, Q=8,192, k=11 | 40 | 240 | 7 |
