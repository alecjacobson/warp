# Matched Warp kNN builder timings

**Historical snapshot.** Measurements, API names, and status below describe the recorded implementation and hardware. Use the [current comparisons](../../comparisons.md) for this branch. Archived reproduction commands assume their original checkout and paths.

Both variants use the same Warp batch query kernels and requested leaf/launch settings. Only the BVH constructor changes. Values below compare fresh paired measurements; the main colored plots retain the earlier LBVH batch measurement as their common denominator.

| Operation | cuBQL SAH relative to LBVH | cuBQL SAH lower median |
| --- | ---: | ---: |
| Build + query, ordinary call | 13.97x slower | 0/40 |
| Reused query, ordinary call | 1.07x faster | 31/40 |
| Reused query, CUDA graph replay | 1.09x faster | 33/40 |

Geometric means over 40 cases. Counts include small differences and are not significance tests. Full-build graph capture succeeds for LBVH; cuBQL construction fails the isolated probe because it synchronizes with the host. Both query graphs pass replay checks after query values change.

| Case | LBVH build + query ms | cuBQL build + query ms | LBVH reused ms | cuBQL reused ms | LBVH query graph ms | cuBQL query graph ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bunny_scan-n100000-q1024-sigma0-k1 | 0.1948 | 3.7042 | 0.0747 | 0.0664 | 0.0323 | 0.0301 |
| bunny_scan-n100000-q1024-sigma0-k32 | 0.3736 | 3.8239 | 0.2437 | 0.2203 | 0.1991 | 0.1818 |
| bunny_scan-n100000-q1024-sigma0.01-k1 | 0.1890 | 3.8302 | 0.0774 | 0.0711 | 0.0392 | 0.0341 |
| bunny_scan-n100000-q1024-sigma0.01-k32 | 0.4090 | 3.9457 | 0.2807 | 0.2429 | 0.2433 | 0.2033 |
| kitti-n100000-q1024-sigma0-k1 | 0.2388 | 6.2225 | 0.0711 | 0.0658 | 0.0325 | 0.0302 |
| kitti-n100000-q1024-sigma0-k32 | 0.5023 | 6.4809 | 0.2765 | 0.2189 | 0.2392 | 0.1817 |
| kitti-n100000-q1024-sigma0.01-k1 | 0.2519 | 6.3367 | 0.0894 | 0.0770 | 0.0508 | 0.0404 |
| kitti-n100000-q1024-sigma0.01-k32 | 0.5023 | 6.5568 | 0.2958 | 0.2641 | 0.2534 | 0.2289 |
| redwood-n100000-q1024-sigma0-k1 | 0.2375 | 5.9274 | 0.0729 | 0.0684 | 0.0344 | 0.0323 |
| redwood-n100000-q1024-sigma0-k32 | 0.4414 | 6.1522 | 0.2308 | 0.2549 | 0.1921 | 0.2186 |
| redwood-n100000-q1024-sigma0.01-k1 | 0.2421 | 5.9536 | 0.0803 | 0.0720 | 0.0428 | 0.0355 |
| redwood-n100000-q1024-sigma0.01-k32 | 0.5169 | 6.2459 | 0.3103 | 0.2696 | 0.2721 | 0.2310 |
| bunny_surface-n100000-q1024-sigma0-k1 | 0.2270 | 5.9148 | 0.0721 | 0.0667 | 0.0334 | 0.0294 |
| bunny_surface-n100000-q1024-sigma0-k32 | 0.4523 | 6.1128 | 0.2342 | 0.2221 | 0.1957 | 0.1823 |
| bunny_surface-n100000-q1024-sigma0.01-k1 | 0.2361 | 5.9038 | 0.0772 | 0.0793 | 0.0403 | 0.0426 |
| bunny_surface-n100000-q1024-sigma0.01-k32 | 0.5016 | 6.1601 | 0.2950 | 0.2426 | 0.2534 | 0.2001 |
| sphere-n100000-q1024-sigma0-k1 | 0.2321 | 5.7647 | 0.0709 | 0.0658 | 0.0325 | 0.0301 |
| sphere-n100000-q1024-sigma0-k32 | 0.4405 | 5.9597 | 0.2194 | 0.2174 | 0.1854 | 0.1779 |
| sphere-n100000-q1024-sigma0.01-k1 | 0.2379 | 5.8486 | 0.0774 | 0.0717 | 0.0388 | 0.0343 |
| sphere-n100000-q1024-sigma0.01-k32 | 0.4713 | 6.1201 | 0.2635 | 0.2367 | 0.2218 | 0.2000 |
| volume-n100000-q1024-sigma0-k1 | 0.2381 | 5.9065 | 0.0797 | 0.0932 | 0.0424 | 0.0421 |
| volume-n100000-q1024-sigma0-k32 | 0.4869 | 6.1066 | 0.2851 | 0.2791 | 0.2485 | 0.2433 |
| volume-n100000-q1024-sigma0.01-k1 | 0.2372 | 5.6799 | 0.0772 | 0.0752 | 0.0426 | 0.0396 |
| volume-n100000-q1024-sigma0.01-k32 | 0.4824 | 5.9625 | 0.2753 | 0.2635 | 0.2414 | 0.2260 |
| clusters-n100000-q1024-sigma0-k1 | 0.2474 | 6.4713 | 0.0840 | 0.0742 | 0.0466 | 0.0376 |
| clusters-n100000-q1024-sigma0-k32 | 0.5357 | 6.8022 | 0.3078 | 0.3347 | 0.2713 | 0.2955 |
| clusters-n100000-q1024-sigma0.01-k1 | 0.2550 | 6.4705 | 0.0906 | 0.0781 | 0.0508 | 0.0406 |
| clusters-n100000-q1024-sigma0.01-k32 | 0.5625 | 6.8255 | 0.3501 | 0.3059 | 0.3086 | 0.2704 |
| thin_plane-n100000-q1024-sigma0-k1 | 0.2241 | 5.7894 | 0.0690 | 0.0660 | 0.0303 | 0.0299 |
| thin_plane-n100000-q1024-sigma0-k32 | 0.4225 | 5.9955 | 0.2152 | 0.2184 | 0.1778 | 0.1777 |
| thin_plane-n100000-q1024-sigma0.01-k1 | 0.2342 | 5.7823 | 0.0773 | 0.0728 | 0.0395 | 0.0352 |
| thin_plane-n100000-q1024-sigma0.01-k32 | 0.4819 | 6.0332 | 0.2755 | 0.2495 | 0.2367 | 0.2101 |
| sphere-n1000000-q8192-sigma0-k11 | 1.5853 | 21.3522 | 0.4413 | 0.3407 | 0.3945 | 0.3009 |
| sphere-n1000000-q8192-sigma0.01-k11 | 1.8532 | 21.8655 | 0.7294 | 0.5419 | 0.6912 | 0.4999 |
| volume-n1000000-q8192-sigma0-k11 | 1.7783 | 21.3745 | 0.6482 | 0.5887 | 0.6163 | 0.5637 |
| volume-n1000000-q8192-sigma0.01-k11 | 1.7992 | 21.2495 | 0.6697 | 0.5655 | 0.6336 | 0.5271 |
| sphere-n1000-q32-sigma0-k32 | 0.1837 | 0.5292 | 0.1396 | 0.1545 | 0.1010 | 0.1172 |
| sphere-n1000-q32-sigma0.01-k32 | 0.1805 | 0.5196 | 0.1376 | 0.1386 | 0.0994 | 0.1017 |
| volume-n1000-q32-sigma0-k32 | 0.2216 | 0.5633 | 0.1765 | 0.1920 | 0.1385 | 0.1551 |
| volume-n1000-q32-sigma0.01-k32 | 0.2171 | 0.5419 | 0.1714 | 0.1812 | 0.1331 | 0.1419 |
