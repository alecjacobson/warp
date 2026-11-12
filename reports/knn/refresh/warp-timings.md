# Fresh Warp timings: ordinary calls and CUDA graph replay

All ratios below are geometric means across the same 40 cases. Greater than one means faster than the simple scalar ordinary call. No per-case fastest-path selection is used.

| Warp path | Execution | Speedup vs simple scalar, build + query | Speedup vs simple scalar, reused index |
| --- | --- | ---: | ---: |
| Warp scalar (simple dense launch) | ordinary call | 1.00x | 1.00x |
| Warp scalar (simple dense launch) | CUDA graph replay | 1.08x | 1.17x |
| Warp scalar (tuned launch) | ordinary call | 2.85x | 4.04x |
| Warp scalar (tuned launch) | CUDA graph replay | 3.31x | 5.44x |
| Warp batch (automatic tuning) | ordinary call | 2.81x | 3.90x |
| Warp batch (automatic tuning) | CUDA graph replay | 3.33x | 5.68x |

Graph replay includes a fresh GPU BVH construction for build + query; reused-index replay queries a prebuilt BVH. Output buffers are preallocated for graphs. Capture, graph instantiation, compilation and input upload are excluded. Sorting and graph scratch operations, when selected, remain part of replay. Each sample is one replay plus synchronization, using the same timer as the ordinary-call comparison.

| Case | Simple scalar build + query (ms) | Automatic batch build + query (ms) | Batch graph build + query (ms) | Simple scalar reused (ms) | Automatic batch reused (ms) | Batch graph reused (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bunny_scan-n100000-q1024-sigma0-k1 | 0.2509 | 0.2216 | 0.1688 | 0.0897 | 0.0711 | 0.0320 |
| bunny_scan-n100000-q1024-sigma0-k32 | 2.4832 | 0.4150 | 0.3439 | 2.3410 | 0.2389 | 0.2002 |
| bunny_scan-n100000-q1024-sigma0.01-k1 | 0.2740 | 0.2251 | 0.1755 | 0.1093 | 0.0773 | 0.0402 |
| bunny_scan-n100000-q1024-sigma0.01-k32 | 3.1355 | 0.4472 | 0.3887 | 2.9718 | 0.2829 | 0.2428 |
| kitti-n100000-q1024-sigma0-k1 | 0.3128 | 0.2697 | 0.2144 | 0.0951 | 0.0725 | 0.0337 |
| kitti-n100000-q1024-sigma0-k32 | 2.5583 | 0.5261 | 0.4774 | 2.3494 | 0.2805 | 0.2391 |
| kitti-n100000-q1024-sigma0.01-k1 | 0.3393 | 0.2826 | 0.2260 | 0.1186 | 0.0891 | 0.0509 |
| kitti-n100000-q1024-sigma0.01-k32 | 3.5484 | 0.5408 | 0.4750 | 3.3178 | 0.2987 | 0.2549 |
| redwood-n100000-q1024-sigma0-k1 | 0.3176 | 0.2651 | 0.2114 | 0.0948 | 0.0718 | 0.0355 |
| redwood-n100000-q1024-sigma0-k32 | 2.5423 | 0.4881 | 0.4261 | 2.3267 | 0.2345 | 0.1939 |
| redwood-n100000-q1024-sigma0.01-k1 | 0.3329 | 0.2733 | 0.2177 | 0.1155 | 0.0812 | 0.0446 |
| redwood-n100000-q1024-sigma0.01-k32 | 3.4368 | 0.5635 | 0.4911 | 3.2266 | 0.3156 | 0.2719 |
| bunny_surface-n100000-q1024-sigma0-k1 | 0.3069 | 0.2598 | 0.2064 | 0.0920 | 0.0699 | 0.0324 |
| bunny_surface-n100000-q1024-sigma0-k32 | 2.6276 | 0.4835 | 0.4245 | 2.3976 | 0.2399 | 0.1981 |
| bunny_surface-n100000-q1024-sigma0.01-k1 | 0.3272 | 0.2673 | 0.2129 | 0.1082 | 0.0788 | 0.0405 |
| bunny_surface-n100000-q1024-sigma0.01-k32 | 3.2718 | 0.5300 | 0.4698 | 3.0459 | 0.2932 | 0.2516 |
| sphere-n100000-q1024-sigma0-k1 | 0.3013 | 0.2590 | 0.2064 | 0.0849 | 0.0714 | 0.0333 |
| sphere-n100000-q1024-sigma0-k32 | 2.5723 | 0.4781 | 0.4093 | 2.3648 | 0.2275 | 0.1847 |
| sphere-n100000-q1024-sigma0.01-k1 | 0.3267 | 0.2663 | 0.2128 | 0.1086 | 0.0760 | 0.0391 |
| sphere-n100000-q1024-sigma0.01-k32 | 3.1547 | 0.5059 | 0.4437 | 2.9202 | 0.2654 | 0.2224 |
| volume-n100000-q1024-sigma0-k1 | 0.3247 | 0.2681 | 0.2113 | 0.1102 | 0.0788 | 0.0406 |
| volume-n100000-q1024-sigma0-k32 | 3.3969 | 0.5228 | 0.4564 | 3.1893 | 0.2930 | 0.2497 |
| volume-n100000-q1024-sigma0.01-k1 | 0.3237 | 0.2646 | 0.2109 | 0.1076 | 0.0789 | 0.0406 |
| volume-n100000-q1024-sigma0.01-k32 | 3.4365 | 0.5146 | 0.4471 | 3.2232 | 0.2837 | 0.2371 |
| clusters-n100000-q1024-sigma0-k1 | 0.3473 | 0.2831 | 0.2269 | 0.1251 | 0.0830 | 0.0463 |
| clusters-n100000-q1024-sigma0-k32 | 3.6064 | 0.5597 | 0.5016 | 3.3933 | 0.3115 | 0.2716 |
| clusters-n100000-q1024-sigma0.01-k1 | 0.3535 | 0.2858 | 0.2298 | 0.1288 | 0.0910 | 0.0529 |
| clusters-n100000-q1024-sigma0.01-k32 | 3.6530 | 0.5945 | 0.5383 | 3.4085 | 0.3543 | 0.3086 |
| thin_plane-n100000-q1024-sigma0-k1 | 0.2949 | 0.2508 | 0.1969 | 0.0822 | 0.0676 | 0.0302 |
| thin_plane-n100000-q1024-sigma0-k32 | 2.5128 | 0.4636 | 0.3959 | 2.2999 | 0.2189 | 0.1775 |
| thin_plane-n100000-q1024-sigma0.01-k1 | 0.3185 | 0.2615 | 0.2103 | 0.1067 | 0.0769 | 0.0393 |
| thin_plane-n100000-q1024-sigma0.01-k32 | 3.3569 | 0.5190 | 0.4479 | 3.1368 | 0.2772 | 0.2355 |
| sphere-n1000000-q8192-sigma0-k11 | 2.4456 | 1.6090 | 1.5514 | 1.0795 | 0.4498 | 0.3993 |
| sphere-n1000000-q8192-sigma0.01-k11 | 2.8802 | 1.8781 | 1.8185 | 1.5671 | 0.7193 | 0.6748 |
| volume-n1000000-q8192-sigma0-k11 | 2.8339 | 1.8023 | 1.7558 | 1.5530 | 0.6614 | 0.6281 |
| volume-n1000000-q8192-sigma0.01-k11 | 2.8493 | 1.8361 | 1.7736 | 1.5746 | 0.6627 | 0.6261 |
| sphere-n1000-q32-sigma0-k32 | 1.4306 | 0.1823 | 0.1470 | 1.3806 | 0.1385 | 0.1012 |
| sphere-n1000-q32-sigma0.01-k32 | 1.4383 | 0.1797 | 0.1460 | 1.3897 | 0.1365 | 0.0991 |
| volume-n1000-q32-sigma0-k32 | 1.5010 | 0.2177 | 0.1829 | 1.4511 | 0.1761 | 0.1368 |
| volume-n1000-q32-sigma0.01-k32 | 1.5104 | 0.2158 | 0.1792 | 1.4653 | 0.1712 | 0.1345 |
