# Pulmonary Chamfer versus PyTorch cdist

Measured 2026-11-20 on an NVIDIA RTX PRO 6000 Blackwell Server Edition **MIG 1g.24gb, 46 SMs**, with Torch 2.7.1+cu128 and Warp code `7c64938`. These are new measurements on the supplied pulmonary data, not an A40 reproduction or encoder-training benchmark. The exact customer script and chunk configuration were not supplied; the baselines implement the described operations.

**The tuned chunked baseline still wins at 1K points. Warp is substantially faster at 4K and larger, including when both indexes are constructed for every loss.** Reusing fixed-target indexes helps, but must be stated explicitly. Default matrix-based cdist also changes a small number of nearest-neighbor selections; its larger-case gradient comparisons do not pass our strict agreement checks.

## Complete forward + backward timings

All rows use **B=32**, N=M, float32, TF32 disabled, and the sum of the two mean squared nearest-neighbor distances. Only decoded/query coordinates require gradients. Five synchronized wall samples after two warmups; upload and compilation excluded. Ordinary calls, without CUDA graph replay. Warp columns use native Warp Tape, preallocated outputs, and include gradient reset, reductions, reverse-index construction and retirement. The fresh-index column also includes target-index construction; target reuse excludes it.

Times are milliseconds; lower is better. The chunked column is the **best measured chunk size among the tested candidates**, selected on these evaluation inputs, not an independent tuning set. This gives PyTorch the benefit of the measured search. † means the timing is retained but the strict gradient-agreement check failed; these entries are not interchangeable exact-gradient results.

| Points/cloud | Full cdist, default | Best tested chunked cdist + gathers | Selected chunk rows | Warp, both indexes built | Warp, target index reused |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 8.413 | 1.326 | 128 | 3.615 | 1.981 |
| 4,096 | 129.519 † | 20.715 † | 32 | 4.787 | 2.917 |
| 16,384 | Does not fit | 655.223 † | 512 | 12.456 | 8.365 |
| 32,768 | Does not fit | 2,671.677 † | 256 | 18.242 | 13.560 |
| 65,536 | Does not fit | 11,291.360 † | 128 | 27.925 | 23.088 |

Full cdist's output matrix alone would occupy 32/128/512 GiB at 16K/32K/65K respectively, exceeding this GPU's 24 GiB slice. Those cases were not launched; “does not fit” is a proven memory lower bound, not a claimed observed allocation exception. At 1K/4K the output is 128 MiB/2 GiB before backward intermediates.

## Warp search with PyTorch autograd

These routes call the same public Warp kNN API but use returned int64 indices in differentiable PyTorch gathers, squared-distance reductions and backward. Search remains nondifferentiable, as in the chunked baseline. The cuBQL mixed route reuses a cuBQL fixed-target index and builds a Warp reverse index each call. These are explicit backend choices, not automatic dispatch. No custom Torch autograd extension is required, and the core Warp API has no Torch dependency.

| Points/cloud | Warp search + Torch, both indexes built | Warp search + Torch, target reused | cuBQL fixed / Warp reverse + Torch | cuBQL fixed / Warp reverse + Warp Tape |
|---:|---:|---:|---:|---:|
| 1,024 | 3.557 | 1.911 | 1.873 | 1.954 |
| 4,096 | 4.435 | 2.569 | 2.443 | 2.797 |
| 16,384 | 10.968 | 6.906 | 6.281 | 7.758 |
| 32,768 | 15.386 | 10.712 | 9.444 | 12.293 |
| 65,536 | 22.967 | 18.095 | 15.382 | 20.412 |

## What cdist means here

- **Full default cdist:** one `torch.cdist(decoded, targets)` matrix, directional minima, square selected distances, mean, then native Torch backward. PyTorch's [documented default](https://docs.pytorch.org/docs/2.7/generated/torch.cdist.html) uses matrix multiplication at these shapes.
- **Chunked matrix cdist + gathers:** call cdist in matrix mode under `torch.no_grad()`, processing decoded rows in blocks. Each pair is visited once, and both directional minima are accumulated on device. Gather selected coordinate pairs and differentiate their squared distances in Torch. No dense cdist backward graph is built.
- **Forced direct-distance diagnostics:** the same APIs with `compute_mode="donot_use_mm_for_euclid_dist"`. This is distinct from the customer's generic phrase “direct search,” which need not mean this compute mode. Both direct diagnostics pass strict loss/gradient checks at 1K/4K, but are much slower in this pinned Torch/GPU environment. Larger direct diagnostics were not timed; no extrapolated numbers enter the tables.

| Points/cloud | Full direct-distance cdist | Chunked direct-distance cdist + gathers |
|---:|---:|---:|
| 1,024 | 198.494 | 123.371 |
| 4,096 | 3,270.471 | 2,012.180 |

The 16K forced-direct chunked worker was stopped during untimed warmup after the small-case costs were known. Its negative process exit and reason are retained; no median is claimed. Larger forced-direct cases are explicit `not_run` records. Increase `--direct-max-points` to opt into them.

## Chunk-size sensitivity

The primary run used at most 1,024 decoded rows per chunk and capped the pairwise output at 1 GiB. Thus its effective rows are 1,024/1,024/512/256/128 across the five sizes. Additional requested rows were 32, 64, 128, 256, 512 and 2,048 at 1K/4K, and 32/64 at larger sizes. The same 1 GiB cap applies throughout. The row limit can exceed N: at 1K, limits of 1,024 and 2,048 both process one 1,024-row block. Distinct settings that collapse to the same effective chunk size remain recorded. This is a bounded search, not a claim of globally optimal cdist tuning.

| Points/cloud | Chunk-row limit | Median ms | Strict gradient check |
|---:|---:|---:|---|
| 1,024 | 32 | 3.204 | Pass |
| 1,024 | 64 | 1.754 | Pass |
| 1,024 | 128 | 1.326 | Pass |
| 1,024 | 256 | 1.704 | Pass |
| 1,024 | 512 | 2.827 | Pass |
| 1,024 | 1024 | 2.789 | Pass |
| 1,024 | 2048 | 2.791 | Pass |
| 4,096 | 32 | 20.715 | Fail |
| 4,096 | 64 | 28.599 | Fail |
| 4,096 | 128 | 43.989 | Fail |
| 4,096 | 256 | 42.718 | Fail |
| 4,096 | 512 | 41.678 | Fail |
| 4,096 | 1024 | 41.099 | Fail |
| 4,096 | 2048 | 40.772 | Fail |
| 16,384 | 32 | 777.880 | Fail |
| 16,384 | 64 | 721.603 | Fail |
| 16,384 | 512 | 655.223 | Fail |
| 32,768 | 32 | 3,194.632 | Fail |
| 32,768 | 64 | 2,922.213 | Fail |
| 32,768 | 256 | 2,671.677 | Fail |
| 65,536 | 32 | 13,802.148 | Fail |
| 65,536 | 64 | 12,196.845 | Fail |
| 65,536 | 128 | 11,291.360 | Fail |

## Loss and gradient validation

All methods use identical input hashes. The Warp nearest-neighbor reference is checked with up to eight exhaustive float32 searches per cloud and direction. The full scalar loss and every decoded-coordinate gradient are reconstructed independently in NumPy float64 from the selected IDs. Tolerances, fixed before measurement, are loss `rtol=2e-4, atol=1e-7`, gradient relative L2 <=5e-4 and maximum error <=`1e-8 + 5e-4 * max(abs(reference_gradient))`. All Warp variants pass, as do the small forced-direct controls and every 1K matrix-based run.

Default/full cdist at 4K and chunked matrix cdist at 4K and above fail strict gradient agreement even though the scalar losses remain extremely close. For the primary chunked configuration, relative gradient L2 differences are 1.465%, 2.593%, 3.830%, and 6.246% at 4K/16K/32K/65K; the largest relative loss difference is about 0.000626%. This does not establish an error in Torch's backward implementation: selection changes occur before the gather/backward.

A follow-up inspected all differing IDs for the primary matrix chunking at 4K and 65K. There are 22 forward + 11 reverse changes at 4K, and 2,505 + 2,481 at 65K. All changed choices have strictly greater float64 squared distance than the Warp choice, rather than exact-distance ties; maximum excess squared distances are about 1.82e-5 and 4.97e-5 respectively. Up to 16 changed queries per direction were exhaustively checked in float64, confirming the reference minimum and the matrix-mode excess. The matrix arithmetic perturbs near-neighbor rankings, while the direct-distance controls agree. Gradients can change appreciably at a nearest-neighbor switch despite a tiny loss change. See [diagnostics](cdist-neighbor-diagnostics.json) and [reproduction script](diagnose_cdist_neighbors.py). These observations are specific to float32, these inputs, this version and the recorded settings.

## Reproduce and inspect evidence

The archive is not committed. Its MD5 is `ba4fee2fdce2d76e8056594c519851c3`; SHA256 is `8a16347cc942777083a0c763e049a693af452a2c7bbb07ad7e8511a1457889a4`. Nine clouds are cycled across B=32. Targets and decoded inputs are independently subsampled without replacement using `SeedSequence([2348, B, N])`; the two sets can overlap. These hashes match the earlier pulmonary runs, but those runs differentiated both coordinate arrays, so their timings are not mixed into this table.

```bash
uv run --no-sync -m warp.examples.benchmarks.benchmark_point_query_cdist \
  --archive /path/to/pulmonary_point_clouds.zip --batches 32 \
  --points 1024 4096 16384 32768 65536 --repeats 5 --output cdist.json
# Repeat --methods chunked_mm with --chunk-rows 32/64/128/256/512/1024/2048
# for the documented shape subsets, retaining separate output paths.
```

Each method/shape runs in a separate process on a shared Warp/Torch CUDA stream. Error statuses and process exit codes remain explicit. Fixed-target setup, archive reading, correctness checking and host output copies are outside timing. The recorded Torch allocator peak does not include Warp-owned allocations and is not a total GPU memory measurement.

Raw aggregate JSON and logs below retain every sample, status, input/archive/source/native hash and validation metric. `cdist-pulmonary.json` is the original 40-row run; `cdist-fresh-targets.json` adds construction-inclusive variants; `cdist-chunk*.json` records sensitivity. Compressed source snapshots match the recorded benchmark source digests. Between runs only diagnostic limits, extra lifecycle variants and CLI safeguards were added; the original timed operations were retained. No kNN library/native code was changed for this comparison.

To repeat the neighbor diagnostic:

```bash
uv run --no-sync python reports/knn/cdist-batch/diagnose_cdist_neighbors.py \
  --archive /path/to/pulmonary_point_clouds.zip --output neighbors.json
```
