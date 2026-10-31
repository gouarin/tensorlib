# PROTOTYPE: lowering of short segments

Throwaway code for the ticket [Lowering of short segments](../../issues/30-lowering-of-short-segments.md). Delete it once the decision is recorded.

## What it is

- `data/` holds the cell intervals of two adapted samurai meshes, dumped with a 12-line helper added to samurai `9e77d1a`. Each line is `level y start end index`, and the storage offset is `index + start`.
  - `advection_2d_level13.txt`: `advection_2d` run with `--max-level 13 --nfiles 4`, last output (t = 0.1). It has 67 815 intervals, 782 374 cells and levels 4 to 13.
  - `rayleigh_taylor_level9.txt`: `lbm-new-D2Q5444-euler-rayleigh-taylor` run with `--max-level 9 --nfiles 4`, last output (t = 2). It has 5 663 intervals, 121 336 cells and levels 7 to 9.
- `jit.hpp` and `jit.cpp` hold the mini IR of prototype 27, cut down to one body. They lower a segment space in four ways (flat, per-cell index list, array-of-structs table, struct-of-arrays table with one base shared by every operand) and with four tails (LLVM's own loop, an overlapping last vector, `llvm.masked` loads and stores, a padded last vector).
- `plan.hpp` holds the plan-time side: the mesh loader (it tiles the mesh to reach 8M cells), the table transforms (bridging and sorting by length), the chunking (256 segments per chunk, one segment per chunk, or about the same number of cells per chunk) and the list of variants.
- `check.cpp` compares every variant with prototype 27's lowering, cell by cell, and prints the shape of each table (`check.txt`).
- `bench.cpp` measures every variant with Google Benchmark (`results-{1,16}threads.json`).

The kernel is the stencil of prototype 27, `out = u - c*((u(+1) - 2u) + u(-1))`, applied on every interval. It reads one field and writes one field, and both share samurai's layout.

## Run

```bash
BUILD=<build dir> ./run.sh
```

It used the same toolchain and VM as prototype 27: LLVM 21.1.8, Kokkos 5.0.2 (OpenMP), Google Benchmark 1.9.1 and GCC 15.2 at `-O3 -mcpu=apple-m3`, on an 18-vCPU aarch64 VM with NEON (no SVE). Each benchmark ran 10 interleaved repetitions, and the tables give medians. Treat VM numbers as orders of magnitude.

To rebuild the meshes, apply the `tl_dump` helper to samurai's `advection_2d.cpp` and `new_D2Q5444_euler_rayleigh_taylor.cpp` (one call after each save). samurai was built from apt packages (`libxtensor-dev`, `libhdf5-dev`, `libfmt-dev`, `libpugixml-dev`, `libcli11-dev`) and HighFive 3.1.1 from source.

## What the real meshes look like

| | advection_2d, level 13 | Rayleigh-Taylor, level 9 |
|---|---|---|
| intervals | 67 815 | 5 663 |
| mean / median length | 11.5 / 4 | 21.4 / 4 |
| p10 / p90 length | 1 / 28 | 1 / 44 |
| cells in intervals shorter than 8 | 17 % | 7 % |
| storage span / cells | 1.53 | 1.47 |
| intervals contiguous with the next one | 0 | 0 |
| after bridging gaps of up to 4 cells | 16 182 intervals, mean 55, +14.5 % cells | 1 833 intervals, mean 72, +8.7 % cells |

- Real segments are much shorter than the synthetic 8 to 64 cells of prototype 27. Half of them have 4 cells or fewer.
- No interval ends where the next one starts in storage. samurai stores the ghost cells between the intervals (gaps of 1 to 4 cells most of the time), so "merging adjacent segments whose offsets continue one another" never applies as stated. What does apply is **bridging**: merge two intervals whose storage gap is at most 4 cells, and compute the gap cells too.
- A third of the storage span is ghost cells. A kernel that walks all the intervals reads that span, so the reachable ceiling is a flat loop over the span (`flat_span`). A flat loop over as many cells as the mesh has (`flat_ceiling`) is 1.5 times faster and cannot be reached with this layout.

## Results (2026-10-31)

All 30 variant and mesh pairs give the same bits as prototype 27's lowering on every cell (`check.txt`).

Percent of `flat_span` (higher is better). 8M is the mesh tiled to about 8 million cells, so it lives in DRAM:

| variant | adv. 1 thr. native | adv. 1 thr. 8M | adv. 16 thr. 8M | RT 1 thr. native | RT 1 thr. 8M | RT 16 thr. 8M |
|---|---|---|---|---|---|---|
| `aos_scalar` (prototype 27) | 48 | 45 | 68 | 94 | 74 | 85 |
| `soa_scalar` (shared base) | 50 | 49 | 71 | 96 | 76 | 89 |
| `soa_overlap4` | 44 | 44 | 70 | 91 | 77 | 84 |
| `soa_pad4` | 63 | 58 | 79 | 96 | 75 | 87 |
| `soa_masked4` | 16 | 16 | 51 | 23 | 21 | 64 |
| `sorted_overlap4` | 40 | 14 | 14 | 76 | 27 | 29 |
| `overlap4_oneseg` | 30 | 31 | 65 | 80 | 69 | 83 |
| `cell_list` | 56 | 54 | 64 | 67 | 56 | 67 |
| `bridged_scalar` | 77 | 72 | 78 | 100 | 78 | 88 |
| `bridged_overlap4_cells` | 78 | 76 | 79 | 96 | 77 | 85 |

At 16 threads on the native meshes (8 to 90 us per launch), the CV reaches 10 to 25 %, so those numbers only show one thing: a bridged table with 256 segments per chunk leaves the 1 833 Rayleigh-Taylor segments in 8 chunks for 16 threads and loses half its speed (42 to 52 % of the span). The same table chunked by cells is back in the noise of the others.

## What it says

1. **Bridging is the lowering that pays.** On the advection mesh, where segments are short, it brings one core from 49 % to 72-78 % of the span ceiling, even though it computes 14.5 % more cells. On 16 threads in DRAM it goes from 71 % to 78-81 %. On Rayleigh-Taylor, whose cells sit mostly in long intervals, it changes nothing, and it costs nothing either.
2. **The tail does not matter once segments are bridged.** LLVM's own loop (vector body plus scalar remainder), the overlapping last vector and the padded last vector are within 5 % of one another. Without bridging, padding to 4 helps (49 to 58-63 %), but it writes past the end of 13 % of the segments into the next segment's cells, which is only correct when one body covers every segment. The overlapping vector does not help. On NEON, `llvm.masked` loads and stores are split into scalar branches and run 3 to 4 times slower on one core. Masks are worth trying only on SVE and AVX-512.
3. **A struct-of-arrays table with one base for all operands is a bit faster than one row per segment** (2 to 5 %). It is also smaller: two integers per segment instead of 1 + n.
4. **Chunk by cells, never by segment count, and never one segment per chunk.** One segment per Kokkos iteration is up to 40 % slower on one core. A fixed count per chunk starves threads once bridging has cut the number of segments.
5. **Do not reorder segments.** Sorting by length breaks storage order and loses 2 to 5 times as soon as the data leaves the cache.
6. **A per-cell index list is not an alternative.** It reaches 54-67 % of the span on one core and does not beat bridging anywhere.
7. **What is left is samurai's layout.** A bridged kernel runs at 72-80 % of a flat loop over the span, and the span is 1.5 times the cells, so a bridged kernel reaches only about 50 % of a flat loop over as many cells. Closing that gap would mean storing ghosts away from the cells, which is a samurai change, not a lowering.

Not measured here: a real 2D stencil that reads the rows above and below through other intervals (the x-direction stencil only reads inside the row); several bodies selected per segment (bridging must then stop at a change of tag); x86 vector widths and masks; GPUs. The GPU belongs in [Rerun the fusion benchmark on an HPC node and a GPU](../../issues/19-rerun-fusion-benchmark-hpc-gpu.md).
