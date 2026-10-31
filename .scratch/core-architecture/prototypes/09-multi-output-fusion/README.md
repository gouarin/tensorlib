# PROTOTYPE: multi-output fusion through a pending window

Throwaway code for the ticket [Prototype of multi-output fusion](../../issues/09-multi-output-fusion-prototype.md). Delete it once the decision is recorded.

## What it is

`tl_window.hpp` is a minimal pending window, about 300 lines, as described in [ADR 0002](../../../../docs/adr/0002-implicit-pending-windows.md):

- `tl::tensor` and `tl::expr` record nodes in a `thread_local` window;
- a host read (`c(5)`) or `tl::evaluate()` evaluates every live output together;
- dead nodes are dropped, and the rest is lowered to a linear program with scratch slots;
- one Kokkos `parallel_for` runs the program as a block interpreter (numexpr style, 512 elements per block);
- a handle that nobody else shares hands its old buffer back for in-place reuse, which is the copy-on-write uniqueness check.

There is no JIT. The `fused_by_hand` variant is the loop a JIT would emit.

- `check.cpp` checks the three fusion cases and counts kernels.
- `bench.cpp` compares the variants with Google Benchmark.

## Run

```bash
DEPS=<dir with install/{kokkos,benchmark} and src/{xtensor,xtl,xsimd}> ./run.sh
```

Built with Kokkos 5.0.0 (OpenMP and Serial), Google Benchmark 1.9.4, xtensor 0.27 with xsimd and `XTENSOR_USE_OPENMP`, and GCC 15.2 at `-O3`. The machine was an 18-vCPU aarch64 VM on Apple Silicon with no GPU.

`run.sh` runs 10 repetitions with random interleaving and reports the median and the coefficient of variation. It uses 1 thread, then all vCPUs but two. The raw data is in `results-1threads.json` and `results-16threads.json`, and the console output in `results.txt`.

## Measurement mistakes in the first version

The first version gave wrong numbers, for two reasons:

- It measured each variant once, as a best-of-N, one variant after the other. On this VM, timings at n = 1M vary 2 to 5 times from one run to the next, so the variants were compared in different machine states. That produced a false "interpreter 3.6 times slower" figure.
- It used all 18 vCPUs. When one vCPU is preempted by the host, the OpenMP barrier waits for it. A one-element kernel launch then took 7 to 10 ms with 18 threads, against 3 us with 16.

Google Benchmark with random interleaving fixes the first problem, and leaving two vCPUs free fixes the second.

## Results (2026-10-30)

`c = a + b; d = a + 2*b` on `double`. Median times in us, with a CV of 2 to 13 % except for the fresh-allocation variant.

16 threads:

| n | two loops | fused by hand | window, fresh alloc | window, buffer reuse | xtensor noalias |
|---|---|---|---|---|---|
| 4 096 | 6.76 | 3.61 | 4.75 | 4.53 | 6.41 |
| 65 536 | 10.6 | 7.16 | 11.1 | 7.52 | 10.7 |
| 1 048 576 | 83.3 | 74.3 | 744 | 112 | 122 |
| 16 777 216 | 2 659 | 2 025 | 4 908 | 2 293 | 2 703 |
| 67 108 864 | 9 827 | 7 332 | 18 125 | 8 920 | 9 800 |

1 thread:

| n | two loops | fused by hand | window, buffer reuse | xtensor noalias |
|---|---|---|---|---|
| 65 536 | 21.4 | 18.3 | 38.6 | 46.0 |
| 1 048 576 | 428 | 614 | 620 | 683 |
| 67 108 864 | 27 508 | 39 293 | 38 026 | 44 803 |

Overhead with 16 threads: recording costs 4.8 ns per node, a bare fused launch plus fence 3.2 us, and a full window cycle (record 2 statements, evaluate, wait) 3.7 us.

## What it shows

- All three cases run as one kernel and give correct values:
  - `c = a + b; d = a + 2*b`;
  - `auto t = a + b; e = t*t; f = t + 2*b`, where `t` is computed once;
  - `g = a + b; g = g + 1; h = 2*g`, where the first `g` is never written.
- With 16 threads, the window with buffer reuse beats two loops and xtensor at every size except 1M. On DRAM-sized data it is 9 % faster than two loops (8 920 against 9 827 us).
- The block interpreter costs 1.2 times the hand-fused loop on DRAM-sized data (8 920 against 7 332 us) and 1.5 times on cache-sized data (112 against 74 us at 1M). On one core with the data in L2 it costs 2.1 times (38.6 against 18.3 us at 64K). It writes each intermediate to a scratch buffer and walks the block once per instruction. As written, it does not map to a GPU, because its scratch buffers are per thread.
- Allocating new outputs on every evaluation costs 2 times at 64M and 6.6 times at 1M, because the first touch faults in fresh pages. Buffer reuse through the uniqueness check removes that cost.
- Graph overhead is small. A full window cycle costs about 0.5 us more than a bare launch.
- On one core, at 1M and 64M, the hand-fused loop is slower than two loops (39.3 against 27.5 ms at 64M). This is not a bug in the prototype. Ten lines of plain C reproduce it without Kokkos, and GCC's assembly is a clean NEON loop (two `ldr q`, `fadd`, `fmla`, two `str q`). Only this mix of two reads and two writes is slow. On the same core, one read with two writes and four reads with one write both reach about 120 GB/s, and changing alignment or array offsets does not help. It is a quirk of this machine, either the VM or the Apple core, and says nothing about fusion in general.
- This VM is not a valid platform for fine ratios. The orders of magnitude hold, but the exact figures need a real HPC node.
