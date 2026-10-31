# PROTOTYPE: compile times of a graph-recording front-end against expression templates

Throwaway code for the ticket [Prototype of compile times for lighter templates](../../issues/11-compile-time-prototype.md). Delete it once the decision is recorded.

## What it compares

`gen.py` generates the same random expressions in four variants. Statement `i` is `auto t_i = E_i(x, y, t_{i-1})`, a tree of depth 4 built from `+ - * /`, scalars, `sin`, `cos`, `tanh`, `exp` and `sqrt`. `x` has shape (N, 1) and `y` shape (1, M), so every statement broadcasts. Every fourth statement and the last one are stored into an output tensor. Each statement depends on the previous one, which is the case where xtensor's `xfunction` types keep growing.

| variant | what the user TU contains |
|---|---|
| `tl` | `tl_front.hpp`: about 70 lines, `<cstddef>` only. Operators take `tl::expr` and record nodes through non-template functions compiled once in `tl_lib.cpp`. |
| `et` | `et.hpp`: minimal C++20 expression templates with concepts and `<cmath>`, evaluated on assignment. |
| `xt_eval` | xtensor 0.27 (xsimd, OpenMP), every `t_i` stored in an `xt::xtensor<double, 2>`. |
| `xt_lazy` | xtensor 0.27, every `t_i` kept as an `auto` `xfunction`. |

`tl_lib.cpp` stands in for the backend: a pending window plus the block interpreter from the multi-output fusion prototype, extended to 2-D broadcasting. There is no JIT.

## Run

```bash
DEPS=<dir with install/benchmark and src/{xtensor,xtl,xsimd}> ./run.sh
```

`measure.py` compiles each TU three times with `g++ -O3 -std=c++20 -fopenmp -c`, one compile at a time, with the variants interleaved. It reports the median wall time, the compiler's peak memory and the object size. `bench.cpp` checks that the four variants agree, then times K = 16 with Google Benchmark. GCC 15.2 on an 18-vCPU aarch64 VM.

## Results (2026-10-30)

Compile time of one TU, median of 3 runs (peak compiler memory in brackets):

| K | tl | et | xt_eval | xt_lazy |
|---|---|---|---|---|
| 1 | 0.01 s (32 MB) | 0.07 s (68 MB) | 1.11 s (429 MB) | 1.06 s (427 MB) |
| 4 | 0.02 s | 0.12 s | 3.00 s | 3.00 s |
| 16 | 0.09 s (47 MB) | 0.45 s (111 MB) | 10.5 s (1.5 GB) | 18.1 s (2.3 GB) |
| 32 | 0.19 s | 0.96 s | 19.4 s | 48.7 s (3.9 GB) |
| 64 | 0.44 s (81 MB) | 1.12 s (182 MB) | 37.2 s (4.2 GB) | 126.9 s (7.4 GB) |

The spread between runs is under 2 %. Full data: `compile-times.json`.

Run time for K = 16, in ms (`run-times.txt`):

| variant | n=256, 16 threads | n=2048, 16 threads | n=256, 1 thread | n=2048, 1 thread |
|---|---|---|---|---|
| tl (interpreter) | 1.43 | 81 | 13.8 | 879 |
| et | 6.18 | 341 | 58.9 | 3 744 |
| xt_eval | 16.8 | 1 020 | 15.9 | 1 050 |
| xt_lazy | 67.7 | 4 318 | 67.0 | - |

All four variants agree to 4.4e-16.

## What it shows

- Recording a graph takes the expression out of the type system. A user TU compiles in 0.01 to 0.44 s, linearly in the number of statements (about 7 ms each), with under 100 MB of memory. xtensor costs 1.1 s for a single statement, because of its headers. After that it costs 0.6 s per statement when intermediates are stored, and grows superlinearly when they stay lazy: 127 s and 7.4 GB at K = 64.
- Minimal expression templates cost 5 to 7 times the recording front-end. They stay 25 to 100 times cheaper than xtensor, so most of xtensor's cost comes from its machinery, not from expression templates as such. They still fuse only within one assignment, and when the chain stays lazy they recompute it for every output.
- At run time, on one thread, the recording front-end with its interpreter is 16 % faster than xtensor with stored intermediates (879 against 1 050 ms). It is 4 times faster than the expression templates, which recompute the lazy chain for each output and broadcast through branches at every access. With 16 threads the gap grows to 12 times, because xtensor does not parallelise strided (broadcasting) assignments unless `strided_parallel_assign` is defined.
- What this does not measure: the cost of a JIT compiling a new kernel at run time (tens of ms cold, then cached, per [Feasibility of JIT compilation with Kokkos](../../issues/03-jit-with-kokkos.md)); Clang, MSVC and nvcc; views and reductions.
