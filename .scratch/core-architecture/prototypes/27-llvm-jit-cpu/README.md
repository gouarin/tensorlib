# PROTOTYPE: LLVM JIT on CPU

Throwaway code for the ticket [Prototype of the LLVM JIT on CPU](../../issues/27-llvm-jit-cpu-prototype.md). Delete it once the decision is recorded.

## What it is

- `jit.hpp` and `jit.cpp` hold a tiny per-element SSA IR, shaped as decided in [Kernel IR](../../issues/22-kernel-ir.md): an iteration space (flat or segmented), loads with stencil shifts, scalar arguments, stores, and several bodies selected by a tag per segment. The IR is lowered to LLVM IR with IRBuilder, optimised with the O3 pipeline and linked with ORC `LLJIT`.
  - FMA contraction is the only fast-math flag.
  - Buffers are `noalias` parameters of an inner function, which is inlined into an exported wrapper taking `double**`.
  - A canonical-block reduction is emitted by hand: blocks of 4096 elements, 8 accumulator lanes in element order, then a fixed tree.
- `kernels.hpp` holds the reference kernels, their hand-written twins compiled by g++, the Kokkos dispatch (`body(begin, end, args)` called from `Kokkos::parallel_for` over chunks of 8192 elements or 256 segments), and the test data.
- `check.cpp` checks the generated kernels against the hand-written ones and prints the bits of the reductions.
- `ceilings.cpp` measures what the machine can do (bandwidth by stream count, FMA peak), to compare the kernels with.
- `compile_times.cpp` measures cold compiles, an object-cache reload and the optimised assembly.
- `bench.cpp` measures run times with Google Benchmark.

## Run

```bash
BUILD=<build dir> ./run.sh
```

It was run with LLVM 21.1.8, Kokkos 5.0.2 (OpenMP), Google Benchmark 1.9.1 and GCC 15.2 at `-O3 -mcpu=apple-m3`, all from Ubuntu 26.04 packages. The machine was an 18-vCPU aarch64 VM on Apple Silicon with no GPU. LLVM detects the CPU as `generic`, with the host's features. Each benchmark has 10 interleaved repetitions, and the tables give medians. Runs used 1 thread and 16 threads. Treat VM numbers as orders of magnitude.

Ubuntu's `libkokkos-dev` ships a `KokkosTargets.cmake` that names a missing `libkokkosalgorithms.a`, so `CMakeLists.txt` links the shared Kokkos libraries directly. `-mcpu=native` fails with GCC 15 on this VM ("no support for sme without sve2"), hence `apple-m3`.

## A measurement mistake caught on the way

The first run made the generated ponio kernel look 1.5 times faster than the hand-written loop. The assembly showed why: GCC did not vectorise the hand-written loop, because its `__restrict` pointers were locals inside the Kokkos lambda. In plain C, with the pointers as function parameters, GCC vectorises the same loop. The hand-written version now calls a `noinline` function with `__restrict` parameters, and the gap is gone. The bias was in the reference, not in the JIT.

## Results (2026-10-30)

### Correctness and reproducibility (`check.txt`)

- ponio20: the JIT and g++ give bit-identical results, since both contract to FMA. Without FMA, 21 % of the elements differ in the last bit.
- The two-body kernel, two separate launches and the hand-written loop give bit-identical results.
- The affine flux turned into a gather matches the atomic scatter, except for 1 element in 10^6, which differs by 1.1e-16 at 4 and 16 threads. The atomics change the order of the additions.
- The canonical sum has the same bits with 1, 4 and 16 threads (`c036218218f1a2ca`). The Kokkos `parallel_reduce` gives different bits at each thread count. Compared with a `long double` sum, the canonical error is 9.8e-13 and the Kokkos error 5e-12 to 9e-12.

### Cold compile cost (`compile-times.txt`, median of 9, fresh `LLVMContext` each time)

| kernel | ops | IR build | O3 | codegen + link | total |
|---|---|---|---|---|---|
| ponio 20 stages | 83 | 0.05 ms | 1.8 ms | 4.6 ms | 6.5 ms |
| segmented stencil | 10 | 0.02 ms | 1.1 ms | 2.8 ms | 4.1 ms |
| two bodies | 13 | 0.04 ms | 1.8 ms | 3.9 ms | 5.6 ms |
| flux as gather | 5 | 0.02 ms | 0.8 ms | 2.5 ms | 3.3 ms |
| canonical sum | - | | | | 4.4 ms |

Growth with kernel size (chain `x = x*s + b`):

| ops | 14 | 50 | 194 | 770 | 3 074 | 12 290 |
|---|---|---|---|---|---|---|
| total | 3.5 ms | 3.5 ms | 4.8 ms | 12 ms | 88 ms | 1 360 ms |

- A kernel costs about 3 ms at minimum, about 70 % of it in codegen and linking.
- The cost is nearly flat up to about 200 operations, then grows faster than the size: about 7 times for 4 times more operations at 3 000 operations, and 15 times at 12 000.
- Building the IR is negligible.

Object cache: building ponio20, optimising it and emitting an object takes 6.4 ms. Reading the object back from disk and linking it with ORC takes 0.056 ms, about 100 times less. The object is 1.4 KB.

### Assembly

- ponio20 has a clean NEON inner loop: 21 `ldr q`, 20 `fmla`, no stack access. The `sp` accesses counted by `compile_times` are in the prologue and around the loop. `noalias` changes nothing in the loop: without it, LLVM adds a runtime alias check before the loop.
- The stencil and the flux are vectorised.
- `live(L)`, with L inputs live across 4 outputs, is vectorised up to L = 24, with growing spills. From L = 32 on, LLVM gives up and emits scalar code.

### Run time (median, us)

ponio 20 stages (22 streams):

| n | threads | JIT | JIT without noalias | hand |
|---|---|---|---|---|
| 65 536 | 1 | 175 | 173 | 173 |
| 1 048 576 | 1 | 3 042 | 3 087 | 3 002 |
| 8 388 608 | 1 | 25 313 | 25 343 | 25 108 |
| 65 536 | 16 | 96 | 106 | 97 |
| 1 048 576 | 16 | 969 | 949 | 899 |
| 8 388 608 | 16 | 6 762 | 6 746 | 6 316 |

Segmented stencil, one or two bodies (segments of 8 to 64 cells, every fifth one a boundary):

| n | threads | stencil only | two bodies, one launch | two launches | hand, two bodies |
|---|---|---|---|---|---|
| 65 536 | 1 | 13.8 | 11.7 | 17.5 | 14.0 |
| 1 048 576 | 1 | 214 | 249 | 299 | 301 |
| 8 388 608 | 1 | 2 209 | 2 819 | 4 027 | 2 835 |
| 65 536 | 16 | 7.0 | 5.9 | 15.2 | 6.3 |
| 1 048 576 | 16 | 30.9 | 46.1 | 56.5 | 46.7 |
| 8 388 608 | 16 | 625 | 545 | 1 034 | 557 |

Affine scatter-add `u(I) -= f; u(I + 1) += f;`:

| n | threads | JIT gather | hand gather | hand atomic scatter |
|---|---|---|---|---|
| 1 048 576 | 1 | 216 | 248 | 1 532 |
| 8 388 608 | 1 | 1 737 | 1 994 | 12 192 |
| 1 048 576 | 16 | 48.9 | 48.7 | 192 |
| 8 388 608 | 16 | 681 | 697 | 1 342 |

Reductions:

| n | threads | canonical sum | Kokkos sum | canonical max | Kokkos max |
|---|---|---|---|---|---|
| 1 048 576 | 1 | 90.7 | 571 | 81.8 | 503 |
| 8 388 608 | 1 | 791 | 4 508 | 771 | 4 001 |
| 1 048 576 | 16 | 13.7 | 57.6 | 13.0 | 47.5 |
| 8 388 608 | 16 | 262 | 427 | 264 | 480 |

Live values, n = 262 144, cost per element and per input stream (ns):

| L | 4 | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|
| 1 thread | 0.11 | 0.17 | 0.25 | 0.56 | 1.35 | 2.90 | 3.43 |
| 16 threads | 0.018 | 0.030 | 0.088 | 0.107 | 0.256 | 0.359 | 0.401 |

### Machine ceilings (`ceilings.cpp`, `ceilings-{1,16}threads.json`)

To tell "as fast as g++" from "as fast as the machine", `ceilings.cpp` measures what the VM can do at all, with plain OpenMP and NEON intrinsics on 8M-element arrays. The CPU has SME2 (Apple M4 family); neither g++ nor the JIT uses it.

Bandwidth with N read streams and 1 write stream (GB/s, every stream counted once; stores have no write-allocate cost here, copy and triad give the same bytes per second):

| streams | 3 | 5 | 9 | 13 | 17 | 22 |
|---|---|---|---|---|---|---|
| 1 thread | 109 | 105 | 91 | 85 | 74 | 58 |
| 16 threads | 272 | 262 | 260 | 253 | 233 | 219 |

One core loses bandwidth as streams are added: 22 streams give half of what 3 give. With 16 threads the memory system is the limit and the stream count matters much less. A read-only sum reaches 80 to 88 GB/s on one core and about 210 GB/s on 16 threads, whatever the number of accumulators (4, 8 or 16 vectors). FMA peak is 53 GFlop/s on one core and 500 GFlop/s on 16 threads (not 16 times more, so the 16 vCPUs are not 16 free performance cores).

The kernels against these ceilings, at 8M elements:

| kernel | ceiling used | 1 thread | 16 threads |
|---|---|---|---|
| ponio20, JIT | 22 streams | 100 % | 100 % (hand: 107 %) |
| flux as gather, JIT | 3 streams | 107 % | 109 % |
| canonical sum | read-only sum | 104 % | about 115 % |
| segmented stencil, JIT | flat stencil | 60 % | 82 % |
| two bodies, one launch | flat stencil | 47 % | 94 % |
| `live(24)` | FMA peak | 27 % | - |

Above 100 % means the kernel beats the ceiling's own loop, by noise or because a shifted reload hits the cache. ponio, the flux and the sum are at the machine's limit. The segmented kernels are not: with segments of 8 to 64 cells, the per-segment cost (reading the row, the tag switch, loop entry, scalar tails) takes 40 % of the time on one core, and the hand-written twin pays it too. `live(24)` is below FMA peak because its 96 weights do not fit in 32 vector registers, so most FMAs wait on a reload; this is the register ceiling at work.

## What it says

1. **The generated code matches the hand-written loop.** On ponio, the gap is within 2 % on one thread and up to 8 % on 16 threads, where the CV is 4 to 7 %. The results are bit-identical. The stencil, the two-body kernel and the flux match too.
2. **`noalias` does not change speed on CPU.** LLVM vectorises without it, behind a cheap runtime check. It stays in the generator because it is free and removes that check.
3. **Cold compiles are cheap and the disk cache works.** A reference kernel costs 3 to 7 ms cold and 0.06 ms from the on-disk object cache. A samurai or ponio run with a few dozen distinct kernels pays a few hundred milliseconds once.
4. **CPU defaults for the ceilings.** For `TL_MAX_KERNEL_OPS`, about 1 000 operations keeps a cold compile under about 30 ms (between the measured 12 ms at 770 and 88 ms at 3 000). For `TL_MAX_KERNEL_LIVE`, 24 lanes on NEON (32 vector registers): the cost per input doubles from 16 to 24 lanes on one thread, and LLVM gives up vectorising at 32. Both are first values for this core, to set per target from capabilities.
5. **The canonical reduction is reproducible and faster.** It has the same bits at every thread count, a smaller error, and is 4 to 6 times faster than a Kokkos `parallel_reduce` on one thread. That reduce is not vectorised, since it has no reassociation.
6. **Turning the affine scatter-add into a gather pays off.** It is 2 to 7 times faster than atomics and deterministic, while atomics already change the last bit at 4 threads.
7. **One launch with two bodies beats two launches.** It runs at the speed of the hand-written loop, and two launches cost 1.2 to 2.6 times more. The gap is largest at 64K, where the second launch adds a few microseconds. At 8M the gap is also 1.9 times, which these runs do not explain (the boundary kernel touches scattered segments); a real node should check it.
8. **Flat kernels reach the machine's ceiling, segmented kernels do not.** ponio, the gather flux and the canonical sum run at the measured bandwidth limit of this VM, so neither the JIT nor g++ can do better there. The segmented stencil runs at 60 % of a flat stencil on one core and 82 % on 16 threads, and the hand-written loop is no faster. The loss is the per-segment cost of short samurai-like intervals, which is a question for how segments are laid out and lowered, not for LLVM.

Not measured here: the GPU paths (NVPTX, AMDGPU), the GPU ceilings and the 32-bit index variant, all in [Rerun the fusion benchmark on an HPC node and a GPU](../../issues/19-rerun-fusion-benchmark-hpc-gpu.md); x86 vector widths; Windows and macOS.
