# Prototype of the LLVM JIT on CPU

Type: prototype
Status: resolved
Blocked by: 12

## Question

[Code-generation strategy](12-code-generation-strategy.md) chose one generator from the tensorlib IR to LLVM IR, compiled in-process with ORC on CPU ([ADR 0008](../../../docs/adr/0008-in-process-llvm-as-the-only-code-generator.md)). Before the spec is split, measure on CPU, without a GPU, what that choice costs and gives:

- cold compile time (IR build, optimisation, ORC link) of two reference kernels: ponio's 20-stage combination `y + dt*(b1*k1 + ... + b20*k20)` with run-time coefficients, and a segmented samurai stencil (`u(i-1)`, `u(i)`, `u(i+1)` over a table of intervals), plus how it grows with kernel size;
- run time of the generated kernels called through `body(begin, end, args)` from `Kokkos::parallel_for`, against the hand-fused loop of [Prototype of multi-output fusion](09-multi-output-fusion-prototype.md), with and without `noalias` metadata, and whether LLVM vectorises them (check the assembly);
- a canonical-block reduction (`tl::sum`, `tl::max`) against a free-order one, and its bit-reproducibility across thread counts;
- first values for the CPU ceilings `TL_MAX_KERNEL_OPS` and `TL_MAX_KERNEL_LIVE`;
- the cost of loading a kernel back from an on-disk object cache;
- from [Kernel IR](22-kernel-ir.md): an affine scatter-add (`u(I) -= f; u(I + 1) += f;`) turned into a gather against the scatter form, and a two-body kernel (interior and boundary segments) against two launches.

Install LLVM in the sandbox (conda-forge or apt), and measure with Google Benchmark and interleaved repetitions, using at most nproc-2 threads on the VM.

## Answer

The prototype ([README](../prototypes/27-llvm-jit-cpu/README.md)) confirms [Code-generation strategy](12-code-generation-strategy.md) and [Kernel IR](22-kernel-ir.md) on CPU and changes neither. Measured on the 18-vCPU aarch64 VM, LLVM 21, at 1 and 16 threads; read the numbers as orders of magnitude.

- **Generated code matches hand-written code and the machine.** ponio20, the affine flux turned into a gather and the canonical sum run at the bandwidth ceiling measured on the same VM (22 streams for ponio, 3 for the flux, read-only for the sum), on 1 and 16 threads. ponio is bit-identical to g++, both contracting to FMA.
- **`noalias` does not change CPU speed.** LLVM vectorises without it behind a runtime alias check. The generator keeps it because it is free.
- **Compile cost.** A reference kernel costs 3 to 7 ms cold, about 70 % of it in codegen and linking, and 0.06 ms from the on-disk object cache. Cost is flat up to about 200 operations, then superlinear (12 ms at 770, 88 ms at 3 000, 1.4 s at 12 000).
- **CPU defaults for the ceilings:** `TL_MAX_KERNEL_OPS` about 1 000 operations (cold compile under about 30 ms) and `TL_MAX_KERNEL_LIVE` 24 lanes on NEON (the cost per input doubles from 16 to 24, LLVM stops vectorising at 32). They are first values for this core; each target sets its own from capabilities.
- **Canonical reductions** give the same bits at 1, 4 and 16 threads, a smaller error than a free-order sum, and are 4 to 6 times faster than Kokkos `parallel_reduce` on one thread.
- **Affine scatter-add as a gather** is 2 to 7 times faster than atomics and deterministic; atomics already change the last bit at 4 threads.
- **One launch with two bodies** matches the hand-written loop; two launches cost 1.2 to 2.6 times more. The 1.9x gap at 8M elements is not explained and goes to [Rerun the fusion benchmark on an HPC node and a GPU](19-rerun-fusion-benchmark-hpc-gpu.md).
- **Segmented kernels are below the machine's ceiling.** The segmented stencil runs at 60 % of a flat stencil on one core and 82 % on 16 threads, with segments of 8 to 64 cells; the hand-written twin is no faster. The loss is the per-segment cost (row read, tag switch, loop entry, scalar tails), a question for how segments are laid out and lowered, added to the map's fog.

Not measured: GPU paths and ceilings, the 32-bit index variant, x86 vector widths, Windows and macOS. The prototype stays in `.scratch` as the primary source (the repository has no git history yet).

### Review log

- The user asked whether the performance is optimal for this architecture; the machine ceilings (`ceilings.cpp`) were measured to answer, giving the eighth conclusion of the README.
- The user validated the verdict, the two CPU ceiling defaults, and the fog entry on segment lowering.
