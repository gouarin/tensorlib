# Prototype of multi-output fusion

Type: prototype
Status: resolved
Blocked by: 08

## Question

Build a rough prototype of the pending window decided in [When the task graph is evaluated](08-evaluation-trigger.md): `c = a + b; d = a + 2*b` (and the same with `auto t = a + b` as an expression) must run as one loop when a host read triggers evaluation. Measure it against two separate loops and against xtensor on CPU and one GPU, and measure the per-node cost of building the graph.

## Answer

Decided with the user on 2026-10-30. Prototype: [prototypes/09-multi-output-fusion/](../prototypes/09-multi-output-fusion/README.md). It was measured with Google Benchmark (10 interleaved repetitions, medians) on an 18-vCPU aarch64 VM with no GPU, using 16 threads. A first version of the benchmark gave wrong figures; the README explains why, and the numbers below replace them.

- **The pending window model is validated.** `c = a + b; d = a + 2*b`, `auto t = a + b; e = t*t; f = t + 2*b` and `g = a + b; g = g + 1; h = 2*g` each run as one kernel, with correct values. Dead states are never written, and an expression shared by two outputs is computed once.
- **Bandwidth.** With 16 threads and 64M doubles, the window takes 8.92 ms. Two Kokkos loops take 9.83 ms, xtensor with `noalias` 9.80 ms, and the hand-fused loop 7.33 ms. The window beats two loops and xtensor at every size except 1M.
- **Code generation must emit straight-line loops.** The numexpr-style block interpreter costs 1.2 times the hand-fused loop on DRAM-sized data, 1.5 times on cache-sized data (112 against 74 us at 1M), and 2.1 times on one core with data in L2. As written, it does not map to a GPU either. At best it stays as a fallback. This feeds [Code-generation strategy](12-code-generation-strategy.md).
- **Buffer reuse is required.** New output allocations on every evaluation cost 2 times at 64M and 6.6 times at 1M, because the first touch faults in fresh pages. The copy-on-write uniqueness check, which gives the old buffer of a uniquely held handle back for in-place reuse, removes the cost. Allocation becomes its own decision: [Memory pools for HPC tensors](20-memory-pools-research.md), then [Buffer reuse and allocation](21-buffer-reuse-and-allocation.md).
- **Graph cost is small.** Recording costs about 5 ns per node. A full window cycle costs 3.7 us, against 3.2 us for a bare fused launch.
- **Open measurements.** On one core, a hand-fused loop is slower than two loops on this machine (39.3 against 27.5 ms at 64M). Plain C reproduces it with clean NEON code, so it is a quirk of the VM or the core, not of the prototype. The GPU was not measured. Both go to [Rerun the fusion benchmark on an HPC node and a GPU](19-rerun-fusion-benchmark-hpc-gpu.md).
