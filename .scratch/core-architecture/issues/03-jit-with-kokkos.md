# Feasibility of JIT compilation with Kokkos

Type: research
Status: resolved

## Question

Can a run-time graph be compiled into fused kernels on CPU, CUDA, HIP and SYCL in a library built on Kokkos? Cover NVRTC, hiprtc, SYCL online compilation, LLVM ORC, MLIR, kernel caches, latency of first execution, constraints on supercomputers (no compiler on compute nodes, read-only file systems), and existing projects that combine Kokkos or a similar layer with a JIT.

## Answer

Feasible on CPU, CUDA, HIP and SYCL, but alongside Kokkos, not through it. Kokkos has no run-time code generation, and its headers cannot go through NVRTC or hiprtc, which compile device code only. Kokkos keeps memory, streams and the CPU thread pool (`cuda_stream()`, `hip_stream()`, `sycl_queue()`, unmanaged Views); the JIT emits native kernels and launches them on those streams.
Preferred shape: one internal IR lowered by an in-process LLVM (ORC for CPU, NVPTX to the driver PTX JIT, AMDGPU, SPIR-V to Level Zero), so compute nodes need no compiler. Proteus, Dr.Jit and AdaptiveCpp SSCP work this way. NVRTC/hiprtc source strings are fine for a first prototype. The DPC++ SYCL source compiler is experimental.
A cold kernel costs tens of milliseconds through the driver, and seconds if heavy headers are parsed. So a two-level cache (memory, then disk) with one writer per job, a fallback to memory only, and a warm-up/read-only mode are required. Running Kokkos C++ through the toolchain at run time (PyKokkos, Kokkos.jl) works, but it is too slow for per-expression fusion.

Findings: [research/03-jit-with-kokkos.md](../research/03-jit-with-kokkos.md)
