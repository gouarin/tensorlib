# In-process LLVM is the only code generator, and the Kokkos backend requires it

The Kokkos backend turns every kernel of the internal IR into LLVM IR and compiles it with an LLVM linked into the process: ORC on CPU, NVPTX then the driver's PTX compiler on NVIDIA, AMDGPU on AMD, SPIR-V then Level Zero on Intel. There is one generator for every target, and compute nodes need no compiler. The JIT is not optional in that backend: when compiling at run time is slow (cold start) or forbidden, a precompiled kernel bundle, loaded read-only, takes its place, and `TL_JIT=off` makes a kernel missing from the bundle an error. The serial backend stays the only interpreter and does not depend on LLVM.

We rejected source strings per vendor (CUDA for NVRTC, HIP for hiprtc) because they need three generators, couple tensorlib to each toolkit's version, and still need LLVM for the CPU and SPIR-V for Intel, whose source compiler is experimental. We rejected Clang in the process compiling a generated C source, because it adds Clang to LLVM and a parse to every cold kernel, and generic C hides target-specific features. We rejected building on Proteus as the generator: it specialises existing annotated kernels rather than generating from a graph, does not cover Intel GPUs and would impose its LLVM version. It may still serve for GPU loading and caching. We rejected a precompiled interpreter running while kernels compile in the background, because it gives two performance paths, results that differ in the last bit between tiers, and does not map to GPUs.

## Consequences

- The Kokkos backend depends on a pinned LLVM version, packaged by Spack and conda-forge. The core and the serial backend stay free of it.
- Generated code is lowered by small per-operation lowerings with a table of intrinsics per target. Developers read the tensorlib IR (`TL_DEBUG=dump`) and, on request, the optimised LLVM IR.
- Cold compilation costs milliseconds per kernel on CPU and tens of milliseconds on GPU. The two-level cache, the bundle and the plan cache absorb it in time-stepping codes.
- ORC is weaker on Windows (COFF). If it fails there, the serial backend is the fallback.
- The GPU paths rest on documented production use (Julia, Triton, AdaptiveCpp) and must be measured when a GPU is available ([Rerun the fusion benchmark on an HPC node and a GPU](../../.scratch/core-architecture/issues/19-rerun-fusion-benchmark-hpc-gpu.md)).
