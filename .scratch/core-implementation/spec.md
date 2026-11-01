# Implementation of the tensorlib core

The spec is [the architecture spec](../core-architecture/spec.md), approved on 2026-10-31. This file only explains how it was cut into tickets and lists them.

## How the tickets are cut

- Each ticket fits one agent session and ends with working, tested behaviour. Its tests follow [Testing and benchmarking](../core-architecture/spec.md#16-testing-and-benchmarking), and later tickets extend the window generator with what they add.
- Each ticket links to the spec sections, decision tickets and ADRs it implements, and lists what it leaves to other tickets.
- Each ticket documents the interface it adds in the same branch: Doxygen comments on every public entity and a page of the Sphinx site (`docs/source/`), built without warnings by CI. See `CONTRIBUTING.md`.
- The order follows [Suggested order of work](../core-architecture/spec.md#suggested-order-of-work): the serial backend and the front end first, so that every later backend has an oracle.
- `ready-for-agent` tickets can be taken by an agent once their blockers are resolved. `ready-for-human` tickets need machines, accounts or a maintainer.
- [Rerun the fusion benchmark on an HPC node and a GPU](../core-architecture/issues/19-rerun-fusion-benchmark-hpc-gpu.md) stays on the architecture map. The GPU lowering tickets carry its device checks.

## Tickets

| # | Ticket | Status | Blocked by |
|---|---|---|---|
| 01 | [Build skeleton and repository layout](issues/01-build-skeleton.md) | ready-for-agent | - |
| 02 | [Hosted CI matrix and header checks](issues/02-ci-matrix-and-header-checks.md) | ready-for-agent | 01 |
| 03 | [fixed<T, N...> and element traits](issues/03-fixed-and-element-traits.md) | ready-for-agent | 01 |
| 04 | [Shapes, dtypes and Array API type promotion](issues/04-shapes-dtypes-and-promotion.md) | ready-for-agent | 01 |
| 05 | [Kernel IR for rectangular kernels](issues/05-kernel-ir-rectangular.md) | ready-for-agent | 04 |
| 06 | [Backend interface and serial backend](issues/06-backend-interface-and-serial-backend.md) | ready-for-agent | 05 |
| 07 | [Tensor storage, copy-on-write and the erased tensor](issues/07-tensor-storage-and-copy-on-write.md) | ready-for-agent | 04, 06 |
| 08 | [Contexts, pending windows and element-wise operations](issues/08-contexts-windows-and-elementwise-operations.md) | ready-for-agent | 03, 07 |
| 09 | [Planner: fusing a window into kernels](issues/09-window-fusion-planner.md) | ready-for-agent | 08 |
| 10 | [Broadcasting](issues/10-broadcasting.md) | ready-for-agent | 08 |
| 11 | [Affine views and slicing](issues/11-affine-views-and-slicing.md) | ready-for-agent | 09 |
| 12 | [Aliasing and in-place semantics](issues/12-aliasing-and-in-place-semantics.md) | ready-for-agent | 10, 11 |
| 13 | [Reductions and scans](issues/13-reductions-and-scans.md) | ready-for-agent | 09 |
| 14 | [Window generator and comparison rules](issues/14-window-generator-and-comparison-rules.md) | ready-for-agent | 10, 11, 13 |
| 15 | [Segment spaces and multi-body kernels](issues/15-segment-spaces-and-multi-body-kernels.md) | ready-for-agent | 11 |
| 16 | [Gathers, scatters and masked writes](issues/16-gathers-scatters-and-masks.md) | ready-for-agent | 13, 15 |
| 17 | [Traced functions, traced combiners and external functions](issues/17-traced-functions-and-external-functions.md) | ready-for-agent | 13 |
| 18 | [Kernel splitting and the plan cache](issues/18-kernel-splitting-and-plan-cache.md) | ready-for-agent | 09 |
| 19 | [Exceptions, poison and finalize](issues/19-errors-poison-and-finalize.md) | ready-for-agent | 09 |
| 20 | [Library operations](issues/20-library-operations-registry.md) | ready-for-agent | 09 |
| 21 | [Memory resources and Umpire pools](issues/21-memory-resources-and-umpire.md) | ready-for-agent | 06, 19 |
| 22 | [Kokkos backend skeleton](issues/22-kokkos-backend-skeleton.md) | ready-for-agent | 21 |
| 23 | [LLVM code generator on CPU](issues/23-llvm-cpu-generator.md) | ready-for-agent | 22 |
| 24 | [Kernel cache and kernel bundles](issues/24-kernel-cache-and-bundles.md) | ready-for-agent | 23 |
| 25 | [CPU lowering of reductions, scans and fixed lanes](issues/25-cpu-reductions-scans-and-fixed-lanes.md) | ready-for-agent | 23, 13 |
| 26 | [CPU lowering of segments, gathers and scatters](issues/26-cpu-segments-gathers-and-scatters.md) | ready-for-agent | 23, 15, 16 |
| 27 | [Debug checks](issues/27-debug-checks.md) | ready-for-agent | 19, 23 |
| 28 | [Kokkos interop kernels](issues/28-kokkos-interop-kernels.md) | ready-for-agent | 22, 09 |
| 29 | [Library operations in the Kokkos backend](issues/29-kokkos-library-operations.md) | ready-for-agent | 20, 22 |
| 30 | [Benchmarks and machine ceilings](issues/30-benchmarks-and-machine-ceilings.md) | ready-for-agent | 25, 26 |
| 31 | [Compile-time budget in CI](issues/31-compile-time-budget.md) | ready-for-agent | 02, 08 |
| 32 | [Distributions](issues/32-distributions.md) | ready-for-agent | 10, 11 |
| 33 | [Communicator interface and threaded test implementation](issues/33-communicator-and-threaded-test-implementation.md) | ready-for-agent | 01 |
| 34 | [Ghost exchanges in the plan](issues/34-ghost-exchanges-in-the-plan.md) | ready-for-agent | 32, 33, 19 |
| 35 | [Reductions across ranks](issues/35-reductions-across-ranks.md) | ready-for-agent | 34, 13 |
| 36 | [Redistribution](issues/36-redistribution.md) | ready-for-agent | 34 |
| 37 | [Views and element reads on partitioned tensors](issues/37-partitioned-views-and-element-reads.md) | ready-for-agent | 32, 34 |
| 38 | [MPI communicator and progress thread](issues/38-mpi-communicator-and-progress-thread.md) | ready-for-agent | 33, 34 |
| 39 | [SPMD checks and errors across ranks](issues/39-spmd-checks-and-remote-errors.md) | ready-for-agent | 35, 38 |
| 40 | [PETSc borrow, wrap and DMDA](issues/40-petsc-borrow-wrap-and-dmda.md) | ready-for-agent | 34, 21 |
| 41 | [PETSc operators and stream-ordered synchronisation](issues/41-petsc-operators-and-stream-ordered-sync.md) | ready-for-agent | 40 |
| 42 | [Tpetra interop](issues/42-tpetra-interop.md) | ready-for-agent | 34, 21 |
| 43 | [C ABI core](issues/43-c-abi-core.md) | ready-for-agent | 11, 19, 20 |
| 44 | [C ABI types, distributions and extensions](issues/44-c-abi-types-distributions-and-extensions.md) | ready-for-agent | 43, 17, 32 |
| 45 | [DLPack import and export](issues/45-c-abi-dlpack.md) | ready-for-agent | 43 |
| 46 | [Python test shim and array-api-tests](issues/46-python-test-shim-and-array-api-tests.md) | ready-for-agent | 44, 45 |
| 47 | [Spack and conda-forge packages](issues/47-packaging-spack-and-conda-forge.md) | ready-for-human | 24, 38 |
| 48 | [Self-hosted GPU runners](issues/48-gpu-ci-runners.md) | ready-for-human | - |
| 49 | [Dedicated benchmark machine](issues/49-benchmark-machine.md) | ready-for-human | 30 |
| 50 | [NVIDIA lowering through NVPTX](issues/50-nvptx-lowering.md) | ready-for-agent | 25, 26, 48 |
| 51 | [AMD lowering through AMDGPU](issues/51-amdgpu-lowering.md) | ready-for-agent | 50 |
| 52 | [Intel lowering through SPIR-V](issues/52-spirv-lowering.md) | ready-for-agent | 50 |
