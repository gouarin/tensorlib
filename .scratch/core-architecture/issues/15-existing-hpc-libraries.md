# Existing HPC libraries that cover part of tensorlib

Type: research
Status: resolved

## Question

Which existing libraries already deliver, with proven performance on distributed HPC machines, some or all of what tensorlib targets: a NumPy-like tensor API, lazy evaluation with fusion across several outputs, Kokkos or equivalent portability (CPU SIMD, threads, NVIDIA/AMD/Intel GPUs), partitioned tensors over MPI, bindings to other languages, and PETSc-like linear algebra? Cover at least Legate and cuPyNumeric (including their C++ API), DaCe, Heat, Arkouda, JAX/XLA with sharding, HPX, Kokkos Kernels, Kokkos Comm, Kokkos FFT, Trilinos (Tpetra), PETSc with its Kokkos backend, Ginkgo, TiledArray, Cyclops Tensor Framework, xtensor, Eigen, Blaze and Fastor. For each: what it covers, primary language and C++ usability, maturity and maintenance, published HPC performance (scaling results, machines), licence, and whether tensorlib could adopt it, extend it, build on it as a dependency, or only learn from it. Finish with the gaps that no existing library fills.

## Answer

No existing library covers all six targets. Legate and cuPyNumeric, the closest match (C++ `NDArray` API, distributed, Diffuse fusion), were archived as end of life with final release v26.06.01, so they can only be studied.
JAX/XLA (multi-output fusion, sharding), DaCe (SDFG fusion, MPI at 512 nodes) and Heat (MPI, eager) already do distributed NumPy well, but from Python. None of them runs on Kokkos or is meant to be used as a C++ library.
Depend on: Kokkos Kernels (local LA), Kokkos-FFT (local FFT), mdspan, and PETSc (Kokkos back end) plus Trilinos/Tpetra for distributed LA, with Ginkgo as an option. Swap "build a PETSc-like layer" for zero-copy interop. Track Kokkos Comm (v0.1, experimental).
Learn only: xtensor (API), XLA, Diffuse and DaCe (fusion), Heat and Tpetra (partition model), Eigen, Blaze, Fastor, CTF. TiledArray is GPL-3, which rules it out.
What nobody fills: a C++ NumPy-like API on Kokkos with lazy multi-output fusion and MPI-partitioned tensors (halos included in the graph) that hands local parts to PETSc/Tpetra without copies, plus Python/Julia/Rust bindings from one core.

Findings: [research/15-existing-hpc-libraries.md](../research/15-existing-hpc-libraries.md)
