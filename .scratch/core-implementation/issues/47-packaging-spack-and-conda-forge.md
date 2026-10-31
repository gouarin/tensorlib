# Spack and conda-forge packages

Status: ready-for-human
Blocked by: 24, 38

## Context

Spec: [Build, toolchains and packaging](../../core-architecture/spec.md#14-build-toolchains-and-packaging).
Decisions: [Minimum C++ standard and build system](../../core-architecture/issues/14-cxx-standard-and-build.md), [Zero-copy interop with PETSc and Tpetra](../../core-architecture/issues/17-petsc-tpetra-interop.md).
Builds on: [Kernel cache and kernel bundles](24-kernel-cache-and-bundles.md), [MPI communicator and progress thread](38-mpi-communicator-and-progress-thread.md).

## What to build

- A Spack recipe with variants `+cuda`, `+rocm`, `+sycl`, `+mpi`, `+umpire`, `+petsc`, `+trilinos`, and a conda-forge feedstock (CPU and CUDA), both resolving one Kokkos installation shared with PETSc and Trilinos.
- CI jobs that build from the recipes.

## Acceptance

- `spack install tensorlib+mpi` and the conda package build and pass an install test.

## Not in this ticket

- The conda-forge submission itself needs a maintainer account: this ticket is for a human.
