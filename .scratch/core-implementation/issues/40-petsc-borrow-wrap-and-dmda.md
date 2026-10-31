# PETSc borrow, wrap and DMDA

Status: ready-for-agent
Blocked by: 34, 21

## Context

Spec: [Interop with PETSc and Tpetra](../../core-architecture/spec.md#11-interop-with-petsc-and-tpetra).
Decisions: [Zero-copy interop with PETSc and Tpetra](../../core-architecture/issues/17-petsc-tpetra-interop.md).
Builds on: [Ghost exchanges in the plan](34-ghost-exchanges-in-the-plan.md), [Memory resources and Umpire pools](21-memory-resources-and-umpire.md).

## What to build

- The `tensorlib::petsc` component: scoped `borrow`, `borrow_local`, `borrow_packed` and `wrap` for `Vec` and DMDA vectors, with locks, ghost staleness and copy-on-write handling.
- `tl::petsc::dmda(d)` and `tl::petsc::distribution(da)` with the axis mapping; mismatch errors naming axis and rank.
- Release behaviour when PETSc keeps a reference (`PetscContainer`, warning, `std::terminate` with checks on).

## Acceptance

- A DMDA heat solver loop alternates tensorlib kernels and `KSPSolve` with no copy in the local-vector path.
