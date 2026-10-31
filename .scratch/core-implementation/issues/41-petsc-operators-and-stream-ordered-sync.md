# PETSc operators and stream-ordered synchronisation

Status: ready-for-agent
Blocked by: 40

## Context

Spec: [Interop with PETSc and Tpetra](../../core-architecture/spec.md#11-interop-with-petsc-and-tpetra).
Decisions: [Zero-copy interop with PETSc and Tpetra](../../core-architecture/issues/17-petsc-tpetra-interop.md).
Builds on: [PETSc borrow, wrap and DMDA](40-petsc-borrow-wrap-and-dmda.md).

## What to build

- `tl::petsc::mult`, `mult_add`, `mult_transpose` as non-fusible window nodes with distribution checks.
- The opt-in `stream_ordered` mode with native events; tests compare it with full synchronisation.

## Acceptance

- `r = b - tl::petsc::mult(A, x)` is one evaluation with fused parts on both sides.
