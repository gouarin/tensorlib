# Tpetra interop

Status: ready-for-agent
Blocked by: 34, 21

## Context

Spec: [Interop with PETSc and Tpetra](../../core-architecture/spec.md#11-interop-with-petsc-and-tpetra).
Decisions: [Zero-copy interop with PETSc and Tpetra](../../core-architecture/issues/17-petsc-tpetra-interop.md).
Builds on: [Ghost exchanges in the plan](34-ghost-exchanges-in-the-plan.md), [Memory resources and Umpire pools](21-memory-resources-and-umpire.md).

## What to build

- The `tensorlib::tpetra` component: `borrow`, `borrow_packed` and `wrap` for `MultiVector` and `Vector`, `DualView` built from unmanaged views with a lazily allocated host side, contiguous maps, `tl::tpetra::apply`.

## Acceptance

- A Tpetra CG loop with tensorlib updates runs without copies for column-major tensors.
