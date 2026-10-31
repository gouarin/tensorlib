# Library operations in the Kokkos backend

Status: ready-for-agent
Blocked by: 20, 22

## Context

Spec: [Backends](../../core-architecture/spec.md#7-backends).
Decisions: [Backend interface](../../core-architecture/issues/10-backend-interface.md), [Build, extend or adopt](../../core-architecture/issues/16-build-extend-or-adopt.md).
Builds on: [Library operations](20-library-operations-registry.md), [Kokkos backend skeleton](22-kokkos-backend-skeleton.md).

## What to build

- `matmul`, `gemv`, `sort` from Kokkos Kernels and Kokkos, FFT from Kokkos-FFT, workspaces from the memory resource.

## Acceptance

- Results match the serial implementations within the comparison rules.
