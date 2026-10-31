# Kokkos backend skeleton

Status: ready-for-agent
Blocked by: 21

## Context

Spec: [Backends](../../core-architecture/spec.md#7-backends), [Build, toolchains and packaging](../../core-architecture/spec.md#14-build-toolchains-and-packaging).
Decisions: [Backend interface](../../core-architecture/issues/10-backend-interface.md), [Feasibility of JIT compilation with Kokkos](../../core-architecture/issues/03-jit-with-kokkos.md).
ADRs: [ADR 0003](../../../docs/adr/0003-runtime-backend-interface-on-lowered-kernels.md).
Builds on: [Memory resources and Umpire pools](21-memory-resources-and-umpire.md).

## What to build

- The `tensorlib::kokkos` component: registration, memory spaces, queues as Kokkos execution-space instances, events, unmanaged Views over resource buffers, capabilities.
- One device per process; default queue per pair of context and device; `tl::on(queue)`.
- Kokkos OpenMP builds added to the CI matrix.

## Acceptance

- Copies and allocation run through Kokkos on the OpenMP backend in CI.
- Core headers still do not include Kokkos (header check).
