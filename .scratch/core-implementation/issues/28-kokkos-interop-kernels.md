# Kokkos interop kernels

Status: ready-for-agent
Blocked by: 22, 09

## Context

Spec: [Backends](../../core-architecture/spec.md#7-backends), [Evaluation](../../core-architecture/spec.md#5-evaluation).
Decisions: [Backend interface](../../core-architecture/issues/10-backend-interface.md), [Reconcile the core with samurai and ponio](../../core-architecture/issues/26-reconcile-with-samurai-ponio.md).
ADRs: [ADR 0003](../../../docs/adr/0003-runtime-backend-interface-on-lowered-kernels.md).
Builds on: [Kokkos backend skeleton](22-kokkos-backend-skeleton.md), [Planner: fusing a window into kernels](09-window-fusion-planner.md).

## What to build

- `<tl/kokkos.hpp>`: a tensor's local part as a Kokkos View with the queue to use; the user kernel declares its reads and writes and joins the window as a non-fusible node.

## Acceptance

- A user `parallel_for` between two fused kernels does not flush unrelated nodes and does not block the host.
