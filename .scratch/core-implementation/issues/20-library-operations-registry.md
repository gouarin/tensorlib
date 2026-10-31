# Library operations

Status: ready-for-agent
Blocked by: 09

## Context

Spec: [Backends](../../core-architecture/spec.md#7-backends).
Decisions: [Backend interface](../../core-architecture/issues/10-backend-interface.md).
ADRs: [ADR 0003](../../../docs/adr/0003-runtime-backend-interface-on-lowered-kernels.md).
Builds on: [Planner: fusing a window into kernels](09-window-fusion-planner.md).

## What to build

- The table of library operations per backend (operation, element type, memory space) and non-fusible plan nodes that evaluate their inputs first.
- Serial implementations of `matmul`, `tensordot`, `sort`, `argsort` and an FFT, with optional BLAS and FFTW.
- A missing operation is an error at node build naming the backend and the type; no silent fallback, no hidden host copy.

## Acceptance

- `r = b - tl::matmul(A, x)` fuses the subtraction around the library node.
- An operation missing for a type throws `tl::unsupported_operation` at the line that recorded it.
