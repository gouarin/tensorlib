# C ABI types, distributions and extensions

Status: ready-for-agent
Blocked by: 43, 17, 32

## Context

Spec: [C ABI and bindings](../../core-architecture/spec.md#13-c-abi-and-bindings).
Decisions: [C ABI and binding surface](../../core-architecture/issues/31-c-abi-and-binding-surface.md).
ADRs: [ADR 0012](../../../docs/adr/0012-one-c-abi-with-explicit-contexts-and-generic-dispatch.md).
Builds on: [C ABI core](43-c-abi-core.md), [Traced functions, traced combiners and external functions](17-traced-functions-and-external-functions.md), [Distributions](32-distributions.md).

## What to build

- Structured `tl_dtype` with components (`fixed`) and opaque user types; memory spaces and layouts. The component rank has no cap, so the extents cannot live in a fixed-size array inside `tl_dtype`.
- `tl_communicator*` from the Fortran MPI handle, `tl_distribution*`, `tl_local_part`.
- `tl_trace` and external-function registration; backend and resource selection by name.

## Acceptance

- A traced combiner from a C callback reduces correctly.
- A communicator from `MPI_Comm_c2f` works across MPI implementations of the test matrix.
