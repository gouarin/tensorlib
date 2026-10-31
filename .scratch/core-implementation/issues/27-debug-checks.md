# Debug checks

Status: ready-for-agent
Blocked by: 19, 23

## Context

Spec: [Errors and debugging](../../core-architecture/spec.md#10-errors-and-debugging).
Decisions: [Error reporting and debug mode](../../core-architecture/issues/24-error-reporting-and-debug-mode.md).
ADRs: [ADR 0007](../../../docs/adr/0007-errors-poison-results-across-ranks.md).
Builds on: [Exceptions, poison and finalize](19-errors-poison-and-finalize.md), [LLVM code generator on CPU](23-llvm-cpu-generator.md).

## What to build

- `TL_DEBUG` and collective `tl::debug_scope` per context; the checks `sync`, `bounds`, `nan` (from finite inputs), `uninit`, `scatter`, `leaks`, `detaches`, `reads`, inserted by lowering and keeping fusion except `sync`.
- Reports naming the operation, element, sites and rank; aggregated diagnostics at `tl::finalize` and `tl::diagnostics()`.

## Acceptance

- Each check has a test that triggers it and one that does not (NaN propagated from inputs is not reported).
- The whole suite runs under `TL_DEBUG=all` in CI.
