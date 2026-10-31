# SPMD checks and errors across ranks

Status: ready-for-agent
Blocked by: 35, 38

## Context

Spec: [Distributions and communication](../../core-architecture/spec.md#9-distributions-and-communication), [Errors and debugging](../../core-architecture/spec.md#10-errors-and-debugging).
Decisions: [Error reporting and debug mode](../../core-architecture/issues/24-error-reporting-and-debug-mode.md), [Communication in the evaluation plan](../../core-architecture/issues/28-communication-in-the-evaluation-plan.md).
ADRs: [ADR 0007](../../../docs/adr/0007-errors-poison-results-across-ranks.md).
Builds on: [Reductions across ranks](35-reductions-across-ranks.md), [MPI communicator and progress thread](38-mpi-communicator-and-progress-thread.md).

## What to build

- Plan-hash mismatch raising `tl::spmd_error`; the `collectives` check; `tl::remote_error` on every rank at the same read; timeouts on every multi-rank test.
- SPMD version of the window generator, including deliberately divergent programs.

## Acceptance

- A divergent program raises `tl::spmd_error` instead of exchanging wrong bands, and never hangs.
