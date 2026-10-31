# Ghost exchanges in the plan

Status: ready-for-agent
Blocked by: 32, 33, 19

## Context

Spec: [Distributions and communication](../../core-architecture/spec.md#9-distributions-and-communication).
Decisions: [Communication in the evaluation plan](../../core-architecture/issues/28-communication-in-the-evaluation-plan.md), [Distribution kinds](../../core-architecture/issues/18-distribution-kinds.md).
ADRs: [ADR 0004](../../../docs/adr/0004-automatic-ghost-updates-in-the-graph.md), [ADR 0010](../../../docs/adr/0010-persistent-communication-patterns-and-a-progress-thread.md).
Builds on: [Distributions](32-distributions.md), [Communicator interface and threaded test implementation](33-communicator-and-threaded-test-implementation.md), [Exceptions, poison and finalize](19-errors-poison-and-finalize.md).

## What to build

- Stale-ghost tracking and exchange points (send after an in-window writer, before the reader otherwise); `u.update_ghosts()`.
- Generated pack and unpack kernels; one message per neighbour per exchange point in one phase; a self-neighbour becomes a copy; the header with the poison flag and the plan hash.
- Interior and boundary split above `TL_OVERLAP_MIN_CELLS` (`TL_OVERLAP`).
- A progress driver in caller mode (the thread mode comes with MPI).

## Acceptance

- A 2-D heat stencil on 1, 2, 3 and 4 thread ranks matches the single-rank result bit for bit under `TL_FP=strict`, with periodic and non-periodic axes, box and star stencils, and empty local parts.
- Poison on one rank reaches its neighbours through ghost headers.
