# MPI communicator and progress thread

Status: ready-for-agent
Blocked by: 33, 34

## Context

Spec: [Distributions and communication](../../core-architecture/spec.md#9-distributions-and-communication), [Testing and benchmarking](../../core-architecture/spec.md#16-testing-and-benchmarking).
Decisions: [Communication in the evaluation plan](../../core-architecture/issues/28-communication-in-the-evaluation-plan.md), [Test and benchmark infrastructure](../../core-architecture/issues/32-test-and-benchmark-infrastructure.md).
ADRs: [ADR 0010](../../../docs/adr/0010-persistent-communication-patterns-and-a-progress-thread.md).
Builds on: [Communicator interface and threaded test implementation](33-communicator-and-threaded-test-implementation.md), [Ghost exchanges in the plan](34-ghost-exchanges-in-the-plan.md).

## What to build

- The `tensorlib::mpi` implementation: MPI 4 persistent and neighbourhood collectives with a fallback, the progress thread (`TL_COMM_PROGRESS=thread|caller`), GPU-aware MPI detection and pinned host staging.
- Real MPI jobs in CI: MPICH and Open MPI at 1 to 4 ranks per PR, 8 and 16 ranks nightly.

## Acceptance

- The distributed tests pass with both MPI implementations and both progress modes.
