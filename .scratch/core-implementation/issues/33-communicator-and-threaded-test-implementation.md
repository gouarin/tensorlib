# Communicator interface and threaded test implementation

Status: ready-for-agent
Blocked by: 01

## Context

Spec: [Distributions and communication](../../core-architecture/spec.md#9-distributions-and-communication), [Testing and benchmarking](../../core-architecture/spec.md#16-testing-and-benchmarking).
Decisions: [Communication in the evaluation plan](../../core-architecture/issues/28-communication-in-the-evaluation-plan.md), [Test and benchmark infrastructure](../../core-architecture/issues/32-test-and-benchmark-infrastructure.md).
ADRs: [ADR 0010](../../../docs/adr/0010-persistent-communication-patterns-and-a-progress-thread.md).
Builds on: [Build skeleton and repository layout](01-build-skeleton.md).

## What to build

- `tl::communicator`: `rank`, `size`, `dup`, `split`, congruence, `halo_pattern`, `reduce_pattern`, `exchange_pattern` with `start` and `complete -> event`, raw point-to-point.
- The `self` implementation.
- A test implementation running every rank as a thread in one process, with fault injection (slow ranks, reordered completions, corrupted headers), usable under ASan, UBSan and TSan.

## Acceptance

- A ring exchange and an allreduce run on 1 to 8 thread ranks under TSan.
- Injected reordering is reproducible from a seed.
