# Dedicated benchmark machine

Status: ready-for-human
Blocked by: 30

## Context

Spec: [Testing and benchmarking](../../core-architecture/spec.md#16-testing-and-benchmarking).
Decisions: [Test and benchmark infrastructure](../../core-architecture/issues/32-test-and-benchmark-infrastructure.md).
Builds on: [Benchmarks and machine ceilings](30-benchmarks-and-machine-ceilings.md).

## What to build

- A physical, reserved machine (the GPU runner when idle is acceptable) that runs the benchmark suite nightly, archives results (data branch or a service such as Bencher), alerts on significant regressions above 5 % (Mann-Whitney on interleaved repetitions), and runs the A/B script on PRs labelled `bench`.

## Acceptance

- A deliberate 10 % slowdown on a branch raises an alert.
