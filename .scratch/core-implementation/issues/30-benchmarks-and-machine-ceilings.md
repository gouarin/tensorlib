# Benchmarks and machine ceilings

Status: ready-for-agent
Blocked by: 25, 26

## Context

Spec: [Testing and benchmarking](../../core-architecture/spec.md#16-testing-and-benchmarking).
Decisions: [Test and benchmark infrastructure](../../core-architecture/issues/32-test-and-benchmark-infrastructure.md), [Requirements from samurai and ponio](../../core-architecture/issues/25-samurai-ponio-requirements.md).
Builds on: [CPU lowering of reductions, scans and fixed lanes](25-cpu-reductions-scans-and-fixed-lanes.md), [CPU lowering of segments, gathers and scatters](26-cpu-segments-gathers-and-scatters.md).

## What to build

- Google Benchmark suite with interleaved repetitions, medians and CV, reporting against measured ceilings (bandwidth by stream count, FMA peak, storage-span ceiling).
- The samurai and ponio reference workloads rewritten with tensorlib on real data, and the micro-benchmarks of the spec.
- A same-job A/B script (base and PR interleaved), and Cachegrind instruction counts for the front end on every PR.

## Acceptance

- The suite runs on a developer machine with one command and writes JSON results.
- The instruction-count job fails when node recording regresses beyond its threshold.

## Not in this ticket

- The dedicated machine and nightly history (infrastructure ticket).
