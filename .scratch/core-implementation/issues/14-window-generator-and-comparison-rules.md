# Window generator and comparison rules

Status: ready-for-agent
Blocked by: 10, 11, 13

## Context

Spec: [Testing and benchmarking](../../core-architecture/spec.md#16-testing-and-benchmarking).
Decisions: [Test and benchmark infrastructure](../../core-architecture/issues/32-test-and-benchmark-infrastructure.md), [Prototype of compile times for lighter templates](../../core-architecture/issues/11-compile-time-prototype.md).
Builds on: [Broadcasting](10-broadcasting.md), [Affine views and slicing](11-affine-views-and-slicing.md), [Reductions and scans](13-reductions-and-scans.md).

## What to build

- A random window generator, grown from the one in [Prototype of compile times for lighter templates](../../core-architecture/issues/11-compile-time-prototype.md): element-wise operations, reductions, broadcasts, slices, shifts, views, in-place updates, dead tensors.
- Shrinking to the smallest failing program, printed as readable C++; reproducible seeds.
- Comparison helpers: bit-exact, ulp bound per function, tolerance proportional to contractible operations.
- A fixed-budget run in the per-PR CI (`generated` label) and a nightly job with fresh seeds.

## Acceptance

- An injected bug in the planner is found and shrunk to a few statements.
- The generator is extended by every later ticket that adds a feature (segments, gathers, poison, partitioned tensors).
