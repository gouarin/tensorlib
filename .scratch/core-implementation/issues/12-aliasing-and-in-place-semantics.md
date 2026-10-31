# Aliasing and in-place semantics

Status: ready-for-agent
Blocked by: 10, 11

## Context

Spec: [Evaluation](../../core-architecture/spec.md#5-evaluation), [Broadcasting](../../core-architecture/spec.md#4-broadcasting).
Decisions: [When the task graph is evaluated](../../core-architecture/issues/08-evaluation-trigger.md), [Broadcasting design](../../core-architecture/issues/13-broadcasting-design.md).
Builds on: [Broadcasting](10-broadcasting.md), [Affine views and slicing](11-affine-views-and-slicing.md).

## What to build

- Aliasing analysis by comparing index maps: an output also read through a different map gets a copy or a kernel split; an identical map stays fused in place.
- Results always equal statement-by-statement execution with value semantics.

## Acceptance

- `u(tl::range(1, n)) = u(tl::range(0, n - 1)) + v` gives the immediate-execution result.
- `a = a.T + a` style cases are tested for every map kind (shift, transpose, broadcast, flip).
