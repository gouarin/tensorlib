# Distributions

Status: ready-for-agent
Blocked by: 10, 11

## Context

Spec: [Distributions and communication](../../core-architecture/spec.md#9-distributions-and-communication), [Tensors, views and expressions](../../core-architecture/spec.md#3-tensors-views-and-expressions), [Broadcasting](../../core-architecture/spec.md#4-broadcasting).
Decisions: [Distribution kinds](../../core-architecture/issues/18-distribution-kinds.md), [Tensor data model](../../core-architecture/issues/07-tensor-data-model.md), [Broadcasting design](../../core-architecture/issues/13-broadcasting-design.md).
Builds on: [Broadcasting](10-broadcasting.md), [Affine views and slicing](11-affine-views-and-slicing.md).

## What to build

- `tl::distribution`: local, replicated, block over an n-d process grid with explicit or default sizes, ghosts (width, stencil shape, periodicity per axis); immutable and shared.
- Compatibility checks (same object, or congruent communicators and matching ranges), broadcasting rules for partitioned operands, errors suggesting `replicate()` or `redistribute()`.
- `local_part()` and `local_part_with_ghosts()` as views; empty local parts; ghost-width validation.

## Acceptance

- All checks run locally without communication (tested with the `self` communicator and fake congruent communicators).
