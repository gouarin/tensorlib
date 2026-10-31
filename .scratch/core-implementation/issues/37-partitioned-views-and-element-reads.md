# Views and element reads on partitioned tensors

Status: ready-for-agent
Blocked by: 32, 34

## Context

Spec: [Tensors, views and expressions](../../core-architecture/spec.md#3-tensors-views-and-expressions), [Distributions and communication](../../core-architecture/spec.md#9-distributions-and-communication).
Decisions: [Views and slicing](../../core-architecture/issues/23-views-and-slicing.md).
Builds on: [Distributions](32-distributions.md), [Ghost exchanges in the plan](34-ghost-exchanges-in-the-plan.md).

## What to build

- Slices that restrict distributions without communication; an integer on a partitioned axis removes it from the grid; boxes in global coordinates reading ghosts; wider shifts as errors naming the axis.
- Element reads of partitioned tensors as collective triggers broadcast by the owner; collective `tl::compress`.

## Acceptance

- A stencil written with global boxes on a partitioned tensor inserts exactly the expected exchanges.
