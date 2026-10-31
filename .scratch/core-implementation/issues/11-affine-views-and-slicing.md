# Affine views and slicing

Status: ready-for-agent
Blocked by: 09

## Context

Spec: [Tensors, views and expressions](../../core-architecture/spec.md#3-tensors-views-and-expressions).
Decisions: [Views and slicing](../../core-architecture/issues/23-views-and-slicing.md).
ADRs: [ADR 0006](../../../docs/adr/0006-views-are-references-distinct-from-tensors.md).
Builds on: [Planner: fusing a window into kernels](09-window-fusion-planner.md).

## What to build

- `tl::view<T, Rank>` as a reference type holding a shared reference to its owner; `tl::tensor` converts to a view of itself; `tl::tensor_like`.
- `operator()` slicing with `tl::range`, `tl::all`, `tl::newaxis`, `tl::ellipsis`, negative indices; out-of-bounds ranges are errors at node build.
- Views from `permute_dims`, `matrix_transpose`, `flip`, `squeeze`, `expand_dims`; `reshape` returning a value (zero-copy when strides allow).
- Full assignment through a view, and partial writes on a uniquely held buffer in place.
- `tl::box`, `tl::offset`, `tl::interior` reduced to affine views.

## Acceptance

- `v = view(a); c = a; v(0) = 7;` gives `a(0) == 7` and `c(0)` unchanged.
- `u(3)` on a 2-D tensor is row 3, a view.
- A 1-D three-point stencil written with boxes fuses into one kernel.

## Not in this ticket

- Partial writes on shared buffers (fused detach needs several bodies, see the segments ticket).
