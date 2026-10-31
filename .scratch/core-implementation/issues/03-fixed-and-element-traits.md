# fixed<T, N...> and element traits

Status: ready-for-agent
Blocked by: 01

## Context

Spec: [Tensors, views and expressions](../../core-architecture/spec.md#3-tensors-views-and-expressions), [Extension points](../../core-architecture/spec.md#12-extension-points), [Style rules](../../core-architecture/spec.md#15-style-rules).
Decisions: [Tensor data model](../../core-architecture/issues/07-tensor-data-model.md), [Prototype of compile times for lighter templates](../../core-architecture/issues/11-compile-time-prototype.md), [Coding style and extension points](../../core-architecture/issues/29-coding-style-and-extension-points.md).
ADRs: [ADR 0011](../../../docs/adr/0011-thin-templates-over-an-erased-core-and-layered-extension-points.md).
Builds on: [Build skeleton and repository layout](01-build-skeleton.md).

## What to build

- `tl::fixed<T, N...>`: inline storage, value semantics, arithmetic and comparisons, `constexpr` throughout, evaluated eagerly (no expression templates).
- `tl::element_traits` and the `tl::element` concept for scalar types and `fixed` over them; user specialisation documented with an example.
- Only whitelisted standard headers in public headers (`<cmath>` is allowed for `fixed`).

## Acceptance

- Arithmetic, broadcasting of a scalar over components and comparisons are tested with `static_assert`.
- A user type specialising `tl::element_traits` satisfies `tl::element` in a test.
- The header check passes.
