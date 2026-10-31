# Contexts, pending windows and element-wise operations

Status: ready-for-agent
Blocked by: 03, 07

## Context

Spec: [Tensors, views and expressions](../../core-architecture/spec.md#3-tensors-views-and-expressions), [Evaluation](../../core-architecture/spec.md#5-evaluation), [Errors and debugging](../../core-architecture/spec.md#10-errors-and-debugging), [Style rules](../../core-architecture/spec.md#15-style-rules).
Decisions: [When the task graph is evaluated](../../core-architecture/issues/08-evaluation-trigger.md), [Error reporting and debug mode](../../core-architecture/issues/24-error-reporting-and-debug-mode.md), [Coding style and extension points](../../core-architecture/issues/29-coding-style-and-extension-points.md), [C ABI and binding surface](../../core-architecture/issues/31-c-abi-and-binding-surface.md).
ADRs: [ADR 0002](../../../docs/adr/0002-implicit-pending-windows.md), [ADR 0011](../../../docs/adr/0011-thin-templates-over-an-erased-core-and-layered-extension-points.md), [ADR 0012](../../../docs/adr/0012-one-c-abi-with-explicit-contexts-and-generic-dispatch.md).
Builds on: [fixed<T, N...> and element traits](03-fixed-and-element-traits.md), [Tensor storage, copy-on-write and the erased tensor](07-tensor-storage-and-copy-on-write.md).

## What to build

- `tl::context` with its pending window; one implicit context per host thread.
- `tl::expr<T>` and node recording through one-line facades constrained by `tl::operand` (`tl::as_expr`, then a compiled body).
- `tl::site` captured with `std::source_location` in functions and constructors.
- Element-wise Array API operations (arithmetic, comparisons, logical, bitwise on integers, math functions) with promotion recorded as `cast`.
- Triggers: host element read, `data()`, `tl::eval(...)`, `tl::eval_all()`, `tl::wait(...)`; one kernel per operation for now, on the serial backend.

## Acceptance

- `c = a + b; d = a + 2*b` computes the right values on the serial backend.
- A shape mismatch throws `tl::shape_error` at the line that recorded the node.
- `auto t = a * b` is an expression; reading it after `a` changes recomputes the old value.
- Recording costs are measured by a micro-benchmark (order of 5 ns per node on the prototype).

## Not in this ticket

- Fusion (next ticket).
