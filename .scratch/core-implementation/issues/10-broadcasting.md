# Broadcasting

Status: ready-for-agent
Blocked by: 08

## Context

Spec: [Broadcasting](../../core-architecture/spec.md#4-broadcasting).
Decisions: [Broadcasting semantics and edge cases](../../core-architecture/issues/05-broadcasting-semantics.md), [Broadcasting design](../../core-architecture/issues/13-broadcasting-design.md).
Builds on: [Contexts, pending windows and element-wise operations](08-contexts-windows-and-elementwise-operations.md).

## What to build

- Broadcasting at node build with the exact Array API rule; in-place and compound assignment never reshape the target.
- Index maps with structural stride 0; flat loop when all shapes match.
- `tl::broadcast_to` and `tl::broadcast_arrays` returning expressions.
- `fixed` element rules: `tensor<vec3> + vec3`, a same-shape `tensor<double>` scaling components, `as_components(u)`, and the ambiguity error.

## Acceptance

- Every case of [Broadcasting semantics and edge cases](../../core-architecture/issues/05-broadcasting-semantics.md) is a test.
- Writing through a broadcast does not compile.
