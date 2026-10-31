# Shapes, dtypes and Array API type promotion

Status: ready-for-agent
Blocked by: 01

## Context

Spec: [Tensors, views and expressions](../../core-architecture/spec.md#3-tensors-views-and-expressions), [Broadcasting](../../core-architecture/spec.md#4-broadcasting), [Evaluation plan, kernel IR and code generation](../../core-architecture/spec.md#6-evaluation-plan-kernel-ir-and-code-generation).
Decisions: [Broadcasting semantics and edge cases](../../core-architecture/issues/05-broadcasting-semantics.md), [Tensor data model](../../core-architecture/issues/07-tensor-data-model.md), [Broadcasting design](../../core-architecture/issues/13-broadcasting-design.md), [Kernel IR](../../core-architecture/issues/22-kernel-ir.md).
Builds on: [Build skeleton and repository layout](01-build-skeleton.md).

## What to build

- Run-time dtype descriptor for the closed v1 set (`bool`, `int8` to `int64`, `uint8` to `uint64`, `float16` and `bfloat16` as storage types, `float32`, `float64`, `complex64`, `complex128`) plus `fixed` components.
- Shape type for fixed and dynamic rank, and the Array API broadcast rule ("is 1": `1` against `0` gives `0`), `constexpr` where possible.
- The Array API promotion table and the scalar promotion rules, `constexpr`.

## Acceptance

- `static_assert` tests cover the full promotion table and the broadcast edge cases of [Broadcasting semantics and edge cases](../../core-architecture/issues/05-broadcasting-semantics.md).
- Incompatible shapes produce a descriptive `tl::shape_error`.
