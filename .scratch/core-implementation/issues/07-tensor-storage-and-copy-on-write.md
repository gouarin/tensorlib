# Tensor storage, copy-on-write and the erased tensor

Status: ready-for-agent
Blocked by: 04, 06

## Context

Spec: [Tensors, views and expressions](../../core-architecture/spec.md#3-tensors-views-and-expressions).
Decisions: [Tensor data model](../../core-architecture/issues/07-tensor-data-model.md).
ADRs: [ADR 0001](../../../docs/adr/0001-value-semantics-with-copy-on-write.md).
Builds on: [Shapes, dtypes and Array API type promotion](04-shapes-dtypes-and-promotion.md), [Backend interface and serial backend](06-backend-interface-and-serial-backend.md).

## What to build

- `tl::tensor<T, Rank = dynamic>` over refcounted storage with value semantics and copy-on-write; fixed to dynamic rank implicit, `rank_cast<R>` checked.
- Run-time memory space and layout (row-major default, column-major, strided); logical C order for `reshape` and `ravel`.
- `tl::any_tensor` and checked conversions with typed tensors.
- Creation functions of the Array API (`empty`, `zeros`, `ones`, `full`, `arange`, `linspace`, `eye`, from host data), `to(space)`, `shape()`, `dtype()`.

## Acceptance

- Copies share storage until a write; `v = view(a); c = a;` semantics are tested once views exist (see the views ticket), and plain copies are tested here.
- Every creation function has a documentation example run in CI.
