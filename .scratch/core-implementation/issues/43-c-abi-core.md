# C ABI core

Status: ready-for-agent
Blocked by: 11, 19, 20

## Context

Spec: [C ABI and bindings](../../core-architecture/spec.md#13-c-abi-and-bindings).
Decisions: [C ABI and binding surface](../../core-architecture/issues/31-c-abi-and-binding-surface.md), [Binding surfaces for a lazy tensor library](../../core-architecture/issues/06-binding-surfaces.md).
ADRs: [ADR 0012](../../../docs/adr/0012-one-c-abi-with-explicit-contexts-and-generic-dispatch.md).
Builds on: [Affine views and slicing](11-affine-views-and-slicing.md), [Exceptions, poison and finalize](19-errors-poison-and-finalize.md), [Library operations](20-library-operations-registry.md).

## What to build

- The `tensorlib::c` component and `<tl/c.h>` in C11: `tl_context*`, `tl_tensor*` with value and view kinds, atomic `tl_retain`/`tl_release` with the dead-output flag.
- Metadata, creation, `tl_view` specs, `tl_assign` with operators, `tl_eval`/`tl_eval_many`.
- The catalogue (`tl_op_find`, `tl_op_count`, `tl_op_info`) generated from the C++ registry, and `tl_apply` with typed attributes.
- `tl_status`, `tl_context_error`, `tl_context_set_site`, `tl_abi_version`, `tl_has_feature`.
- Tests written in C on the serial backend (`abi` label).

## Acceptance

- A C program builds a window, evaluates it and reads results through the catalogue only.
- A context moved between threads keeps its window (tested).
