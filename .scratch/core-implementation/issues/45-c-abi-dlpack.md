# DLPack import and export

Status: ready-for-agent
Blocked by: 43

## Context

Spec: [C ABI and bindings](../../core-architecture/spec.md#13-c-abi-and-bindings).
Decisions: [C ABI and binding surface](../../core-architecture/issues/31-c-abi-and-binding-surface.md), [Binding surfaces for a lazy tensor library](../../core-architecture/issues/06-binding-surfaces.md).
ADRs: [ADR 0012](../../../docs/adr/0012-one-c-abi-with-explicit-contexts-and-generic-dispatch.md).
Builds on: [C ABI core](43-c-abi-core.md).

## What to build

- Read-only DLPack 1.0 export as a trigger with copy-on-write, `tl_borrow_begin`/`tl_borrow_end` for writable access, `stream` ordering or full synchronisation.
- Zero-copy import for dense row- or column-major buffers on the context's device, copies otherwise, `READ_ONLY` honoured; no implicit gather.

## Acceptance

- NumPy cannot write into an exported tensor; a write inside a borrow is visible after it ends.
