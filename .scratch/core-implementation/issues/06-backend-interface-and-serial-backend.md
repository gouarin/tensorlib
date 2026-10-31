# Backend interface and serial backend

Status: ready-for-agent
Blocked by: 05

## Context

Spec: [Backends](../../core-architecture/spec.md#7-backends), [Memory](../../core-architecture/spec.md#8-memory).
Decisions: [Backend interface](../../core-architecture/issues/10-backend-interface.md), [Buffer reuse and allocation](../../core-architecture/issues/21-buffer-reuse-and-allocation.md).
ADRs: [ADR 0003](../../../docs/adr/0003-runtime-backend-interface-on-lowered-kernels.md).
Builds on: [Kernel IR for rectangular kernels](05-kernel-ir-rectangular.md).

## What to build

- The run-time backend registry and interface: `compile(ir, capabilities)`, `launch(kernel, space, buffers, queue) -> event`, queue-ordered `allocate`, `free`, `copy`, `make_queue`, events and waits, declared capabilities.
- Memory spaces (backend, kind, device number).
- The type-erased, queue-ordered `tl::memory_resource` interface, and aligned `operator new` for the serial backend.
- The serial backend: host only, one thread, no dependency, an interpreter of the kernel IR.

## Acceptance

- IR written by hand runs on the serial backend and gives the expected values.
- The serial build has no LLVM or Kokkos symbol.
