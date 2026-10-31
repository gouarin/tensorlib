# Memory resources and Umpire pools

Status: ready-for-agent
Blocked by: 06, 19

## Context

Spec: [Memory](../../core-architecture/spec.md#8-memory).
Decisions: [Memory pools for HPC tensors](../../core-architecture/issues/20-memory-pools-research.md), [Buffer reuse and allocation](../../core-architecture/issues/21-buffer-reuse-and-allocation.md).
ADRs: [ADR 0005](../../../docs/adr/0005-queue-ordered-memory-resources-with-umpire.md).
Builds on: [Backend interface and serial backend](06-backend-interface-and-serial-backend.md), [Exceptions, poison and finalize](19-errors-poison-and-finalize.md).

## What to build

- `tl::umpire_resource` over Umpire's `ResourceAwarePool` as the default resource per space; `tl::set_resource` before the first allocation; optional `tl::cccl_resource` and `tl::rmm_resource` headers.
- Frees deferred in queue order; a `(ptr, event)` list for resources that are not queue-ordered.
- `tl::staggered` for large host allocations.
- Out of memory: drain, release, retry once, then poison the window's outputs (`tl::out_of_memory` at the read); `tl::release_memory(space)`; pools destroyed by `tl::finalize`.

## Acceptance

- An OOM injected by a test resource poisons the outputs and throws at the read with the space and byte counts.
- No host blocking on frees (checked with a test resource that records calls).
