# Run-time backend interface on lowered kernels

A backend is a run-time interface, registered at start-up, and the core hands it kernels already lowered to an internal IR, plus calls to named library operations. The core owns the graph, fusion and kernel splitting. Backends only compile (JIT with a kernel cache), launch on queues with events, allocate, and run the library operations. Serial and Kokkos backends coexist in one process. Core headers never include Kokkos.

Kokkos users will expect a backend made of compile-time parallel patterns (`parallel_for`, `parallel_reduce` with functors). We rejected that, for three reasons:

- generated kernels run beside Kokkos, not through it, because Kokkos has no run-time code generation;
- templates over functors bring back the compile times this library is meant to avoid;
- it pulls every target down to what all of them share.

We also rejected handing each backend the raw graph to fuse its own way, as XLA does per platform, because each backend would then rewrite and maintain its own fusion.

## Consequences

- Fusion is written once, but the IR becomes a stable internal contract between the core and every backend.
- Backends publish capabilities (SIMD width, subgroups, shared memory, atomics, native scan) so per-target lowering can use them without changing the IR.
- A user lambda cannot enter the IR. Custom kernels go through an explicit Kokkos interop header, which acts as a non-fusible trigger.
- Kokkos keeps execution spaces and streams. Tensor storage comes from the backend's queue-ordered allocator and is exposed as unmanaged Views.
- Amended on 2026-10-30 ([Reconcile the core with samurai and ponio](../../.scratch/core-architecture/issues/26-reconcile-with-samurai-ponio.md)): generic lambdas can be traced with symbolic arguments through `tl::apply`, so their recorded operations enter the IR and fuse. Interop kernels join the pending window as non-fusible nodes instead of acting as triggers.
- Amended on 2026-10-30 ([Code-generation strategy](../../.scratch/core-architecture/issues/12-code-generation-strategy.md)): the backend compiles kernels with an in-process LLVM, its only code generator ([ADR 0008](0008-in-process-llvm-as-the-only-code-generator.md)).
