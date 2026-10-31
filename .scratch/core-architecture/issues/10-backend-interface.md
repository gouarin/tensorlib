# Backend interface

Type: grilling
Status: resolved
Blocked by: 07, 03, 16

## Question

What must the backend abstraction expose so that Kokkos is the main implementation and a serial backend stays possible? Given [Build, extend or adopt](16-build-extend-or-adopt.md), Kokkos Kernels and Kokkos-FFT live behind the Kokkos backend and `mdspan` is the core view type; given [Feasibility of JIT compilation with Kokkos](03-jit-with-kokkos.md), Kokkos may own memory and streams while generated kernels run beside it. Decide memory spaces, execution spaces, parallel patterns (for, reduce, scan), SIMD, and asynchronous execution, without reducing everything to the lowest common denominator.

## Answer

Decided with the user on 2026-10-30. The vocabulary is in `CONTEXT.md`, and the choice of a run-time interface at the level of lowered kernels is recorded in [ADR 0003](../../../docs/adr/0003-runtime-backend-interface-on-lowered-kernels.md).

- **Boundary.** The core owns the graph. It fuses and splits a pending window, then hands the backend kernels in an internal IR, plus calls to library operations. The backend compiles each kernel (JIT with a kernel cache), launches it, and runs the library calls. Fusion logic exists once, in the core. The IR format is decided in [Kernel IR](22-kernel-ir.md).
- **Run-time interface with a registry.** A backend is a run-time interface, not a template parameter. Backends register at start-up, and serial and Kokkos can coexist in one process. A tensor's memory space names its backend. Core headers do not include Kokkos. There are no dynamically loaded plugins in v1.
- **Interface:**
  - `compile(ir, capabilities)` returns a cached kernel;
  - `launch(kernel, n-d iteration space, buffers, queue)` returns an event;
  - library calls go through a declared table of operation, element type and memory space;
  - `allocate(space, bytes, queue)`, `free(buffer, after events)` and `copy(src, dst, queue)` are ordered by queue;
  - queues and events: `make_queue(device)`, a cross-queue wait on an event, a host wait.
- **Memory spaces.** A space is a backend, a kind and a device number. The kinds are host, device, pinned host and managed. Each backend says which kinds it supports and which spaces a queue can access (managed memory on GH200 or MI300A is reachable from both sides). The serial backend has host only.
- **Asynchrony.** A queue is a Kokkos execution-space instance: a CUDA or HIP stream, or a SYCL queue. Every buffer keeps the event of its last write and the events of its pending reads. Dependencies between queues go through events. The default queue is one per pair of host thread and device, and `tl::on(queue)` picks another.
- **No lowest common denominator.** Every backend runs the whole IR: map, reduction and scan over an n-d iteration space. On top of that, it publishes capabilities (SIMD width, subgroup size, shared memory, supported atomics, native scan, cache sizes, maximum threads), and per-target lowering uses them. The serial backend implements only the required set.
- **Library operations.** gemm, gemv, FFT, sort and dense solvers come from Kokkos Kernels and Kokkos-FFT in the Kokkos backend, and from optional BLAS and FFTW in the serial one. An operation missing for a type or a space is an error raised when the node is built, naming the backend and the type. The core never falls back silently, and never copies to the host behind the user's back.
- **Allocation.** The Kokkos backend allocates storage itself, through Umpire when it is found and a plain allocator otherwise. It exposes buffers as unmanaged Kokkos Views. Kokkos keeps execution spaces and streams but no longer owns tensor storage, which corrects [Feasibility of JIT compilation with Kokkos](03-jit-with-kokkos.md) on this point. The policy (pools, in-place reuse, temporaries) belongs to [Buffer reuse and allocation](21-buffer-reuse-and-allocation.md).
- **Devices.** The interface carries a device number everywhere. The v1 Kokkos backend drives one device per process, so multi-GPU means partitioned tensors over MPI, one rank per GPU. Several devices per process stay a backend extension.
- **Serial backend.** It is the reference and debugging backend: one thread, host only, no dependency (no LLVM, no Kokkos). It runs the IR with an interpreter, like the one in [Prototype of multi-output fusion](09-multi-output-fusion-prototype.md). It is the oracle in tests (same IR, results compared), the backend for an immediate debug mode, and the minimal build for CI and sanitizers. It is not a performance target.
- **Escape hatch for user kernels.** The opt-in header `<tensorlib/kokkos.hpp>` hands out the local part of a tensor as a Kokkos View, with the queue to use after the dependencies have been waited on. The user writes their own `parallel_for` (Kokkos SIMD included), then gives the event back. This is a non-fusible trigger.

## Amendment (2026-10-30)

[Reconcile the core with samurai and ponio](26-reconcile-with-samurai-ponio.md): an interop kernel declares its reads and writes and joins the pending window as a non-fusible node, instead of acting as a trigger. Generic lambdas traced with `tl::apply` enter the IR as recorded operations.

## Amendment (2026-10-30, allocation)

[Buffer reuse and allocation](21-buffer-reuse-and-allocation.md): Umpire is required by the Kokkos backend, with no plain-allocator fallback, and the default pool is `ResourceAwarePool`. Storage comes from one queue-ordered memory resource per space, which users can replace ([ADR 0005](../../../docs/adr/0005-queue-ordered-memory-resources-with-umpire.md)).
