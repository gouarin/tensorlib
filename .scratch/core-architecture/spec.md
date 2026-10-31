# tensorlib core architecture

Status: approved by the user, 2026-10-31. Map: [Core architecture of tensorlib](map.md).

This spec gathers the decisions of the map into one description of the core, ready to be split into implementation tickets. It states what was decided and links to the ticket that holds the details and the reasoning. When this spec and a ticket disagree, the ticket wins, except for the points settled in [Reconciliations](#reconciliations). Terms in bold are defined in [`CONTEXT.md`](../../CONTEXT.md), and the ADRs are in [`docs/adr/`](../../docs/adr/).

## 1. Purpose and scope

tensorlib is a C++ tensor library close to NumPy and the Array API standard, in the spirit of xtensor. It fuses operations at run time across several outputs, runs on Kokkos (CPU threads, SIMD, NVIDIA, AMD and Intel GPUs) and carries partitioned tensors over MPI from the start. Its first users are C++ HPC developers writing simulation codes, and its acceptance workloads are the hot loops of [samurai](https://github.com/hpc-maths/samurai) and [ponio](https://github.com/hpc-maths/ponio) ([Build, extend or adopt](issues/16-build-extend-or-adopt.md), [Requirements from samurai and ponio](issues/25-samurai-ponio-requirements.md)).

tensorlib builds only the C++ tensor layer. It depends on mdspan, on Kokkos, Kokkos Kernels and Kokkos-FFT through its Kokkos backend, on LLVM for code generation, on Umpire for memory pools and on MPI behind its own communication interface. It interoperates with PETSc and Trilinos/Tpetra instead of writing a linear algebra layer ([Existing HPC libraries](issues/15-existing-hpc-libraries.md)).

Out of scope for this effort, as listed on the map: implementing the MPI backend and multi-GPU execution beyond the hooks fixed here, implementing the Python, Julia and Rust bindings, a PETSc-like layer, distributed FFT, a Python-first distributed NumPy, and samurai storing its ghosts apart from its intervals.

The first milestone runs in shared memory (CPU and one GPU per process), but every interface below already carries distributions and communicators.

## 2. Architecture at a glance

```text
  C++ API (tl::tensor, tl::view, tl::expr, fixed, free functions)      C ABI <tl/c.h>
        |  thin template facades, one line each (ADR 0011)                    |
        v                                                                     v
  erased core: recording front end ------------------------------------------+
        |  nodes recorded into the pending window of a context (ADR 0002, 0012)
        |  shapes, dtypes, distributions checked at node build
        v  trigger
  planner: evaluation plan (ADR 0009)
        |  fusion, splitting by two ceilings, aliasing, buffer plan,
        |  ghost exchanges and reductions across ranks, plan cache
        +--> kernels in the kernel IR -------> backend.compile / launch (ADR 0003)
        +--> library operations, interop kernels, PETSc/Tpetra operators
        +--> communication patterns -------> tl::communicator (ADR 0010)
        v
  backends: serial (interpreter, oracle)  |  Kokkos + in-process LLVM (ADR 0008)
  memory: one queue-ordered tl::memory_resource per space, Umpire by default (ADR 0005)
```

## 3. Tensors, views and expressions

From [Tensor data model](issues/07-tensor-data-model.md), [Views and slicing](issues/23-views-and-slicing.md) and [When the task graph is evaluated](issues/08-evaluation-trigger.md).

- **Types.**
  - `tl::tensor<T, Rank = dynamic>` is one template for fixed and dynamic rank. Fixed to dynamic is implicit and free, `rank_cast<R>` goes the other way with a run-time check, and mixing ranks gives dynamic rank.
  - `tl::fixed<T, N...>` is a small tensor with inline storage, evaluated eagerly and `constexpr`. It is used in local computations and as an element type (`tensor<vec3>`), and is never partitioned.
  - `tl::view<T, Rank>` is a reference type (**view**), and `tl::expr<T>` is a deferred **expression**, never stored and recomputed in each consumer.
  - `tl::any_tensor` is the **erased tensor**: dtype, rank, memory space and layout known only at run time. It serves the C ABI and the code generator. Conversions with typed tensors are explicit and checked.
- **Value semantics.** Copies are values with copy-on-write over refcounted storage ([ADR 0001](../../docs/adr/0001-value-semantics-with-copy-on-write.md)). A view refers to its owner, so a write through a view detaches the owner from its copies first.
- **Element types.** Static in C++. v1 has `bool`, `int8` to `int64`, `uint8` to `uint64`, `float16` and `bfloat16` (storage types computed in `float32`), `float32`, `float64`, `complex64`, `complex128`, and `fixed` of these. User element types are described by specialising `tl::element_traits` ([Coding style and extension points](issues/29-coding-style-and-extension-points.md)).
- **Run-time attributes.** Memory space (backend, kind host/device/pinned/managed, device number) and layout (row-major by default, column-major and strided allowed) are run-time attributes, not part of the type. Logical semantics (`reshape`, `ravel`) always follow C order. Data moves explicitly with `to(space)`, and mixing spaces in one operation is an error at node build.
- **Distribution.** Every tensor has one, local by default (section 9).
- **Expressions.** `auto t = a * b` is a `tl::expr`. It captures the values of its inputs through copy-on-write. `tl::tensor<T> t = a * b` declares a tensor, written at evaluation. `tl::broadcast_to` and slices of an expression are expressions, read-only.
- **Slicing.** Through `operator()` with Array API semantics: `tl::range(start, stop, step)`, `tl::all`, `tl::newaxis`, `tl::ellipsis`, negative indices from the end. As many integers as the rank read one element, a host read and so a trigger. Anything else returns a view. An out-of-bounds range is an error at node build.
- **Writes through views.** A write covering the whole owner with the identity map is a full assignment. A partial write on a unique buffer writes in place. A partial write on a shared buffer detaches it in one fused pass, and all partial writes to one owner in a window share that detach.
- **Stencils.** `tl::box` with `tl::offset` shifts, `tl::interior(u, w)`. They reduce to affine views. On a partitioned tensor, boxes are in global coordinates and a shift reads ghosts when their width covers it.
- **Non-affine access.** `u(mask)` is write-only (an in-place `where`); extraction uses `tl::compress`, a non-fusible trigger. `u(idx)` reads as a fusible gather. `u(idx) = e` is a scatter (duplicates unspecified, an error under the `scatter` check). `u(idx) += e` accumulates duplicates ([ADR 0006](../../docs/adr/0006-views-are-references-distinct-from-tensors.md)). Integer tensors have the bitwise operators.
- **Segmented views.** `tl::segment_space S(lengths)`, rebuilt by samurai at each adaptation, and `S.on(offsets, stride)` give writable views with one base offset per segment. Views on the same space are congruent by identity. A segment space can declare its gaps dead and accessible, which allows bridging (section 6).
- **Shape changes.** `permute_dims`, `matrix_transpose`, `flip`, `squeeze` and `expand_dims` return views. `reshape` always returns a value (zero-copy through copy-on-write when strides allow).
- **Output parameters.** `tl::tensor` converts implicitly to a view of itself, so `tl::view<T>` is the type of output parameters: `f(double t, const tl::tensor<double>& y, tl::view<double> dy)`.

## 4. Broadcasting

From [Broadcasting semantics and edge cases](issues/05-broadcasting-semantics.md) and [Broadcasting design](issues/13-broadcasting-design.md).

- The Array API algorithm exactly ("is 1": `1` against `0` gives `0`). Shapes are checked at node build, so a mismatch is reported at its line.
- In-place and compound assignment never reshape the target. The right-hand side must broadcast to it.
- Inside a kernel, each operand reads through one affine index map per segment: base offset, one stride per result dimension (0 when stretched), component stride for `fixed`. Slices, transposes and shifts compose into it. Aliasing is decided by comparing maps.
- `tensor<vec3> + vec3` broadcasts the value, a `tensor<double>` of the same shape scales every component, and anything else goes through `as_components(u)`. `u + tensor<double>({3})` is an ambiguity error.
- Partitioned operands must have matching partitions on every real dimension, never be split along a stretched one, or be replicated. Anything needing communication is an error suggesting `replicate()` or `redistribute()`.

## 5. Evaluation

From [When the task graph is evaluated](issues/08-evaluation-trigger.md) ([ADR 0002](../../docs/adr/0002-implicit-pending-windows.md)) and [C ABI and binding surface](issues/31-c-abi-and-binding-surface.md) ([ADR 0012](../../docs/adr/0012-one-c-abi-with-explicit-contexts-and-generic-dispatch.md)).

- **Context and pending window.** Operations are recorded into the **pending window** of a **context**. Each C++ host thread has an implicit context, and the C ABI passes one explicitly. A context is used from one thread at a time and may move between threads.
- **Triggers.** A trigger evaluates every live output of the window together:
  - host reads and exports: element access, `data()`, DLPack, borrows, `to(space)`, a scalar in control flow;
  - `tl::eval(...)`, `tl::eval_all()`;
  - a non-fusible consumer that needs its inputs evaluated (library operation);
  - deterministic limits: node count, planned bytes of temporaries, never local memory pressure.
- **SPMD.** Every rank records the same graph in the same order. A trigger involving a partitioned tensor is collective.
- **Results as if immediate.** Results are those of statement-by-statement execution with value semantics. An output also read through a different index map gets a copy or a kernel split. There is no `noalias` escape hatch.
- **Asynchrony.** A trigger plans, compiles and launches, then returns. The host waits only on a host read, an export or `tl::wait(...)`.
- **Non-fusible nodes inside the window.** Kokkos interop kernels, `tl::petsc::mult`, `tl::tpetra::apply`, redistributions, `tl::compress` and library operations are plan nodes that declare their reads and writes. Interop kernels and operator nodes join the window instead of flushing it ([Reconcile the core with samurai and ponio](issues/26-reconcile-with-samurai-ponio.md)).
- **Traced functions.** `tl::apply` traces a generic lambda once on symbolic arguments. It records ordinary operations that fuse with the window. Data-dependent branches use `tl::where`.

## 6. Evaluation plan, kernel IR and code generation

From [Kernel IR](issues/22-kernel-ir.md) ([ADR 0009](../../docs/adr/0009-evaluation-plan-and-per-element-ssa-kernel-ir.md)), [Code-generation strategy](issues/12-code-generation-strategy.md) ([ADR 0008](../../docs/adr/0008-in-process-llvm-as-the-only-code-generator.md)), the prototypes [Multi-output fusion](issues/09-multi-output-fusion-prototype.md), [Compile times](issues/11-compile-time-prototype.md), [LLVM JIT on CPU](issues/27-llvm-jit-cpu-prototype.md) and [Lowering of short segments](issues/30-lowering-of-short-segments.md).

### Evaluation plan

- The **evaluation plan** orders kernels and non-fusible nodes: library operations, interop kernels, external operators, ghost exchanges, reductions across ranks, redistributions.
- It is cached automatically per window signature (structure key plus buffer identity and aliasing). A window seen before pays only recording (about 5 ns per node) and hashing (under 1 us). Turning a stable plan into a CUDA or HIP graph is a backend optimisation to measure. There is no explicit capture API in v1.
- **Kernel size.** The planner splits a fused kernel when it exceeds `TL_MAX_KERNEL_OPS` (operations) or `TL_MAX_KERNEL_LIVE` (live lanes), cutting where the stored temporary is smallest. The rule is deterministic and identical on every rank. First CPU defaults on NEON: about 1 000 operations and 24 lanes.

### Kernel IR

- A kernel is a per-element SSA program over an iteration space that is either n-d rectangular or **segmented** (1-d segments with trailing ordinary axes, one body tag per segment). There are no loops, tiles or shared memory in the IR: lowering decides them per target.
- Operands: element type, affine map per segment or gather. Stride 0, the inner-stride-is-1 flag and stencil shifts are structural. Other strides, offsets, extents, segment tables and scalars are arguments, so samurai's adaptations and ponio's `dt` never recompile. `tl::constant(2.0)` puts a value into the key on purpose.
- `fixed` values are lane-typed SSA values (`f64x5`) loaded with a component stride. AoS is the default layout.
- Scalar types are the closed set of section 3. The IR has no implicit conversions; the front end applies Array API promotion and records `cast`.
- Outputs: stores (optionally masked), reductions (built-in or traced combiners declared associative, canonical block order), scatters. Scans are kernels of their own.
- Affine scatter-adds are turned into gathers. Indirect scatter-adds use atomics by default, and a sort-based deterministic path under `TL_FP=strict` or `tl::deterministic(idx)`.
- Several bodies per kernel, selected per segment (interior and boundaries, fused detach).
- 64-bit signed indices, with a 32-bit variant selected at launch when every address fits and the backend declares the capability (on for GPUs, off for CPUs pending measurement).
- One closed set of primitives plus `call`, which invokes a registered external function (host symbol, optional LLVM bitcode per target).
- Versioned text form that round-trips (`TL_DEBUG=dump`, `tl-opt` and FileCheck tests), binary serialisation for kernel bundles, structural hash computed incrementally while recording.

### Code generation

- One generator from the kernel IR to LLVM IR, compiled by an LLVM linked into the Kokkos backend: ORC on CPU, NVPTX then the driver's PTX JIT on NVIDIA, AMDGPU on AMD, SPIR-V then Level Zero on Intel.
- Two-level kernel cache (memory, then disk at `$XDG_CACHE_HOME/tensorlib` or `TL_CACHE_DIR`). Only local rank 0 of a node writes, through atomic rename. A kernel bundle produced by a training run or by `tl-aot` loads read-only. `TL_JIT=off` makes a kernel missing from the bundle an error.
- CPU: Kokkos owns the threads. The JIT emits `body(begin, end, args)`, called by a precompiled trampoline from `Kokkos::parallel_for` or `parallel_reduce`. The inner loop is scalar with `noalias` and vectorisation metadata. Explicit vector code is emitted only for `fixed` lanes and canonical reductions.
- **Floating point.** FMA contraction allowed. Reductions use canonical blocks independent of thread count and SIMD width, combined in a fixed tree: bit-identical for any thread count on one machine. Across ranks, reproducible in practice by default and guaranteed at a fixed rank count under `TL_FP=strict` (section 9). `TL_FP=strict` also turns contraction off. Fast-math flags on NaN and Inf are never used.
- **Segmented kernels** ([Lowering of short segments](issues/30-lowering-of-short-segments.md)):
  - The plan **bridges** consecutive segments with a storage gap of at most 4 cells (tunable) when the segment space declares its gaps dead and accessible, all operands share the same offsets, and the same body runs on both sides.
  - The table is a struct of arrays: one length and one base per segment, the base shared by every operand that shares the offset table, one base column per distinct table otherwise.
  - Segments keep storage order. Work is chunked by cell count, never by a fixed number of segments, never one segment per chunk.
  - The tail of a segment is LLVM's own loop. Masked tails are to be measured on SVE, AVX-512 and GPUs.

## 7. Backends

From [Backend interface](issues/10-backend-interface.md) ([ADR 0003](../../docs/adr/0003-runtime-backend-interface-on-lowered-kernels.md)) and [Feasibility of JIT compilation with Kokkos](issues/03-jit-with-kokkos.md).

- A backend is a run-time interface registered at start-up. Serial and Kokkos can coexist, a tensor's memory space names its backend, and core headers never include Kokkos. No dynamically loaded plugins in v1.
- Interface:
  - `compile(ir, capabilities)` returns a cached kernel;
  - `launch(kernel, space, buffers, queue)` returns an event;
  - library calls through a declared table of operation, element type and memory space;
  - queue-ordered `allocate`, `free` and `copy`;
  - queues and events: `make_queue(device)`, cross-queue waits, host waits.
- Capabilities (SIMD width, subgroup size, shared memory, atomics, native scan, cache sizes, 32-bit indices, native events, kernel-size ceilings) drive per-target lowering. Every backend runs the whole IR.
- A missing library operation for a type or space is an error at node build. The core never falls back silently and never copies to the host behind the user's back.
- **Kokkos backend.** One device per process, so multi-GPU means one rank per GPU. Queues are Kokkos execution-space instances. Storage comes from the memory resources (section 8) and is exposed as unmanaged Views. Library operations come from Kokkos Kernels and Kokkos-FFT. In-process LLVM and Umpire are required.
- **Serial backend.** One thread, host only, no dependency (no LLVM, no Kokkos). It interprets the IR, is the test oracle and the minimal build for CI and sanitizers, and is not a performance target. Optional BLAS and FFTW for library operations.
- **Kokkos interop.** `<tl/kokkos.hpp>` hands out a tensor's local part as a Kokkos View with the queue to use. The user's kernel declares reads and writes and enters the window as a non-fusible node.

## 8. Memory

From [Memory pools for HPC tensors](issues/20-memory-pools-research.md) and [Buffer reuse and allocation](issues/21-buffer-reuse-and-allocation.md) ([ADR 0005](../../docs/adr/0005-queue-ordered-memory-resources-with-umpire.md)).

- One type-erased, queue-ordered `tl::memory_resource` per memory space. The default is an Umpire `ResourceAwarePool`. `tl::umpire_resource`, `tl::cccl_resource` and `tl::rmm_resource` let an application share its pool. `tl::set_resource` must run before the first allocation in a space.
- Only an assignment target takes back its old buffer, checked on the host at evaluation. Capacity rule: reuse when the new size fits and fills at least half.
- Frees are deferred in queue order and never block the host.
- Temporaries between split kernels are allocated at their first writer and freed after their last reader is enqueued. A planned arena can replace this behind the planner interface.
- `tl::staggered` shifts large host allocations by `(i × 64 B) mod 4 KiB` (off on devices until measured). Acceptance test: ponio `heat_bench` with `rkc_202`.
- `t.resize(shape)` means `t = tl::empty<T>(shape)`.
- Out of memory: drain, release, retry once, then poison the window's outputs; `tl::out_of_memory` is thrown at the read (section 10). `tl::release_memory(space)` returns pooled memory. `tl::finalize` destroys pools before `Kokkos::finalize`.
- No NUMA logic in v1: one rank per NUMA domain.

## 9. Distributions and communication

From [Distribution kinds](issues/18-distribution-kinds.md) ([ADR 0004](../../docs/adr/0004-automatic-ghost-updates-in-the-graph.md)), [Communication in the evaluation plan](issues/28-communication-in-the-evaluation-plan.md) ([ADR 0010](../../docs/adr/0010-persistent-communication-patterns-and-a-progress-thread.md)) and [Reconcile the core with samurai and ponio](issues/26-reconcile-with-samurai-ponio.md).

- **Kinds in v1.** Local, replicated, and block over an n-d Cartesian process grid with explicit sizes per axis or a `PETSC_DECIDE`-like default split. Block-cyclic and index-list distributions come later.
- A distribution is immutable and shared. It holds the communicator, grid, ownership ranges, and per axis the ghost width, periodicity and stencil shape (box or star).
- **Compatibility.** Same object, or congruent communicators with matching ownership ranges on the combined axes; ghosts ignored; checked locally. Mixing incompatible distributions is an error suggesting `redistribute()`. A replicated operand combines with any partitioned one. Local with partitioned is an error.
- **Ghost band** (regular distributions only). Local storage includes it. `local_part()` sees owned cells, and `local_part_with_ghosts()` sees owned cells plus ghosts. Empty local parts are legal. A ghost width larger than a neighbour's part is an error at creation.
- **samurai.** Fields are local tensors; samurai keeps its mesh, balancing and MPI. tensorlib provides segmented gather and scatter kernels for its halos.
- **Automatic ghost updates.** The plan inserts an exchange only when a kernel of the window reads stale ghosts. The send goes right after an in-window writer, or just before the reader otherwise. No exchange stays in flight after an evaluation. `u.update_ghosts()` forces one.
- **Overlap.** A kernel whose interior exceeds `TL_OVERLAP_MIN_CELLS` is split into interior and boundary launches (`TL_OVERLAP=always|never|auto`). It is a local decision that never changes messages.
- **Messages.** One message per neighbour per exchange point, in one phase to every stencil neighbour. Generated pack and unpack kernels. A tensorlib header with the poison flag and the plan hash.
- **Reductions across ranks.** One non-blocking packed collective per exchange point, carrying the poison word and the plan hash. Completion is an event, so a 0-d result such as `dt` stays on the device. MPI allreduce with a non-commutative tensorlib operation by default. A fixed-shape tree in rank order under `TL_FP=strict` or `tl::deterministic(...)`.
- **Redistribution.** Explicit, non-fusible, non-blocking plan node. v1 cases: replicated to block, block to replicated, block to one rank and back, block to block. Patterns are cached per pair of distributions.
- **`tl::communicator`.** `rank`, `size`, `dup`, `split`, congruence, and three persistent pattern factories (`halo_pattern`, `reduce_pattern`, `exchange_pattern`) with `start` and `complete -> event`. The MPI implementation uses MPI 4 persistent and neighbourhood collectives with a fallback. `self` serves one rank and serial builds.
- **Progress.** One progress thread per rank (`TL_COMM_PROGRESS=thread|caller`). GPU-aware MPI when detected, pinned host staging otherwise.
- **Determinism.** Every decision that changes a message depends only on the recorded operations, distributions and global settings. A mismatch of the plan hash raises `tl::spmd_error`.

## 10. Errors and debugging

From [Error reporting and debug mode](issues/24-error-reporting-and-debug-mode.md) ([ADR 0007](../../docs/adr/0007-errors-poison-results-across-ranks.md)) and [Minimum C++ standard and build system](issues/14-cxx-standard-and-build.md).

- Typed exceptions derived from `std::exception` (`tl::shape_error`, `tl::unsupported_operation`, `tl::device_error`, `tl::out_of_memory`, `tl::spmd_error`, `tl::remote_error`). Device code never throws. No `-fno-exceptions` support in v1.
- **Sites.** Every statement records a `tl::site` (file, line, function) through `std::source_location`, captured in functions and constructors, not in operators. Kernels are named after their fused sites and show up in Kokkos Tools, Nsight and rocprof.
- **Host usage errors** (shapes, distributions, borrow locks, PETSc/Tpetra split mismatches) are always checked and thrown at once with the site.
- **Kernel errors in release.** A failed kernel or an OOM poisons its outputs. Reading a poisoned tensor throws. `tl::wait()` and `tl::eval_all()` throw the first unreported error of the context, and unread errors are reported at `tl::finalize`. Sticky device errors are `fatal()`.
- **Distributed.** Evaluation never throws. Poison travels in ghost headers and reduction status words, and every rank throws at the same collective read: the original error where it started, `tl::remote_error` elsewhere.
- **Checks.** `TL_DEBUG=...` or `tl::debug_scope` (a collective trigger) turn on, without rebuilding: `sync`, `bounds`, `nan` (from finite inputs), `uninit`, `scatter`, `collectives`, `leaks`, `detaches` and `reads`. They run on the real backend and keep fusion, except `sync`. `TL_ENABLE_ASSERTIONS` covers host-side header assertions only.
- **Where throwing is impossible.** A reference still held by PETSc or Tpetra when a borrow ends: kept alive and warned in release, `std::terminate` with checks on. Live tensors at `tl::finalize` are listed, and are a failing exit code under `leaks`.

## 11. Interop with PETSc and Tpetra

From [Zero-copy interop with PETSc and Tpetra](issues/17-petsc-tpetra-interop.md) and [research/17](research/17-petsc-tpetra-apis.md).

- Optional components `tensorlib::petsc` and `tensorlib::tpetra`, outside the core header whitelist. The core never depends on them.
- v1 scope: `Vec` (Seq and MPI), DMDA vectors, Tpetra `MultiVector` and `Vector` in both directions. `Mat` and `CrsMatrix` are opaque operators.
- **Borrow** (tensorlib owns): `tl::petsc::borrow(t, access)`, `tl::tpetra::borrow(t, access)`. Scoped: evaluate, wait, make unique for writes, lock `t`, give it back at scope end with ghosts marked stale after a write. `borrow_local` gives a ghosted tensor as a DMDA local vector without copy. `borrow_packed` packs it for a global vector.
- **Wrap** (the other library owns): `tl::petsc::wrap(v, access)`, `tl::tpetra::wrap(mv, access)`. Scoped, an ordinary tensor over borrowed storage, deep copies of remaining sharers at scope end. Resizing a wrapped tensor is an error.
- **Distributions.** `tl::petsc::dmda(d)` builds and caches a DMDA from a block distribution. `tl::petsc::distribution(da)` reads one back and keeps the DM. Contiguous Tpetra maps map to 1-D blocks. A non-contiguous map is an error.
- **Device synchronisation.** Full synchronisation by default. Opt-in `stream_ordered` with native events. tensorlib, PETSc with Kokkos and Trilinos must share one Kokkos installation.
- **Operators.** `tl::petsc::mult`, `mult_add`, `mult_transpose` and `tl::tpetra::apply` are non-fusible window nodes. Solvers stay inside explicit borrows.
- Borrows and wraps of partitioned tensors are collective.

## 12. Extension points

From [Coding style and extension points](issues/29-coding-style-and-extension-points.md) ([ADR 0011](../../docs/adr/0011-thin-templates-over-an-erased-core-and-layered-extension-points.md)).

- Element types: specialise `tl::element_traits`, checked by `tl::element`.
- Run-time behaviour: registries behind type-erased interfaces for backends, memory resources, library operations and external functions.
- New operations: compositions, either plain functions on `tl::expr` or traced functions (`tl::apply`, traced combiners in `tl::reduce`).
- External functions: a pure scalar function registered with a required host symbol and optional LLVM bitcode per target (CMake helper `tl_add_device_function`), reached through the IR's `call`. A GPU target without bitcode raises `tl::unsupported_operation` at plan build.
- No customisation point objects, no `tag_invoke`, no user IR primitives.

## 13. C ABI and bindings

From [Binding surfaces for a lazy tensor library](issues/06-binding-surfaces.md) and [C ABI and binding surface](issues/31-c-abi-and-binding-surface.md) ([ADR 0012](../../docs/adr/0012-one-c-abi-with-explicit-contexts-and-generic-dispatch.md)).

- Python, Julia and Rust all go through one C ABI: component `c`, header `<tl/c.h>` in C11, `tl_` prefix, target `tensorlib::c`, implementation in `src/c/`. nanobind or Cython may wrap it for Python, never bind the C++ types.
- **Handles.** One opaque refcounted `tl_tensor*` (value or view), with no expressions. `tl_context*` owns the window and the last error. `tl_retain`/`tl_release` are atomic and safe from any thread.
- **Functions.**
  - An explicit core: lifetime, metadata, creation, `tl_view` with specs, `tl_assign`, `tl_eval`/`tl_eval_many`, DLPack, errors, version.
  - Every array operation goes through `tl_apply` on a `tl_op*` from `tl_op_find`, with typed attributes, and a catalogue generated from the C++ registry.
  - `tl_abi_version()` major.minor, additions only, `tl_has_feature`.
- **Types and attributes.**
  - Structured `tl_dtype` with up to 4 component dimensions, so `fixed` tensors are creatable. User element types pass as opaque.
  - Memory space as an enumeration plus device, row- and column-major layouts.
  - `tl_communicator*` from the Fortran MPI handle, opaque `tl_distribution*`, `tl_local_part` returning a view.
- **Extensions.** `tl_trace` makes a catalogue operation or combiner from a binding callback, and external functions register with a host symbol free of interpreter locks. Backends and resources are chosen by name only.
- **Semantics.** The C++ semantics: `a[idx] += v` accumulates. Bindings may clip ranges or map a mask read to `compress`, and must never rewrite a compound indexed assignment into a read then a write.
- **Errors.** `tl_status` codes. Detail in the context (`tl_context_error`), sites through `tl_context_set_site`.
- **DLPack.**
  - Export is a trigger and read-only, with copy-on-write. Writable access only inside `tl_borrow_begin`/`tl_borrow_end`.
  - `stream` given means ordering on the consumer's queue; absent means full synchronisation.
  - Import is zero-copy when the buffer is dense, row- or column-major and on the context's device. Otherwise it copies.
  - No implicit gather of partitioned tensors.

## 14. Build, toolchains and packaging

From [C++ standard support across target toolchains](issues/04-toolchain-support-matrix.md), [Techniques to cut compile times](issues/02-compile-time-techniques.md), [Minimum C++ standard and build system](issues/14-cxx-standard-and-build.md) and [Prototype of compile times for lighter templates](issues/11-compile-time-prototype.md).

- C++20 everywhere. Device toolchains follow Kokkos 5 (NVCC 12.2, ROCm 6.2, IntelLLVM 2024.2.1). The host floor is GCC 11.1, Clang 16 (with libstdc++ 11 or libc++ 16 and newer), Apple Clang 16 and MSVC 19.30, above Kokkos 5, for `<source_location>`. No modules. Public headers are strict C++20; C++23 enters the API all at once when the floor moves.
- Public headers include only `<cstddef>`, `<cstdint>`, `<type_traits>`, `<concepts>`, `<utility>`, `<initializer_list>`, `<span>`, `<array>`, `<compare>`, `<limits>`, `<source_location>` for `tl::site`, and `<cmath>` for `fixed`. No expression templates for run-time-sized tensors.
- CMake 3.25.2 or newer with presets. `find_package(tensorlib)` exports `tensorlib::core`, `tensorlib::kokkos`, `tensorlib::petsc`, `tensorlib::tpetra` and `tensorlib::c`. Dependencies through `find_package`, `FetchContent` opt-in with `TL_FETCH_DEPS=ON`, nothing vendored.
- Packages: Spack (`+cuda`, `+rocm`, `+sycl`, `+mpi`, `+umpire`, `+petsc`, `+trilinos`) and conda-forge (CPU and CUDA), with one shared Kokkos installation.
- Platforms: Linux (GCC, Clang), macOS (Apple Clang), Windows (MSVC; ORC serves the CPU there, with the serial backend as fallback), NVIDIA, AMD and Intel GPUs.

## 15. Style rules

From [Coding style and extension points](issues/29-coding-style-and-extension-points.md). The code must be very easy to read, concise, modern C++ and very flexible; these rules make that checkable.

- **Call style.** Operations are free functions with Array API names and semantics. Options are a small aggregate per family with designated initializers: `tl::sum(a, {.axis = 0, .keepdims = true})`. One form per call. `tensor` members are only intrinsic properties (`shape()`, `dtype()`, `distribution()`), `operator()` for views, and assignment.
- **Thin templates.** A public operation is a template constrained by `tl::operand` whose body is one line: `tl::as_expr` on the operands, then a call to a non-template function compiled once in a `.cpp`. `fixed<T, N...>` is the only substantial generic code in public headers.
- **Names.** `snake_case` for types, functions and concepts; `PascalCase` template parameters; no member prefixes; `_options` suffix for option aggregates. Namespaces `tl`, `tl::detail`, and one per optional component (`tl::kokkos`, `tl::serial`, `tl::mpi`, `tl::petsc`, `tl::tpetra`). `TL_` prefix for macros and environment variables, no API macros.
- **Layout.** `include/tl/` public, `include/tl/detail/` internal, `src/<component>/` mirroring it, one file per glossary concept named after it. Users include `<tl/tl.hpp>` plus one header per optional component.
- **Local rules.**
  - Observe by `const&`, take sinks by value and move. Raw pointers never own; smart pointers stay internal. `std::span` or `std::initializer_list` for small host lists.
  - `noexcept` on accessors, moves and `swap` only.
  - `constexpr` for `fixed`, shape, index and promotion arithmetic, tested with `static_assert`.
  - A class only for an invariant; plain aggregates for data (options, `tl::box`, `tl::site`); behaviour in free functions. Public inheritance only for exceptions and the pure abstract registry interfaces.
  - Local variables `const` when they do not change.
  - `[[nodiscard]]` on the types `expr`, `tensor` and `view`, and on a function only when dropping its result is a bug.
- **Enforcement.** clang-format and a curated clang-tidy profile. `-Wall -Wextra -Wpedantic -Wconversion -Wshadow` and `/W4`, errors in CI only. CI checks the header whitelist, the absence of C++23 conditional paths and the one-line facade rule. Every public function has a documentation example that CI compiles and runs on the serial backend. `CONTRIBUTING.md` holds the review checklist: glossary names, one way to write each thing, extensions through the mechanisms of section 12, readable without opening the `.cpp`.

## 16. Testing and benchmarking

From [Test and benchmark infrastructure](issues/32-test-and-benchmark-infrastructure.md).

- **Oracles.** The serial backend for code generation. A test-only Python shim over `<tl/c.h>` runs array-api-tests and NumPy comparisons. Hand-written values for examples and numerical cases.
- **Comparison.** Bit-exact under `TL_FP=strict` for correctly rounded IEEE operations and canonical reductions. Ulp bounds per function and target for transcendentals. A tolerance proportional to contractible operations in the default mode. Reproducibility promises tested bit-exact.
- **Generated tests.** A window generator with shrinking to readable C++ and reproducible seeds. A fixed budget per PR, long nightly runs.
- **Multi-rank.** A threaded test `tl::communicator` with fault injection and sanitizers for most tests. Real MPICH and Open MPI at 1 to 4 ranks per PR, more ranks and GPU-aware MPI nightly.
- **CI.**
  - Per PR on hosted runners: GCC and Clang at floor and latest, Apple Clang, MSVC; serial-only and Kokkos OpenMP with LLVM builds; ASan, UBSan and TSan; the suite under `TL_DEBUG=all`; CUDA, HIP and SYCL compile-only.
  - Self-hosted GPU runners nightly and on the `gpu` label. Periodic campaigns in computing centres.
- **Benchmarks.**
  - Google Benchmark with interleaved repetitions, reported against measured machine ceilings, on the samurai and ponio workloads rewritten with tensorlib.
  - A dedicated physical machine with nightly history, alerts on significant regressions above 5 %, and same-job A/B on the `bench` label.
  - Instruction counts per PR for the front end.
- **Tools.** GoogleTest. Header and facade checks by preprocessing. A compile-time budget on instantiations and front-end time (`-ftime-trace`).
- **Order.** The serial oracle, the comparison rule, GoogleTest and the header check come with the first ticket. The other pieces arrive with the feature they test.

## Reconciliations

Some tickets were amended by later ones. Where the text of a ticket still says otherwise, this spec settles it as follows.

1. **Header and target names.** [Backend interface](issues/10-backend-interface.md) writes `<tensorlib/kokkos.hpp>`. The header is `<tl/kokkos.hpp>`, per the layout of [Coding style and extension points](issues/29-coding-style-and-extension-points.md). CMake targets keep the `tensorlib::` namespace of [Minimum C++ standard and build system](issues/14-cxx-standard-and-build.md).
2. **C++23 in headers.** [Minimum C++ standard and build system](issues/14-cxx-standard-and-build.md) allowed optional C++23 paths behind feature-test macros. Public headers have none ([Coding style and extension points](issues/29-coding-style-and-extension-points.md)).
3. **Per thread becomes per context.** Since [ADR 0012](../../docs/adr/0012-one-c-abi-with-explicit-contexts-and-generic-dispatch.md), what earlier tickets attach to a host thread belongs to a context:
   - the pending window;
   - the scope of `tl::debug_scope`;
   - the "first unreported error" thrown by `tl::wait()` and `tl::eval_all()`;
   - the default queue of [Backend interface](issues/10-backend-interface.md), which becomes one per pair of context and device.

   A C++ host thread's implicit context keeps every earlier behaviour for C++ code.
4. **Indirect scatter-add.** [Views and slicing](issues/23-views-and-slicing.md) leaves "atomics or colouring" to lowering. [Kernel IR](issues/22-kernel-ir.md) fixes atomics by default, a deterministic sort path under `TL_FP=strict` or `tl::deterministic(idx)`, and colouring as a possible internal optimisation.
5. **Interop kernels.** [Backend interface](issues/10-backend-interface.md) made them triggers. They are non-fusible window nodes ([Reconcile the core with samurai and ponio](issues/26-reconcile-with-samurai-ponio.md)).
6. **Umpire.** Optional in [Backend interface](issues/10-backend-interface.md), required by the Kokkos backend since [Buffer reuse and allocation](issues/21-buffer-reuse-and-allocation.md).
7. **Out of memory.** Thrown at the trigger in [Buffer reuse and allocation](issues/21-buffer-reuse-and-allocation.md). It poisons the window's outputs and throws at the read, since [Error reporting and debug mode](issues/24-error-reporting-and-debug-mode.md).
8. **Reproducibility across ranks.** [Code-generation strategy](issues/12-code-generation-strategy.md) promised reproducibility at a fixed rank count. By default it is "in practice". It is guaranteed under `TL_FP=strict` ([Communication in the evaluation plan](issues/28-communication-in-the-evaluation-plan.md)).
9. **Segment offsets.** [Views and slicing](issues/23-views-and-slicing.md) gives one offset table per operand. [Lowering of short segments](issues/30-lowering-of-short-segments.md) shares one base column among operands with the same table and bridges only then.

10. **Host floor.** [Minimum C++ standard and build system](issues/14-cxx-standard-and-build.md) and [Test and benchmark infrastructure](issues/32-test-and-benchmark-infrastructure.md) name GCC 10.4 and Clang 14, Kokkos 5's floor. The amendment on the floor raises the host floor to GCC 11.1, Clang 16 and Apple Clang 16, so that `<source_location>` is available to `tl::site`.

## Open items

These do not block splitting the spec into implementation tickets. Each one sets defaults or confirms a choice made on CPU evidence.

- **Hardware checks**, in [Rerun the fusion benchmark on an HPC node and a GPU](issues/19-rerun-fusion-benchmark-hpc-gpu.md):
  - the fusion benchmark on an x86 node and a GPU;
  - NVPTX and AMDGPU generation, GPU kernel-size ceilings, CUDA or HIP graphs;
  - Umpire pools on small allocations, staggered offsets on GPUs;
  - 32-bit indices, atomic against deterministic scatter-add;
  - x86 ceilings and the 1.9x two-launch gap;
  - `TL_OVERLAP_MIN_CELLS`;
  - segment bridging and masked tails on x86 and GPUs.
- **Fog left on the map.** Communication on the device: a device path for large partial reductions, stream-triggered communication (MPIX streams, NCCL, RCCL, Kokkos Comm), and pack fusion into writers. To weigh once GPU numbers exist.
- **Not measured yet:**
  - 2-D stencils that read neighbouring rows through other intervals;
  - kernels with several bodies on real meshes;
  - Windows and macOS performance.
- **Infrastructure to secure:** self-hosted GPU runners (NVIDIA, AMD if possible), a dedicated benchmark machine, and allocations in computing centres.
- **Defaults to set at the first implementation:** the compile-time budget threshold, the ulp bounds per function and target, and the CI budget on the fixed cost of an evaluation.

## Suggested order of work

A dependency order for the implementation tickets, not a plan. Each step comes with the tests of section 16 that match it. The tickets are listed in [Implementation of the tensorlib core](../core-implementation/spec.md).

1. **Skeleton.** CMake, presets, header layout, CI checks, GoogleTest, the serial backend with a minimal kernel IR interpreter, `fixed`, shapes and promotion as `constexpr`.
2. **Front end.** `tensor`, `view`, `expr`, `any_tensor`, contexts and pending windows, sites, node recording and checks, broadcasting, slicing, copy-on-write.
3. **Planner.** Fusion of a window into kernels, aliasing, buffer plan with the capacity rule, plan cache, kernel-size splitting, errors and poison. All on the serial backend, with the window generator.
4. **Kokkos backend.** Memory resources with Umpire, queues and events, the LLVM generator on CPU, kernel cache and bundles, canonical reductions, segmented kernels with bridging, Kokkos interop.
5. **Distributions and communication.** Distributions, `tl::communicator` (threaded test implementation first, then MPI), ghost exchanges, reductions across ranks, redistribution, progress thread.
6. **Interop and ABI.** PETSc and Tpetra components, the C ABI with its catalogue, DLPack, and the Python test shim.
7. **GPU targets.** NVPTX, AMDGPU and SPIR-V lowering and the device checks of the open items.
