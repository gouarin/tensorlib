# Error reporting and debug mode

Type: grilling
Status: resolved
Blocked by: 14

## Question

With deferred, asynchronous evaluation ([ADR 0002](../../../docs/adr/0002-implicit-pending-windows.md)) and typed exceptions ([Minimum C++ standard and build system](14-cxx-standard-and-build.md)), how does a user find the line that caused an error raised while a kernel runs (NaN checks, out-of-bounds in a user Kokkos kernel, device faults)? Decide whether graph nodes record their source location (`std::source_location`) and at what cost, what the next wait reports (which kernel, which statements were fused into it), and what an immediate debug mode does: evaluate statement by statement, on the serial backend or on the real one, with bounds and NaN checks, and how it is switched on (environment variable, scope, build option). Distributed runs need the same answer on every rank.

Requirements from [Zero-copy interop with PETSc and Tpetra](17-petsc-tpetra-interop.md): report using a tensor locked by a borrow, a reference still held by PETSc or Tpetra when a borrow ends, and a mismatch between a distribution and a DMDA or Map, naming the axis and the rank.

Requirements from [Buffer reuse and allocation](21-buffer-reuse-and-allocation.md): an out-of-memory error on one rank of a distributed run (the other ranks would block at the next collective), filling storage with NaN after `resize` in debug mode to catch reads of indeterminate values, and reporting a tensor still alive at `tl::finalize`.

Requirements from [Views and slicing](23-views-and-slicing.md): list copy-on-write detaches caused by partial writes, by source line; report duplicate indices in a plain scatter `u(idx) = e`; count element reads of partitioned tensors per source line, to find host loops that broadcast one value per iteration.

## Answer

Decided with the user on 2026-10-30. Vocabulary (site, check, poisoned tensor) is in `CONTEXT.md`. How errors travel across ranks is recorded in [ADR 0007](../../../docs/adr/0007-errors-poison-results-across-ranks.md).

- **Sites.** Every statement records a `tl::site` (file, line, function), in release and debug alike: outputs of a window, `tl::eval`, triggers. Intermediate nodes take the site of the statement that consumes them. `std::source_location` is one pointer on GCC, Clang and MSVC, negligible next to the ~5 ns per node measured in [Prototype of multi-output fusion](09-multi-output-fusion-prototype.md). Overloaded operators cannot take a defaulted `source_location`, so capture happens in the few real functions and constructors (tensor construction from an expression, `assign`, `tl::eval`, `tl::apply`, free functions). Per-operator capture through a converting operand type was rejected: it adds a conversion to every operand, worse overload diagnostics and compile time. The C ABI takes a `tl::site` from the caller, so Python, Julia and Rust report the line of the user's script.
- **Switching checks on.** `TL_DEBUG` sets the default at `tl::initialize`, and `tl::debug_scope guard{...};` (RAII, per host thread) changes it inside a region, for example one flux scheme from a checkpoint, without rebuilding the application. Entering or leaving a scope is a trigger, collective under SPMD, so one window never mixes modes. The CMake option `TL_ENABLE_ASSERTIONS` covers only host-side assertions in headers (bounds of `a(i, j)`), like `_GLIBCXX_ASSERTIONS`, and does not change the ABI.
- **Checks.** They combine: `TL_DEBUG=nan,bounds` or `tl::debug_scope guard{tl::check::nan | tl::check::bounds};`, with `all` as a shortcut. Kernels are generated at run time, so the lowering inserts the checks without a rebuild.
  - `sync`: each statement is its own trigger followed by a wait, so the error comes out at its line with the user's call stack. Fusion stays inside a statement, and each node keeps its operation, so a report reads "line 42, operation `sqrt`, element (i, j), rank 17".
  - `bounds`: indices of gathers and scatters, boxes and shifts past the declared ghosts.
  - `nan`: an operation that produces NaN or Inf from finite inputs, like an IEEE trap. Propagated NaN is not reported, so code that uses NaN as a missing value gets no false positive, and the report points at the origin.
  - `uninit`: storage from `resize`, `tl::empty` and reused buffers is filled with NaN (a pattern for integers).
  - `scatter`: duplicate indices in a plain scatter `u(idx) = e` (`+=` accumulates and is not checked).
  - `leaks`, `detaches`, `reads`: see below.

  Except for `sync`, checks run inside fused kernels and keep fusion; the report then names the failing operation and the sites fused into the kernel. Checks run on the real backend, so GPU-only bugs stay reproducible; the serial interpreter is a separate backend choice (`TL_BACKEND=serial`), not part of debug mode. Inside a user Kokkos kernel tensorlib sees nothing: bounds there come from Kokkos' own `Kokkos_ENABLE_DEBUG_BOUNDS_CHECK`, and `nan` checks the kernel's outputs after it runs.
- **Kernel errors in release.** A failed kernel poisons the tensors it was writing, and the poison passes to everything computed from them, like an exception in a `std::future`. Reading a poisoned tensor throws; reading an unrelated tensor does not. `tl::wait()` and `tl::eval_all()` throw the first unreported error of the thread, and errors never read are reported at `tl::finalize`. Sticky device errors (illegal address under CUDA or HIP) have `fatal() == true` and poison everything, so later calls throw the same error instead of an obscure backend message. A report gives the error type, the backend message, the kernel, the sites fused into it, the site of the trigger that launched it and the rank.
- **Distributed runs.** Evaluation never throws. A poisoned rank keeps taking part in collectives: ghost messages carry a poison flag in their header and reductions carry a status word, so poison follows the data to other ranks. Every rank throws at the same collective read: the original error where it started, `tl::remote_error` (origin rank, error kind, site, `origin()`) elsewhere. An agreement `allreduce` at every collective trigger was rejected for its latency on every evaluation. A crashed process is left to the MPI launcher.
- **Out of memory.** It poisons the outputs instead of throwing at the trigger, so one rank's OOM does not block the others. This amends [Buffer reuse and allocation](21-buffer-reuse-and-allocation.md).
- **Usage errors on the host** are always checked and thrown at once with the site: using a tensor locked by a borrow, writing to a tensor borrowed for reading, a distribution that does not match a DMDA or a Map (naming the axis and the rank), incompatible shapes. Each costs a flag test or a few integer comparisons. This amends "debug mode checks both" in [Zero-copy interop with PETSc and Tpetra](17-petsc-tpetra-interop.md).
- **Where throwing is impossible.**
  - At the end of a borrow, in a `noexcept` destructor, PETSc may still hold the `Vec`. In release the buffer is kept alive by a `PetscContainer` composed on the `Vec`, the tensor detaches on its next write, a warning names the borrow's site and a counter is reported at `tl::finalize`. With checks on, the report is written and the process calls `std::terminate`, as a joinable `std::thread` does.
  - At `tl::finalize`, live tensors are listed on stderr with their sites and the rank, then pools are destroyed anyway. The `leaks` check turns this into a failing exit code, for CI.
- **Performance diagnostics.** `detaches` counts copy-on-write detaches per site (count and bytes copied), `reads` counts element reads of partitioned tensors per site. Both keep fusion. A table aggregated over ranks (min and max) is written at `tl::finalize`, which is collective, and `tl::diagnostics()` returns the same counters for tests. Kernels are always named after their sites (`tl:flux.cpp:40-42`) and evaluations and inserted ghost exchanges are Kokkos Tools regions, so Nsight Systems, rocprof and Kokkos profilers show user lines; the name is built once, when the kernel enters the cache. This is also how a user sees where the graph inserted ghost exchanges ([ADR 0004](../../../docs/adr/0004-automatic-ghost-updates-in-the-graph.md)).
- **C ABI.** Poison, `fatal()` and `tl::remote_error` map to status codes with the report; bindings turn them into native exceptions.
- **Requirements passed on:**
  - to [Kernel IR](22-kernel-ir.md): nodes carry their operation and a site index, the lowering can insert checks per node, and kernels carry the list of fused sites;
  - to the MPI backend hooks: a poison flag in the ghost-exchange header and a status word in reductions.

Grilling log (2026-10-30):

- Q1 source locations: option A. Every statement records a tl::site, always; intermediate nodes inherit it; capture in a handful of entry points, not per operator; bindings supply the site.
- Q2 switching on: option C. TL_DEBUG default, tl::debug_scope per host thread (collective trigger at entry and exit), TL_ENABLE_ASSERTIONS for host header assertions only.
- Q3 content: option B. Combinable checks (sync, bounds, nan from finite inputs, uninit, scatter, all) on the real backend; serial backend is an orthogonal choice.
- Q4 release errors: option B. Poisoned outputs with propagation, fatal() for sticky errors, unread errors reported at tl::finalize.
- Q5 distributed: option B. Poison crosses ranks with ghost messages and reduction status; all ranks throw at the same collective read; tl::remote_error elsewhere; OOM poisons instead of throwing at the trigger.
- Q6 host usage errors: recommendation. Always checked and thrown at once; end of borrow with a live reference: keep alive and warn in release, terminate with checks on; live tensors at finalize listed, leaks check fails the process.
- Q7 diagnostics: recommendation. detaches and reads as checks in the same mechanism, report at tl::finalize and tl::diagnostics(); kernels named after their sites and Kokkos Tools regions always on.

## Comments

2026-10-31: [Communication in the evaluation plan](28-communication-in-the-evaluation-plan.md) adds a `collectives` check. It compares the hash of the communication plan across ranks before launching, and checks that every rank received the same bits from each reduction across ranks.

2026-10-31: amended by [C ABI and binding surface](31-c-abi-and-binding-surface.md): at the C ABI, the error detail lives in the context (`tl_context_error`), and bindings supply sites with `tl_context_set_site`.
