# C ABI and binding surface

Type: grilling
Status: resolved
Blocked by: 29

## Question

[Binding surfaces for a lazy tensor library](06-binding-surfaces.md) settled on a small versioned C ABI over an opaque, refcounted and possibly unevaluated dynamic-rank handle, with DLPack at the edge. Implementing the Python, Julia and Rust bindings is out of scope, but the surface they call belongs in the spec. Decide its shape:

- the handle model: one erased tensor handle, or separate handles for views and expressions; refcounting and thread rules; how a handle maps to the pending window of the calling host thread;
- the function set: one entry point per operation, or operations by name with a generic `tl_apply`; how the set grows without breaking the ABI; versioning and feature queries;
- element types, memory spaces and distributions as run-time values, and how the extension points of [Coding style and extension points](29-coding-style-and-extension-points.md) appear at the ABI;
- views and slicing ([Views and slicing](23-views-and-slicing.md)) through the ABI, and whether `a[idx] += v` in Python accumulates duplicates as in C++ ([ADR 0006](../../../docs/adr/0006-views-are-references-distinct-from-tensors.md)) or follows NumPy;
- traced functions and reductions with user combiners called from a binding language;
- errors and poison ([ADR 0007](../../../docs/adr/0007-errors-poison-results-across-ranks.md)) as status codes, and DLPack import and export for evaluated, partitioned and device tensors.

## Answer

Decided with the user on 2026-10-31. The overall shape is recorded in [ADR 0012](../../../docs/adr/0012-one-c-abi-with-explicit-contexts-and-generic-dispatch.md). Context is in `CONTEXT.md`, and Pending window now belongs to a context.

- **One surface.** Python, Julia and Rust all go through the C ABI. A Python binding may use nanobind or Cython to wrap it, never to bind the C++ types. The ABI is the optional component `c`: header `<tl/c.h>` in C11, `tl_` prefix, CMake target `tensorlib::c`, implementation in `src/c/` calling the public C++ API, no MPI in the header. Its tests are written in C and run on the serial backend in CI.
- **Handle.** One opaque, refcounted `tl_tensor*`, with a kind (value or view) that can be queried. Every operation accepts either kind as an operand and returns a value, possibly not yet evaluated. A view handle holds a reference on its owner and writes into it. Expressions do not exist at the ABI: `t = a * b` is a pending value, fused at the trigger and stored only if it is still alive then, like a named `tl::tensor` in C++.
- **Context.** `tl_context*` is passed to every call that records or triggers. It owns the pending window and the last error. It is used from one thread at a time and may move between threads (task migration in Julia and async Rust). In C++, each host thread has an implicit context. `tl_retain`/`tl_release` are atomic and safe from any thread. A release sets an atomic "dead output" flag, read by the window at evaluation, and never locks the window. A handle's shape and dtype are immutable. Two threads that write the same tensor without synchronisation race, as in C++. Under SPMD, collective triggers concern matching contexts in the same order on every rank.
- **Function set.**
  - A fixed core of explicit functions: lifetime, metadata, creation from data, views and slicing, assignment, `tl_eval`/`tl_eval_many`, DLPack, errors, version.
  - All array operations, built in or registered, go through `tl_apply(ctx, op, inputs, attrs, outputs)` on a `tl_op*` looked up once with `tl_op_find`. Options are typed attributes.
  - A queryable catalogue (`tl_op_count`, `tl_op_info`) gives each operation's name, arity and attribute schema. It is generated from the C++ registry filled by the compiled bodies of [ADR 0011](../../../docs/adr/0011-thin-templates-over-an-erased-core-and-layered-extension-points.md), with Array API names.
  - `tl_abi_version()` is major.minor, and the ABI grows only by addition. Optional parts are detected with `tl_has_feature` or by an operation missing from the catalogue.
- **Element types.** A structured `tl_dtype`: kind, bits, component rank (no cap) and component extents. `fixed<double,3,3>` is `{float, 64, 2, {3,3}}`. `fixed` tensors can be created from a binding. The logical shape excludes components, and a core function gives the component view. User element types from `tl::element_traits` appear as `{opaque, registered id}` with a name and a size: a binding can receive them, pass them to operations and return them, but cannot create or export them. DLPack always exports the component view.
- **Run-time attributes.**
  - The memory space is an enumeration (`host`, `device`, `pinned`, `managed`) plus a device index.
  - Row-major and column-major layouts are both exposed, strides can be queried, and creation accepts both orders.
  - `tl_communicator*` is created from the Fortran handle (`tl_communicator_from_fortran`), or from a native `MPI_Comm` when the binding uses the same MPI. `tl_communicator_self()` serves runs without MPI.
  - `tl_distribution*` is opaque and refcounted, built by core functions (`local`, `replicated`, `block` with grid, sizes, ghosts and periodicity). Queries give the global shape, the local part and its offsets.
  - `tl_local_part` returns a view, with a variant that includes the ghost band. Writing ghosts marks them modified, as in C++.
- **Extensions from a binding.**
  - `tl_trace(ctx, callback, user_data, signature, &op)` runs the binding's callback once, on the host, on placeholder handles. The resulting `tl_op` joins the catalogue and also serves as a reduction combiner.
  - External functions are registered with a host symbol (required) and optional LLVM bitcode per target, as in C++. A host symbol from a binding must be pure, thread-safe and free of any interpreter lock (for example `numba.cfunc(nogil=True)` or Julia `@cfunction`). This is a documented rule that the ABI cannot check.
  - Backends and memory resources are chosen by name (`tl_context_set_backend`) and registered in C++ only.
- **Views and assignment.**
  - `tl_view(ctx, t, specs, n, &out)` takes specs that are an integer, a range, `newaxis`, an ellipsis, an index tensor or a mask. `tl_assign(ctx, target, value, op)` covers `=` and compound assignment.
  - The ABI carries the C++ semantics of [Views and slicing](23-views-and-slicing.md): out-of-bounds ranges are errors, masks are write-only, and `a[idx] += v` accumulates duplicates in every language (the equivalent of `np.add.at`; the Array API leaves it unspecified).
  - A binding may clip ranges itself, or turn a mask read into `compress`. It must never rewrite a compound indexed assignment into a read followed by a write, which would race on duplicates.
- **Errors.**
  - Every fallible call returns a `tl_status` that only grows: `TL_OK`, `TL_SHAPE_ERROR`, `TL_INVALID_ARGUMENT`, `TL_UNSUPPORTED_OPERATION`, `TL_OUT_OF_MEMORY`, `TL_POISONED`, `TL_DEVICE_ERROR`, `TL_SPMD_ERROR`, `TL_REMOTE_ERROR`, `TL_INTERNAL`.
  - The detail (message, site, origin rank, report) lives in the context and is read with `tl_context_error(ctx)`, valid until the next call on that context. Functions without a context fail only on programming errors and return a static message.
  - Bindings supply sites with `tl_context_set_site(ctx, file, line, function)`. Without a site, kernels are named after their operations.
  - Device errors surface at the next trigger of the same context. Poison gives the same `TL_POISONED` and report on every rank at the same read ([ADR 0007](../../../docs/adr/0007-errors-poison-results-across-ranks.md)).
- **DLPack.**
  - Export is a trigger and is read-only (DLPack 1.0 `READ_ONLY` flag). The deleter holds a reference on the storage, so a later tensorlib write detaches by copy-on-write. To write from another tool, `tl_borrow_begin`/`tl_borrow_end` evaluate, detach to a unique buffer and wait, then give a writable DLPack. Any tensorlib access to the tensor during the borrow is an error. This is the `borrow` of [Zero-copy interop with PETSc and Tpetra](17-petsc-tpetra-interop.md).
  - With `stream`, the consumer's queue waits on our event. Without it, the export synchronises fully.
  - Import is zero-copy for a dense row- or column-major buffer on the context's device, and the tensor keeps the deleter. Other strides are copied into a dense tensor. A `READ_ONLY` buffer is copied at the first tensorlib write. The producer must not write while pending operations read the buffer. A device foreign to the context is an error.
  - A partitioned tensor is never gathered implicitly: export `tl_local_part`, or call an explicit collective operation such as `gather` first.
- **Amendments.** [When the task graph is evaluated](08-evaluation-trigger.md): the pending window belongs to a context, and each C++ host thread has an implicit one. [Error reporting and debug mode](24-error-reporting-and-debug-mode.md): the error detail lives in the context.

Grilling log (2026-10-31):

- Q1, handle model: option A. One opaque refcounted handle with a value/view kind; no expressions at the ABI.
- Q2, windows and threads: option B. Explicit `tl_context*` owning the window, movable between threads; implicit per-thread context in C++; atomic refcount, release sets a dead-output flag.
- Q3, function set: option C. Explicit core for lifetime, metadata, views, assignment, evaluation, DLPack, errors and version; `tl_apply` with a generated, queryable catalogue for every array operation; major.minor version and feature queries.
- Q4, element types: option B. Structured `tl_dtype` with components (rank at most 4), `fixed` creatable from bindings, user types opaque. Amended 2026-11-01: the user removed the cap on the component rank, in C++ and in the ABI.
- Q5, spaces, layouts, distributions: option A. Memory-space enumeration plus device, row- and column-major layouts, `tl_communicator*` from the Fortran handle, opaque `tl_distribution*`, `tl_local_part` returns a view.
- Q6, extensions from bindings: option A. Traced functions and combiners, external functions with a pure host symbol free of interpreter locks and optional bitcode; backends and resources chosen by name, registered in C++.
- Q7, views and `a[idx] += v`: option A. The ABI carries the C++ semantics (accumulation); bindings may clip ranges or turn mask reads into `compress`; rewriting a compound indexed assignment into read then write is forbidden.
- Q8, errors: option B. `tl_status` enumeration that only grows, detail stored in the context, sites through `tl_context_set_site`.
- Q9, DLPack: option A. Read-only export with copy-on-write, writable access only in a scoped borrow, zero-copy import when dense and on the context's device, no implicit gather.
- Q10, one surface: option A. Every binding goes through the C ABI. The user first preferred nanobind wherever possible, and the follow-up offered nanobind over an erased C++ core with the C ABI as its mechanical translation. The user kept a single C ABI because it serves all three languages at once; nanobind may still wrap it for Python.
