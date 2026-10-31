# When the task graph is evaluated

Type: grilling
Status: resolved
Blocked by: 01, 16

## Question

When does deferred work run: explicitly, on read, at the end of a scope, or another rule? Decide how users see it, how aliasing and in-place updates such as `a += transpose(a)` behave, and how the trigger allows `c = a + b; d = a + 2*b` to run as one loop.

## Answer

Decided with the user on 2026-10-30, from [State of the art of lazy, fused tensor libraries](01-lazy-fusion-state-of-the-art.md). Vocabulary is in `CONTEXT.md`. The model is recorded in [ADR 0002](../../../docs/adr/0002-implicit-pending-windows.md).

- **Pending window.** Operations do not run when written. They accumulate in a pending window, one per host thread. A trigger evaluates every live output of the window together, so `c = a + b; d = a + 2*b` becomes one loop with nothing extra to write. `tl::eval(c, d)` and `tl::eval_all()` force evaluation explicitly.
- **Triggers:**
  - host reads and exports: element access, `data()`, DLPack export, handing data to PETSc or Tpetra, `to(space)`, a scalar used in control flow;
  - explicit calls: `tl::eval(...)`, `tl::eval_all()`;
  - a consumer that cannot be fused (Kokkos Kernels gemm, FFT, sort, solver), which evaluates its inputs first;
  - deterministic resource limits: number of nodes, planned bytes of temporaries. Never local memory pressure, which differs between ranks.
- **Distributed runs follow SPMD rules.** Every rank builds the same graph in the same order, and any trigger that involves a partitioned tensor is collective: all ranks of its communicator must reach it, as with an MPI collective. Purely local work goes through `local_part()` and a local evaluation.
- **Aliasing.** Results are always those of immediate, statement-by-statement execution with value semantics. When an output is also read through a different index map (transpose, shift, broadcast), the library inserts a copy or splits the kernel. An identical index map stays fused in place. There is no `noalias` escape hatch.
- **Expressions and tensors.** `auto t = a * b` gives an expression (`tl::expr<T>`): a graph node that is never stored and is inlined into each consumer. Inside one window, fusion computes it once. An expression captures the values of its inputs through copy-on-write, so reading it after `a` changes recomputes the old `a * b`. `tl::tensor<T> t = a * b` declares a tensor, written when the window is evaluated.
- **Asynchrony.** A trigger compiles and launches kernels on the backend stream or queue, then returns. The program waits only on a host read, an export, or `tl::wait(...)`.

## Comments

2026-10-31: amended by [C ABI and binding surface](31-c-abi-and-binding-surface.md): the pending window belongs to a context (ADR 0012); each C++ host thread has an implicit one.
