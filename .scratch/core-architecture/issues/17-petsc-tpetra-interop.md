# Zero-copy interop with PETSc and Tpetra

Type: grilling
Status: resolved
Blocked by: 07, 18

## Question

How does a partitioned tensor hand its local parts to PETSc (`Vec`, `Mat`, DMDA) and Trilinos/Tpetra (`MultiVector`, `CrsMatrix`) without copies, and how does it wrap their data in return? Decide ownership and lifetime across the boundary, how the partition maps onto PETSc layouts and Tpetra `Map`, how ghost and halo regions line up with DMDA, how pending lazy work is flushed before handing data over, and which communicator is shared.

Requirement from [Reconcile the core with samurai and ponio](26-reconcile-with-samurai-ponio.md): hand out an evaluated local buffer the way samurai uses `VecCreateSeqWithArray` today.

## Answer

Decided with the user on 2026-10-30. The API facts behind it are in [PETSc and Tpetra APIs for zero-copy interop](../research/17-petsc-tpetra-apis.md). The terms Borrow and Wrap are defined in `CONTEXT.md`.

- **Scope of v1.** Both directions for `Vec` (Seq and MPI), DMDA vectors, and Tpetra `MultiVector` and `Vector`. PETSc `Mat` and Tpetra `CrsMatrix` stay opaque operators applied to tensors. tensorlib does not build matrices.
- **Borrow** (tensorlib owns the data, the other library sees it). `tl::petsc::borrow(t, access)` and `tl::tpetra::borrow(t, access)` are scoped (RAII).
  - Start of the borrow: the pending window is evaluated, the host waits for the last write to `t`, the buffer is made unique (copy-on-write) for a write access, and `t` is locked.
  - While borrowed: using `t` in a tensorlib expression is an error for a write borrow, and writing to `t` is an error for a read borrow. Debug mode checks both.
  - End of the borrow: the `Vec` is destroyed, and a reference still held by PETSc is an error. The graph takes `t` back and marks its ghosts stale after a write.
  - The alias-forever model (`to_vec` plus manual `sync`/`mark_modified`) was rejected: like explicit ghost updates in [ADR 0004](../../../docs/adr/0004-automatic-ghost-updates-in-the-graph.md), a forgotten call gives wrong results without an error. It can be added later without breaking borrows.
- **Wrap** (the other library owns the data, tensorlib sees it). `tl::petsc::wrap(v, access)` and `tl::tpetra::wrap(mv, access)` are scoped too, and mirror the borrow. They use `VecGetArray*AndMemType` and `getLocalViewDevice`.
  - The result is an ordinary tensor over borrowed storage, with the distribution derived from the object.
  - End of scope: tensorlib evaluates what reads or writes the buffer, waits, deep-copies any tensor copies still sharing it, then restores the array.
  - A memory type that matches no memory space of the backend is an error.
- **Ghost band.** The band storage of [Distribution kinds](18-distribution-kinds.md) stays. The owned cells of a ghosted tensor are not contiguous, so the cases are:
  - zero copy: a tensor without ghosts to a global `Vec` or a `MultiVector`, and a ghosted tensor to the DMDA local vector (`borrow_local`), whose layout is identical;
  - explicit copy: a ghosted tensor to a global vector goes through `borrow_packed`, a fused packing kernel on the device into a pool buffer, copied back at the end for a write access;
  - a plain `borrow` of a ghosted tensor as a global vector is an error that suggests `borrow_packed`. The copy costs as much as `DMGlobalToLocal`, which DMDA codes already pay.
- **Who describes the split.** Both directions.
  - `tl::petsc::dmda(d)` builds a DMDA from a block distribution and caches it on the immutable distribution object.
  - `tl::petsc::distribution(da)` reads `DMDAGetInfo` and `DMDAGetOwnershipRanges` and keeps a reference to the DMDA, so a wrap followed by a borrow gives back the same `DM`.
  - Axis mapping: tensor axis 0 is `k` (or `j` in 2-D), the last spatial axis is `i`, and a trailing component axis or `tensor<vec3>` is `dof`. Process grid, `lx/ly/lz`, stencil width and shape, and periodicity map one to one.
  - A plain MPI `Vec` and a contiguous `Tpetra::Map` give a 1-D block with explicit sizes. A non-contiguous Map is an error, since index-list distributions are deferred ([Reconcile the core with samurai and ponio](26-reconcile-with-samurai-ponio.md)). A block distribution exports as a contiguous Map numbered by rank.
  - The distribution's communicator is passed as is. PETSc duplicates it internally, and `Teuchos::MpiComm` wraps it.
  - A mismatch between the two descriptions is an error raised at borrow time, naming the axis and the rank.
- **Synchronisation on devices.**
  - The contract is full synchronisation: the host waits for the last tensorlib write before handing data over, then for the PETSc stream (`PetscDeviceContextSynchronize`) or a Kokkos fence (Tpetra) before taking it back. PETSc has no API to accept a foreign stream.
  - The opt-in `stream_ordered` mode orders the two streams with native events (`cudaStreamWaitEvent`) without blocking the host. It is available when the backend declares native events, and tests compare both paths.
  - tensorlib, PETSc built with Kokkos, and Trilinos must share one Kokkos installation. This goes into the Spack and conda-forge recipes of [Minimum C++ standard and build system](14-cxx-standard-and-build.md).
- **Operators in the pending window.**
  - `tl::petsc::mult`, `mult_add` and `mult_transpose`, and `tl::tpetra::apply`, record a non-fusible node that reads `x` and produces a pool-managed tensor with the row distribution of the operator, like the interop kernels of [Reconcile the core with samurai and ponio](26-reconcile-with-samurai-ponio.md). The node borrows its operands itself at evaluation.
  - `r = b - tl::petsc::mult(A, x)` is one evaluation, with the fusible parts on either side.
  - Column and row distributions are checked when the node is built.
  - Solvers (`KSPSolve`, `SNESSolve`, `TSSolve`) stay inside explicit borrows. They have state and convergence diagnostics, and a lazy failure would surface far from its cause.
- **Consequences.**
  - Borrows and wraps of partitioned tensors are collective, because they trigger evaluation under the SPMD rules of [ADR 0002](../../../docs/adr/0002-implicit-pending-windows.md).
  - A Tpetra `MultiVector` is column-major (`LayoutLeft`). A 2-D tensor `(N, k)` borrows without copy when its local layout is column-major, otherwise the borrow is an error that suggests `borrow_packed`.
  - tensorlib builds the Tpetra `DualView` itself from unmanaged views. The host side comes from its pool and is allocated only when Tpetra asks for host access, which avoids the host mirror that the `t_dev` constructor always allocates.
  - The interop ships as optional components `tensorlib::petsc` and `tensorlib::tpetra`, with headers outside the core whitelist and dependencies found by `find_package`. The core never depends on PETSc or Trilinos.
- **Requirements passed on:**
  - to [Buffer reuse and allocation](21-buffer-reuse-and-allocation.md): packing buffers for `borrow_packed`, operator outputs and Tpetra host sides come from the pool;
  - to [Kernel IR](22-kernel-ir.md): a non-fusible external-operator node that declares its reads and writes;
  - to [Error reporting and debug mode](24-error-reporting-and-debug-mode.md): lock violations during a borrow, references left at the end of a borrow, and split mismatches naming the axis and rank.

## Amendment (2026-10-30)

[Buffer reuse and allocation](21-buffer-reuse-and-allocation.md): an assignment that changes the size of a wrapped tensor would move it out of the PETSc or Tpetra storage, so it is an error. The buffers listed above come from the memory resource of their space.

## Amendment (2026-10-30)

[Error reporting and debug mode](24-error-reporting-and-debug-mode.md): lock violations during a borrow are always checked and thrown at once, not only in debug mode. A reference still held by PETSc or Tpetra when a borrow ends cannot throw from the destructor: in release the buffer stays alive through a `PetscContainer` composed on the `Vec`, the tensor detaches on its next write and a warning is written; with checks on, the process terminates after the report.
