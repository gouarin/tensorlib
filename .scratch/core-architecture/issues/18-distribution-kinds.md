# Distribution kinds

Type: grilling
Status: resolved
Blocked by: 25

## Question

Which distribution kinds does the core describe in v1 (local, replicated, 1D block along one axis, nD block over a process grid, block-cyclic, user-defined or irregular partitions), how is a distribution described and compared (two distributions are compatible when?), how are ghost widths declared per axis, and how do zero-size local parts and uneven splits behave? Use Heat's split axis, Tpetra `Map` and PETSc DMDA as references.

Scope set by [Reconcile the core with samurai and ponio](26-reconcile-with-samurai-ponio.md): regular distributions only. samurai fields are local tensors with samurai's own exchanges, using tensorlib's segmented gather and scatter for packing. Index-list distributions (Tpetra `Map`, PetscSF) are left for later.

## Answer

Decided with the user on 2026-10-30. The vocabulary is in `CONTEXT.md`, and automatic ghost updates are recorded in [ADR 0004](../../../docs/adr/0004-automatic-ghost-updates-in-the-graph.md).

- **Kinds in v1.** The kinds are local, replicated, and block over an n-d Cartesian process grid, where each axis is either split or not. A single split axis is Heat's model, and a fully split grid is PETSc DMDA's.
  - `tl::block(comm, grid{4, 2, 1})`;
  - `tl::block(comm, grid{P}, axis = 0)`;
  - explicit sizes per axis (`sizes{{300, 212}, {256, 256}}`, like DMDA's `lx`, `ly`).

  Block-cyclic comes later, with ScaLAPACK interop. Index-list distributions are left for later ([Reconcile the core with samurai and ponio](26-reconcile-with-samurai-ponio.md)).
- **Description.** A distribution is an immutable, shared object. It holds:
  - the communicator;
  - the process grid;
  - the ownership range of each rank along each axis;
  - the ghost width and periodicity of each axis;
  - the stencil shape.
- **Compatibility.** Two distributions are compatible when they are the same object (fast path), or when their communicators are congruent and their ownership ranges match on the axes being combined. The check is local, with no MPI call per operation. Ghost widths do not count, so a tensor with ghosts combines with the same tensor without them. A mismatch is an error that suggests `redistribute()`. The broadcasting rules are in [Broadcasting design](13-broadcasting-design.md).
- **Ghosts.**
  - Declared at creation: a width per axis, a stencil shape (box exchanges corners, star does not, as in DMDA), and periodicity per axis.
  - Local storage includes the band. `local_part()` sees the owned cells, and `local_part_with_ghosts()` sees owned cells plus ghosts.
  - The band model applies to regular distributions only ([Tensor data model](07-tensor-data-model.md), amended).
- **Automatic ghost updates.** The graph knows which ghost values a kernel reads (shifted index maps reaching past the owned range) and which are stale (written since the last exchange). It inserts the exchange itself. The exchange is collective under the SPMD rules of [ADR 0002](../../../docs/adr/0002-implicit-pending-windows.md), and the interior kernel may run while it is in flight. `u.update_ghosts()` forces an exchange.
- **Uneven splits and empty parts.**
  - Default split as with `PETSC_DECIDE`: with `n = q*p + r`, the first `r` ranks get `q + 1`.
  - Empty local parts are legal everywhere. Their rank launches no kernel, but it still joins collective triggers, and its reductions contribute the identity.
  - A ghost width larger than a neighbour's part, including an empty one, is an error raised when the distribution is created.
- **Left to the fog "distribution hooks in the graph":** how exchanges overlap with computation, how several exchanges are aggregated into one message, and how cross-process reductions enter the graph.
