# Automatic ghost updates in the graph

For regular partitioned tensors, the graph decides when ghost values need an exchange. It knows which ghost cells a kernel reads, through shifted index maps that reach past the owned range, and whether they are stale because the tensor was written since the last exchange. It then inserts the exchange as a collective step, which may overlap with interior computation. `update_ghosts()` remains available to force one.

PETSc users expect explicit `DMGlobalToLocal` calls. We rejected explicit-only updates because they are easy to forget or to repeat. A forgotten update gives wrong results with no error, and a repeated one wastes a message. They also hide from the library the information it needs to overlap communication with computation. We rejected "explicit with detection" because it keeps the boilerplate without enabling overlap.

## Consequences

- Every rank must build the same graph, so that inserted exchanges match across ranks (SPMD rules of ADR 0002).
- The graph tracks a ghost-validity state per tensor, and the kernel IR must expose which ghost cells each kernel reads.
- Debugging needs a way to see where exchanges were inserted (ticket "Error reporting and debug mode").
- This applies to regular distributions only. Codes with their own meshes, such as samurai, keep their own exchanges and use tensorlib's segmented gather and scatter.
