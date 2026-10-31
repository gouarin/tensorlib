# Kernel errors poison results, and the poison crosses ranks with the data

When a kernel fails, or an evaluation runs out of memory, evaluation does not throw. It marks the tensors the kernel was writing as poisoned and stores the error in them. Every result computed from a poisoned tensor is poisoned too, like an exception stored in a `std::future`. The error is thrown when a poisoned tensor is read, or by `tl::wait()` and `tl::eval_all()`. In a distributed run, a poisoned rank keeps taking part in collectives: its ghost messages carry a poison flag and reductions carry a status word. Every rank therefore throws at the same collective read. The rank where the error started throws the original error; the others throw `tl::remote_error`, which names that rank, the error kind and the source site.

We rejected throwing at the next wait, whatever it reads. Under asynchronous evaluation ([ADR 0002](0002-implicit-pending-windows.md)) that wait can be in unrelated code, and a caller who catches the error cannot tell which tensors are still valid. We rejected an explicit agreement at every collective trigger (an `allreduce` of the status on each evaluation of a partitioned window), because it adds a global latency to every evaluation, stencil-only windows included, and distributed performance comes first. We also rejected throwing on the failing rank alone: the other ranks would block at the next collective until the job times out.

## Consequences

- An out-of-memory error during evaluation poisons the outputs instead of throwing at the trigger, which amends [Buffer reuse and allocation](../../.scratch/core-architecture/issues/21-buffer-reuse-and-allocation.md).
- After a recoverable error, every tensor that is not poisoned is correct, so `try`/`catch` around a time step is meaningful, on one rank or many.
- Sticky device errors (an illegal address under CUDA or HIP) have `fatal() == true` and poison everything; the rank still sends its flags, so the other ranks throw too. A crashed process is left to the MPI launcher.
- The MPI backend owes two hooks: a poison flag in the ghost-exchange header and a status word in reductions. An error never read is reported at `tl::finalize`.
- Tensors carry one error pointer per buffer, read only at waits.
