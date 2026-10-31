# Implicit pending windows, with expressions distinct from tensors

Tensor operations are deferred into a pending window, one per host thread, and a trigger evaluates every live output of that window in fused kernels. Triggers are a host read or export, an explicit `tl::eval`, a non-fusible consumer, or a deterministic resource limit. This makes the reference workload `c = a + b; d = a + 2*b` run as one loop without the user doing anything. We rejected explicit-only evaluation, because users forget it and must group outputs by hand. We also rejected `jit`/`compile`-style regions, which force users to restructure their code into compiled functions. Writing `auto t = a * b` gives a `tl::expr`, never stored, while assigning to a `tl::tensor` asks for storage. Naming a subexpression for readability therefore costs nothing, as it does in xtensor.

## Consequences

- In distributed runs, any trigger that involves a partitioned tensor is collective (SPMD). Resource limits must be deterministic across ranks.
- Results must match immediate execution, so the library inserts copies or splits kernels when an output is read through a different index map.
- Evaluation is asynchronous, so errors raised while kernels run surface at the next wait, not at the line that built the expression.

Amended by [ADR 0012](0012-one-c-abi-with-explicit-contexts-and-generic-dispatch.md): the pending window belongs to a context, and each host thread has an implicit one.
