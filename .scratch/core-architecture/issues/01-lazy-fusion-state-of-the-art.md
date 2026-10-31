# State of the art of lazy, fused tensor libraries

Type: research
Status: resolved

## Question

How do existing libraries detect fusion opportunities, including fusion across several outputs that share inputs (`c = a + b; d = a + 2*b`), and when do they trigger evaluation? Cover at least xtensor, Eigen, Blaze, Legate/cuNumeric, JAX/XLA, Dask, Halide, MLX, ArrayFire, Taskflow and HPX. For each one: graph or template based, trigger model, fusion scope, cost of graph construction, behaviour on GPU and in distributed runs.

## Answer

- Template libraries (xtensor, Eigen, Blaze) fuse only within one assignment statement; none fuses `c = a + b; d = a + 2*b`, and none has a multi-assignment API.
- Run-time graph libraries see across statements up to a trigger: explicit eval (`af::eval`, `mx.eval`, `dask.compute`), host access, a non-fusible consumer, resource heuristics (ArrayFire: tree height, kernel parameter size, memory pressure), or a delimited region (`jax.jit`, `mx.compile`, Halide pipeline, Legate trace, Diffuse window).
- Sibling multi-output fusion exists in XLA (automatic, cost model + reachability check), MLX `compile` (element-wise subgraphs, multi-output `Compiled` primitive), ArrayFire `af::eval(c, d)` (same dims only) and Halide `compute_with` (manual). Dask shares input tasks but emits no fused kernel; Taskflow and HPX do no array fusion.
- JIT cost is amortised by caching on a hash of the graph signature. In distributed runs, XLA partitions then fuses per device; Diffuse (research, on Legate) fuses windows of partitioned tasks and finds that kernel fusion plus temporary removal, not task fusion alone, gives the speedup, with halos limiting fusion.

Findings: [research/01-lazy-fusion-state-of-the-art.md](../research/01-lazy-fusion-state-of-the-art.md)
