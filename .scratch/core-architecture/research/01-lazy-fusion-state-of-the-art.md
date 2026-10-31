# State of the art of lazy, fused tensor libraries

Research for ticket `issues/01-lazy-fusion-state-of-the-art.md`. Written 2026-10-30.

Question: how do existing libraries detect fusion opportunities, including fusion across several outputs that share inputs (`c = a + b; d = a + 2*b`), and when do they trigger evaluation? For each library: graph or template based, trigger model, fusion scope, cost of graph construction, behaviour on GPU and in distributed runs.

Sources are primary (official docs in the project repos, source code, papers). Claims marked "(inference)" are my reading, not a statement from the source. Claims marked "(not re-checked)" come from prior knowledge and could not be fetched in this session (bitbucket.org, gitlab.com, arrayfire.org and the MLX doc site were blocked by the sandbox firewall, so the same documents were read from GitHub mirrors or repos).

## Summary table

| Library | Family | Trigger | Fusion scope | Multi-output fusion (`c`, `d` sharing `a`, `b`) | Graph cost | GPU | Distributed |
|---|---|---|---|---|---|---|---|
| xtensor | C++ expression templates | assignment to a container, element access, `xt::eval` | one assignment statement | no, `a` and `b` read twice | zero at run time, compile time only | no | no |
| Eigen (core + Tensor) | C++ expression templates | assignment, `.eval()` | one assignment statement | no | zero at run time | Tensor module: `c.device(gpu) = expr`, one kernel per assignment | no |
| Blaze | C++ "smart" expression templates | assignment | one assignment, but compound ops (products) are evaluated into temporaries on purpose | no | zero at run time | no in mainline | no (HPX backend is shared memory only) |
| Legate / cuPyNumeric | run-time task stream over Legion | each op becomes a task, issued asynchronously; Legion orders them by data dependences | mainline: none (window size 1); research system Diffuse: windows of tasks | Diffuse only, through windowed task fusion plus an MLIR kernel JIT | per-op Python + runtime dependence analysis; traces memoize it | yes, one task per op per GPU | yes, this is its core purpose |
| JAX / XLA | traced graph (jaxpr -> HLO), compiled by XLA | `jax.jit` region compiled on first call, cached per shape/dtype; outside `jit`, op by op with async dispatch | whole jitted function | yes: XLA sibling and producer-consumer multi-output fusion | tracing + full compile on first call, then cached | yes | yes, SPMD partitioner inserts collectives, fusion runs per device after partitioning |
| Dask (array) | run-time task graph (HighLevelGraph, Blockwise layers) | `.compute()`, `persist`, `dask.compute(c, d)`, implicit conversion | Blockwise layer fusion (graph rewrite, no kernel codegen) and linear task chain fusion | shares `a`, `b` tasks if computed together, no single fused kernel | Python dicts, large for millions of tasks, documented as a real cost | via CuPy chunks, one CuPy call per op per chunk (inference) | yes, this is its core purpose |
| Halide | staged DSL: algorithm + schedule, compiled AOT or JIT | `realize` / `compile_to_*` | the whole pipeline; fusion decided by the schedule | yes, manually: `Pipeline({c, d})` + `d.compute_with(c, x)` | compile time is the main cost, autoschedulers exist | yes, through schedule directives | not upstream (research "Distributed Halide") |
| MLX | run-time lazy graph | `mx.eval`, `print`, `.item()`, NumPy conversion, control flow on arrays, save | without `mx.compile`: no fusion, one kernel per op; with `mx.compile`: element-wise + broadcast subgraphs fused | yes inside `mx.compile`: fused `Compiled` primitive has several outputs | per-node graph cost "grows with the compute graph size"; compile cached per shape/dtype | Metal and CUDA | `mx.distributed` collectives, no graph-level partitioning |
| ArrayFire | run-time JIT tree (AST) per array | `eval`, `sync`, use by a non-JIT function, heuristics (tree height, kernel parameter size, memory pressure) | element-wise JIT ops in one tree | yes: `af::eval(c, d)` builds one kernel with several outputs, but only if all outputs have the same dims | node allocation + hashing; generated kernels cached by hash | CUDA, OpenCL, oneAPI, CPU | no |
| Taskflow | run-time task graph of opaque callables | `executor.run(taskflow)` | none (tasks are opaque) | no | cheap graph nodes, can be built once and rerun | `tf::cudaGraph` wraps native CUDA graphs | no |
| HPX | futures / dataflow runtime | futures become ready; `hpx::dataflow` | none at the array level | no | per-task future and scheduling overhead | through HPX-Kokkos / CUDA futures | yes, AGAS + parcels |

## Two families, and what each can see

Template-based libraries (xtensor, Eigen, Blaze) build the expression tree as a C++ type. The tree is "evaluated" in one loop when it is assigned to a container. The fusion scope is therefore exactly one assignment statement. Nothing in the type system links two statements, so `c = a + b; d = a + 2*b` gives two loops that each read `a` and `b`. None of the three offers a multi-assignment API. xtensor's documentation states it directly: "if you use the same expression in two assign statements, the computation of the expression will be done twice" ([xtensor expression.rst](https://github.com/xtensor-stack/xtensor/blob/master/docs/source/expression.rst)).

Graph-based libraries (MLX, ArrayFire, JAX/XLA, Dask, Legate) record operations at run time. They can see across statements, but only up to the point where something forces evaluation. Every one of them needs a trigger, and the trigger is where multi-output fusion is won or lost. The patterns observed are:

1. An explicit evaluation call that takes several arrays at once: `af::eval(c, d)`, `mx.eval(c, d)`, `dask.compute(c, d)`.
2. Host access to the data: printing, `.item()`, conversion to NumPy, scalar used in control flow (MLX, ArrayFire, JAX).
3. A non-fusible consumer forces its fusible inputs (ArrayFire: "the variable is required to compute a different variable that is not JIT-supported").
4. Resource heuristics: ArrayFire flushes a tree when it gets too tall, when the kernel parameter block would exceed the platform limit, or under memory pressure.
5. A delimited region compiled as a whole: `jax.jit`, `mx.compile`, Halide pipelines, Legate traces, Diffuse windows. This is the only pattern in which the library sees the full set of outputs before choosing kernels.

Multi-output fusion of siblings that share inputs exists in four places: XLA (automatic, cost-model driven), MLX `compile` (automatic within element-wise subgraphs), ArrayFire `af::eval(a, b, ...)` (automatic but only for same-shape outputs requested together), Halide `compute_with` (manual, schedule-driven). Diffuse handles it implicitly by fusing a window of tasks. Dask shares the input tasks but does not produce one kernel.

## Per library

### xtensor

- Family: expression templates. "An expression such as `x + y * sin(z)` does not hold the result. Values are only computed upon access or when the expression is assigned to a container" ([expression.rst](https://github.com/xtensor-stack/xtensor/blob/master/docs/source/expression.rst)).
- Trigger: assignment, element access, or `xt::eval`, which returns a new container or a reference if the argument is already a container (same file).
- Fusion scope: one assignment. Same expression in two statements is computed twice (quote above). By default xtensor assigns through a temporary to stay safe against aliasing, and `xt::noalias(b) = a + c` removes that temporary because "xtensor cannot detect such cases automatically" ([container.rst](https://github.com/xtensor-stack/xtensor/blob/master/docs/source/container.rst)).
- Graph cost: none at run time, all in compile time (template instantiation).
- GPU and distributed: none in xtensor itself.

### Eigen

- Family: expression templates, both in the dense core and in the unsupported Tensor module. "The most common way to evaluate an expression is to assign it to a Tensor"; `.eval()` materialises an intermediate and is itself an operation that is evaluated later ([Tensor README](https://github.com/PX4/eigen/blob/master/unsupported/Eigen/CXX11/src/Tensor/README.md), GitHub mirror of the GitLab file).
- Trigger: assignment to a `Tensor` or `TensorMap`, or explicit `.eval()`.
- Fusion scope: one assignment.
- GPU: the Tensor module evaluates an assignment on a device chosen with `c.device(my_device) = a + b`, with `DefaultDevice`, `ThreadPoolDevice` and `GpuDevice` (same README, "Evaluating With the DefaultDevice" / "Evaluating On GPU"). Each assignment is one evaluation on that device (inference: one kernel launch per statement, so two for `c` and `d`).
- Distributed: none.

### Blaze

- Family: "smart expression templates". The design point, from the Blaze authors' paper "Expression Templates Revisited" (Iglberger et al., SIAM J. Sci. Comput. 2012, doi:10.1137/110830125, not re-checked), is that naive expression templates hurt for compound operations such as matrix products, so Blaze evaluates those into temporaries and keeps element-wise parts fused.
- Trigger: assignment.
- Fusion scope: one assignment, element-wise only.
- Parallelism: OpenMP, C++ threads, Boost threads or HPX, chosen at compile time (`BLAZE_USE_HPX_THREADS`, HPX has priority), with size thresholds below which Blaze stays serial "in case Blaze deems the parallel execution as counterproductive" ([Tutorial.h, HPX Parallelization section](https://github.com/parsa/blaze/blob/master/blaze/Tutorial.h), [SMP.h](https://github.com/parsa/blaze/blob/master/blaze/system/SMP.h), GitHub mirror of the bitbucket repo).
- GPU: none in mainline. Distributed: none (HPX is used as a shared-memory thread pool here).

### Legate / cuPyNumeric

- Family: run-time task stream. Each NumPy call becomes one or more Legate tasks over partitioned stores, handed to Legion, which builds a dependence graph and runs tasks asynchronously across CPUs, GPUs and nodes ([cuPyNumeric README](https://github.com/nv-legate/cupynumeric)).
- Trigger: none in the usual sense. Execution is deferred and asynchronous; the program blocks only when it needs a value (inference from the Legion model).
- Fusion in mainline: none at present. Legate has a scheduling window, but its default is `LEGATE_WINDOW_SIZE_DEFAULT 1` ([env_defaults.h](https://github.com/nv-legate/legate/blob/main/src/cpp/legate/utilities/detail/env_defaults.h)), so each op is one task. `legate::experimental::Trace` ([trace.h](https://github.com/nv-legate/legate/blob/main/src/cpp/legate/experimental/trace.h)) memoizes Legion's dependence analysis for repeated loops; it reduces runtime overhead but does not fuse kernels.
- Fusion in research: Diffuse (Yadav et al., ASPLOS 2025, [arXiv:2406.18109](https://arxiv.org/abs/2406.18109)) buffers "tasks into a window of tasks to be analyzed before submission to the underlying runtime". Fusibility is checked on a scale-free IR: two index tasks fuse when their dependences are at most point-wise (launch domain equivalence, true, anti and reduction dependence constraints), with partition aliasing checked by constant-time equality, not by intersecting sub-stores. Fusion is greedy over the longest fusible prefix of the window. Stores written then read inside the fused prefix, not used later and with no live user reference, become task-local temporaries. Fused task bodies are composed in MLIR, then loop-fused and lowered "to GPU kernel launches or OpenMP parallel regions". Results are memoized for isomorphic task streams. Speedups are 1.86x geo-mean (0.93x to 10.7x) on up to 128 GPUs, and the authors report that task fusion alone gave no speedup; the gains came from kernel fusion plus temporary removal. JIT warmup is large (CFD 5.1 s to 10.9 s on 8 GPUs, break-even near 25 iterations). Cross-GPU aliasing (halos) reduces fusion: CFD gains more on one GPU than on many.
- Multi-output: not discussed explicitly in Diffuse; siblings that read the same stores fall in the same window and the constraints allow multiple read-only uses, so they can fuse (inference).

### JAX / XLA

- Family: traced graph. `jax.jit` wraps arguments in tracers that "record all JAX operations performed on them", producing a jaxpr compiled by XLA ([jit-compilation.md](https://github.com/jax-ml/jax/blob/main/docs/jit-compilation.md)).
- Trigger: inside `jit`, compilation on the first call, then the cached executable is reused; static arguments, shapes and dtypes key the cache (same file). Outside `jit`, JAX sends "one operation at a time to the accelerator", which "limits the ability of the XLA compiler to optimize". Results are futures (asynchronous dispatch), and `block_until_ready` waits ([async_dispatch.rst](https://github.com/jax-ml/jax/blob/main/docs/async_dispatch.rst)).
- Fusion detection: XLA:GPU's main pass is PriorityFusion, which ranks producers by `priority = time_unfused - time_fused` from a cost model, including the cost of duplicating a producer, and fuses greedily from a priority queue ([priority_fusion.h](https://github.com/openxla/xla/blob/main/xla/backends/gpu/transforms/priority_fusion.h)).
- Multi-output fusion: a dedicated pass fuses sibling ops ("common input parameters have to be read only once") and producer-consumer pairs whose producer has other users, building a fusion with a tuple root ([gpu multi_output_fusion.h](https://github.com/openxla/xla/blob/main/xla/backends/gpu/transforms/multi_output_fusion.h)). It checks reachability so no cycle is introduced. `c = a + b; d = a + 2*b` is exactly the sibling case. There is also a generic profit-driven sibling pass with a reachability map and a priority worklist ([service/multi_output_fusion.h](https://github.com/openxla/xla/blob/main/xla/service/multi_output_fusion.h)) and a CPU variant.
- Graph cost: Python tracing plus a full XLA compile per new signature; amortised by the cache.
- Distributed: sharding annotations drive the SPMD partitioner (GSPMD, [arXiv:2105.04663](https://arxiv.org/abs/2105.04663)), which inserts collectives; fusion then runs on the per-device program. Communication is visible to the compiler (inference about pass ordering from the GSPMD paper's pipeline).

### Dask (array)

- Family: run-time task graph. Arrays are `HighLevelGraph`s made of layers, many of them `Blockwise`.
- Trigger: `.compute()`, `persist`, `dask.compute(x, y)` for several outputs at once; graph optimisation runs "just before you submit the graph" ([phases-of-computation.rst](https://github.com/dask/dask/blob/main/docs/source/phases-of-computation.rst)).
- Fusion detection: `optimize_blockwise` fuses stacked Blockwise layers ("like `x.T + 123` we can fuse these into a single Blockwise operation. This happens before any actual tasks are generated"); `fuse_roots` merges independent root layers feeding a Blockwise layer to avoid moving data between workers ([blockwise.py](https://github.com/dask/dask/blob/main/dask/blockwise.py)). Low-level `cull`, `inline`, `inline_functions` and `fuse` (merge linear chains) work on the task dict ([optimize.rst](https://github.com/dask/dask/blob/main/docs/source/optimize.rst)).
- Fusion is a graph rewrite: a fused task calls the per-chunk functions in sequence, there is no kernel code generation (inference from blockwise.py).
- Multi-output: `dask.compute(c, d)` merges both graphs so `a` and `b` chunks are produced once, but `c` and `d` stay separate tasks (inference: `fuse` only merges linear chains).
- Graph cost: documented as significant: "when these dictionaries become large (millions of tasks) the overhead of constructing them can become considerable", plus serialisation and scheduler cost; advice is bigger chunks (phases-of-computation.rst).
- GPU: chunks can be CuPy arrays; each op remains a separate CuPy call per chunk (inference).
- Distributed: yes, `dask.distributed` scheduler.

### Halide

- Family: staged DSL separating algorithm from schedule, compiled ahead of time or JIT.
- Trigger: `realize` (JIT) or `compile_to_*` (AOT).
- Fusion: chosen by the schedule (`compute_at`, `compute_root`, inlining by default). Several outputs form one pipeline with `Pipeline(const std::vector<Func> &outputs)` ([Pipeline.h](https://github.com/halide/Halide/blob/main/src/Pipeline.h)). Sibling loop fusion is explicit: `compute_with` schedules "the iteration over this stage to be fused with another stage 's' from outermost loop to a given LoopLevel", requiring the same schedule over the fused loops and no dependency between the two stages ([Func.h](https://github.com/halide/Halide/blob/main/src/Func.h)). `d.compute_with(c, x)` is the `c`/`d` case.
- Graph cost: compile time; autoschedulers (Mullapudi2016, Adams2019, Li2018, Anderson2021) pick schedules.
- GPU: yes, through `gpu_blocks` / `gpu_threads` schedules and several GPU targets.
- Distributed: not upstream; research work "Distributed Halide" (Denniston, Kamil, Amarasinghe, PPoPP 2016, not re-checked).

### MLX

- Family: run-time lazy graph. "When you perform operations in MLX, no computation actually happens. Instead a compute graph is recorded" ([lazy_evaluation.rst](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/lazy_evaluation.rst)).
- Trigger: `mx.eval(...)` on one or several arrays, plus implicit evaluation on `print`, conversion to NumPy, `memoryview`, saving, `.item()`, and scalar arrays used in control flow (same file). The docs frame the tradeoff: "some fixed overhead with each graph evaluation" against "slight overhead which grows with the compute graph size"; "a few tens of operations to many thousands of operations per evaluation should be okay", and the natural place is once per outer iteration.
- Fusion: lazy evaluation alone does not compile or fuse ("MLX does not compile and rerun compute graphs. They are all generated dynamically"). `mx.compile` fuses ([compile.rst](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/compile.rst)). In the source, fusible primitives are unary, binary, ternary and broadcast; fusion stops at a depth limit (`max_compile_depth = 11`), at `max_compile_arrays = 24` inputs, on stream mismatch, at non-fusible primitives, and when a global output has a different shape. Broadcasts with parents outside the section are split so fusion can continue ([compile.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/compile.cpp), `compile_fuse`).
- Multi-output: the fused `Compiled` primitive takes vectors of inputs and outputs, and intermediates used outside the fused section are added as extra outputs (compile.cpp). So `c` and `d` with shared `a`, `b` become one kernel when compiled.
- Recompilation on change of shape, rank, dtype or number of inputs; `shapeless=True` avoids shape-triggered recompiles (compile.rst).
- GPU: Metal and CUDA backends. Distributed: `mx.distributed` collectives; the graph is not partitioned by the library (inference).

### ArrayFire

- Family: run-time JIT. Each JIT-supported call adds a node to the array's AST; the tree becomes one generated kernel ([docs/pages/jit.md](https://github.com/arrayfire/arrayfire/blob/master/docs/pages/jit.md)).
- Trigger: explicit `eval`, or "the variable is required to compute a different variable that is not JIT-supported" (same file). Heuristics in the backend also force evaluation of children when the tree height exceeds `getMaxJitSize()`, when kernel parameters would exceed the platform limit (4096 bytes on NVIDIA), or when JIT-held buffers create memory pressure (half of the bytes allocated so far) ([cuda/Array.cpp](https://github.com/arrayfire/arrayfire/blob/master/src/backend/cuda/Array.cpp), `passesJitHeuristics`).
- Multi-output: `evalMultiple` evaluates several arrays in one generated kernel, but only when all have the same dims; otherwise it falls back to evaluating each one (Array.cpp). Generated kernels are named from the node set and looked up in a module cache by a deterministic hash before compiling ([cuda/jit.cpp](https://github.com/arrayfire/arrayfire/blob/master/src/backend/cuda/jit.cpp)).
- Graph cost: node allocation per op, hashing per eval, compile on cache miss. The docs note that on small sizes run time is "dominated by on-the-fly compilation of the JIT code".
- GPU: CUDA, OpenCL, oneAPI and CPU backends. Distributed: none.

### Taskflow

- Family: static or dynamic task graph of opaque C++ callables; `executor.run(taskflow).wait()` runs it, `executor.async` / `silent_dependent_async` build graphs dynamically ([README](https://github.com/taskflow/taskflow)).
- Fusion: none; tasks are black boxes, so there is no expression to fuse.
- GPU: `tf::cudaGraph` builds native CUDA graphs ([taskflow/cuda/cuda_graph.hpp](https://github.com/taskflow/taskflow/blob/master/taskflow/cuda/cuda_graph.hpp)); this reduces launch overhead, not memory traffic.
- Distributed: none. Relevance for tensorlib: a possible scheduling substrate, not a fusion model.

### HPX

- Family: asynchronous many-task runtime with futures, `hpx::dataflow`, and a global address space (AGAS) with parcels for remote actions ([HPX README](https://github.com/STEllAR-GROUP/hpx)).
- Fusion: none at the array level; HPX schedules whatever tasks it is given. Array libraries on top (Blaze with `BLAZE_USE_HPX_THREADS`, the archived Phylanx project) do their own fusion or none.
- GPU: through HPX-Kokkos and CUDA futures. Distributed: yes, the core purpose.

## What this means for tensorlib

These points are inference from the material above, offered as input for the core-architecture decisions.

1. Templates alone cannot fuse the reference workload `c = a + b; d = a + 2*b`. No template library does it. Getting it requires either an explicit multi-assignment API (for example assigning a tuple of expressions to a tuple of containers in one loop, which none of xtensor, Eigen, Blaze offers) or a run-time graph.
2. A run-time graph needs a trigger policy. The options that let the library see all outputs at once are a delimited region (`jit`, `compile`, trace, window) or a multi-array evaluation call (`af::eval(c, d)`, `mx.eval(c, d)`). Implicit triggers (host access, non-fusible consumer, resource limits) are needed anyway as safety nets; ArrayFire's height, parameter size and memory pressure heuristics are a concrete list.
3. Multi-output sibling fusion in the systems that have it uses the same ingredients: a reachability check to avoid cycles (XLA), an output set known at fusion time (all), and a shape compatibility rule (ArrayFire requires equal dims, MLX stops at outputs of a different shape, XLA requires matching fusion kinds).
4. JIT compile cost is paid once per graph signature and cached by a hash of the graph (ArrayFire, MLX, XLA, Diffuse). Shape-agnostic caching (MLX `shapeless`) avoids recompiles when only extents change, which matters for dynamic-rank tensors.
5. For distributed runs, Diffuse shows that fusion must reason about partitions, not just element-wise ops: halo aliasing cuts fusion, and kernel fusion plus temporary removal, not task fusion alone, gave the speedup. XLA partitions first, then fuses per device. Both argue for putting partition information into the graph IR, which matches the map's decision to carry partitioned tensors from the start.
6. Graph construction cost is real for per-op Python graphs (Dask, MLX), so a C++ graph should keep per-node cost small and support memoized reuse of a whole graph (Legate traces, `jax.jit` cache, Diffuse memoization of isomorphic streams).
