# Existing HPC libraries that cover part of tensorlib

Ticket: [15-existing-hpc-libraries](../issues/15-existing-hpc-libraries.md)
Date of survey: 2026-10-30

## Question

Which existing libraries already deliver, with proven performance on distributed HPC machines, some or all of what tensorlib targets: a NumPy-like tensor API, lazy evaluation with fusion across several outputs, Kokkos or equivalent portability (CPU SIMD, threads, NVIDIA/AMD/Intel GPUs), partitioned tensors over MPI, bindings to other languages, and PETSc-like linear algebra? For each one: what it covers, language and C++ usability, maturity, published HPC performance, licence, and whether tensorlib should adopt it, extend it, depend on it, or only learn from it.

## Method and confidence

- Repository facts (licence, last push, last release, archived flag, README claims, header contents) were read directly from GitHub with `gh api` on 2026-10-30. These are marked **[verified]**.
- Performance numbers come from the paper abstracts or the authors' own publication pages. They are self-reported by each project. Marked **[paper]**.
- Anything I conclude without a direct source is marked **[inference]**.
- Some vendor doc sites (docs.nvidia.com, docs.jax.dev) returned HTTP 403 to the fetcher, so for those I used the GitHub copy of the same docs.

## Headline findings

1. **Legate and cuPyNumeric are end of life.** Both GitHub repositories were archived with the note "End of life; no longer maintained. Final public release: v26.06.01." The cuPyNumeric README says "we do not recommend starting new projects with cuPyNumeric" and points users to nvmath-python's distributed module, CuPy with mpi4py or Dask, JAX multi-controller, and PyTorch distributed. **[verified]** ([legate](https://github.com/nv-legate/legate), [cupynumeric](https://github.com/nv-legate/cupynumeric)). This was the closest existing match to tensorlib (C++ `NDArray` API, distributed, task fusion via Diffuse) and it cannot be a dependency any more. The underlying Legion runtime is still maintained (last push 2026-10-04). **[verified]** ([legion](https://github.com/StanfordLegion/legion))
2. **No surveyed library combines all six targets.** The ones that come closest are Python-first (JAX/XLA, DaCe, Heat, Arkouda), and none of them is built on Kokkos or is meant to be used from C++ as a library.
3. **The Kokkos ecosystem already provides most of the local building blocks.** Kokkos Kernels covers local dense and sparse linear algebra, Kokkos-FFT covers local FFTs, and PETSc and Trilinos/Tpetra provide distributed linear algebra on Kokkos, proven at scale. tensorlib should depend on these rather than rewrite them.
4. **The piece nobody provides in C++** is a NumPy-like tensor API on Kokkos with lazy multi-output fusion and a partitioned-tensor model over MPI. That is the only part where building is justified.

## Comparison table

Legend: Y = yes, P = partial or limited, N = no, X = project is end of life.

| Library | NumPy-like API | Lazy eval + multi-output fusion | Portability (CPU SIMD / threads / NVIDIA / AMD / Intel GPU) | Partitioned tensors over MPI | Bindings | PETSc-like LA | Primary language | Status (2026-10) | Licence | Verdict for tensorlib |
|---|---|---|---|---|---|---|---|---|---|---|
| Legate + cuPyNumeric | Y (Python), P (C++ `NDArray`) | Y (Diffuse task + kernel fusion, research) | CPU, OpenMP, NVIDIA only | Y (Legion, not MPI-style) | Python | P (Legate Sparse) | Python over C++ | **X, archived, final v26.06.01** | Apache-2.0 | Learn only |
| DaCe | Y (NumPy subset in Python) | Y (SDFG transformations, map fusion) | CPU, NVIDIA, AMD (HIP), FPGA; Intel GPU not listed | Y (MPI library nodes, auto distribution in SC21 paper) | Python front end, emits C++ | N | Python, generates C++ | Active, v2.0.0-beta1 (2026-10-24) | BSD-3 | Learn (IR and fusion); possible codegen backend later |
| Heat | Y (Python `DNDarray`) | N (eager, PyTorch per node) | CPU, CUDA, ROCm via PyTorch | Y (mpi4py, `split` axis) | Python only | P (some dense LA) | Python | Active, v1.9.0 (2026-09-24) | MIT | Learn (partition model) |
| Arkouda | P (pandas/NumPy-like client) | N | CPU (Chapel); GPU limited | Y (Chapel, not MPI) | Python client | N | Chapel server + Python | Active, v2026.10.14 | MIT | Learn only |
| JAX / XLA | Y (`jax.numpy`, Array API) | Y (XLA fusion incl. multi-output fusion passes) | CPU, NVIDIA, AMD, TPU | Y (sharding, `shard_map`, multi-controller; NCCL, not MPI) | Python; C API via PJRT | P (dense, no distributed sparse solvers) | Python over C++ | Very active | Apache-2.0 | Learn (fusion, sharding); alternative for Python-only users |
| HPX | P (parallel STL algorithms) | N | CPU threads; GPU via HPX-Kokkos | P (`partitioned_vector`, AGAS, not MPI) | N | N | C++ | Active, v2.0.0-rc1 (2026-09-28) | BSL-1.0 | Learn; optional runtime |
| Kokkos Kernels | N | N | Kokkos (all) + vendor TPLs | N (local only) | N | Local BLAS, LAPACK subset, sparse, batched | C++ | Active, 5.2.2 (2026-09-10) | Apache-2.0 + LLVM exc. | **Depend on** |
| Kokkos Comm | N | N | Kokkos (all) | P (MPI and experimental NCCL on Views) | N | N | C++ | Experimental, v0.1.0, "not yet ready for general use" | Apache-2.0 + LLVM exc. | Depend on later / contribute |
| Kokkos-FFT | P (`numpy.fft`-like) | N | Kokkos + FFTW, cuFFT, hipFFT, oneMKL | N (local only) | N | N | C++20 | Active, v2.0.0 (2026-07-27) | MIT or Apache-2.0 + LLVM exc. | **Depend on** |
| Trilinos (Tpetra and solvers) | N | N | Kokkos (all) | Y (Map, Import/Export, MultiVector, CrsMatrix) | P (PyTrilinos2) | **Y** (Belos, Ifpack2, MueLu, Amesos2) | C++ | Active, 17.2.1 (2026-08-31) | Per package, mostly BSD-3 | **Depend on / interoperate** |
| PETSc (Kokkos backend) | N | N | Kokkos, CUDA, HIP, SYCL back ends | Y (Vec, Mat, DMDA halos, PetscSF) | Python (petsc4py), Fortran, Julia (PETSc.jl) | **Y** (the reference) | C | Active | BSD-2 | **Depend on / interoperate** |
| Ginkgo | N | N | Own back ends: OpenMP, CUDA, HIP, SYCL | Y (distributed matrices and solvers over MPI) | P (pyGinkgo, community) | Y (sparse solvers and preconditioners) | C++ | Active, v1.11.0 (2025-12-15) | BSD-3 | Optional dependency for sparse solvers |
| TiledArray | P (Einstein-notation tensor expressions) | P (lazy expressions) | CPU, CUDA, HIP | Y (MADNESS runtime, block-sparse) | N | N (tensor contractions, not solvers) | C++ | Active commits; last release v1.1.0 (2024-11) | **GPL-3.0** | Learn only (licence) |
| Cyclops Tensor Framework | P (Python `numpy.ndarray`-like) | N | CPU (MPI + OpenMP); GPU support old | Y (cyclic distribution over MPI) | Python | P (ScaLAPACK interop) | C++ | Low activity: last release v1.5.5 (2019), few commits since 2023 | BSD-2 | Learn only |
| xtensor | Y (closest C++ NumPy API) | P (lazy expression templates, single output) | xsimd SIMD; partial threads; no GPU | N | Python, Julia, R (xtensor-python, xtensor-julia, xtensor-r) | P (xtensor-blas) | C++ | Active, 0.27.1 | BSD-3 | Learn; possibly reuse API design |
| Eigen (Tensor module) | P | P (expression templates, single output) | SIMD, thread pool, CUDA/HIP, SYCL devices | N | N | Dense + sparse (single node) | C++ | Active, 5.0 (2025-09-30); Tensor still in `unsupported/` | MPL-2.0 | Learn only |
| Blaze | P (vector/matrix, not N-d) | P (smart expression templates) | SIMD up to AVX-512; OpenMP, C++ threads, HPX | N | N | Dense + sparse (single node) | C++ | Stale: last release 3.8.2, no activity on main mirror since 2025 | BSD-3 | Learn only |
| Fastor | N (small fixed-size tensors) | P (compile-time expression and contraction optimisation) | SIMD only | N | N | N | C++ | Stale: last release V0.6.4 (2023-06), no commits in last 6 months | MIT | Learn (fixed-shape tensors) |

Sources for the status, release and licence columns: GitHub API on 2026-10-30, see the per-library sections.

## Per-library notes

### Legate and cuPyNumeric

- **Covers**: drop-in NumPy replacement mapped onto the Legion task runtime; data and work are partitioned automatically across CPUs and GPUs. A C++ API exists: `src/cupynumeric/operators.h` exposes `NDArray add(...)`, `dot`, `sum`, `where`, `sort`, `reshape` and more, with C++ examples for a stencil, GEMM and Black-Scholes. **[verified]** ([operators.h](https://github.com/nv-legate/cupynumeric/blob/main/src/cupynumeric/operators.h), [examples/cpp](https://github.com/nv-legate/cupynumeric/tree/main/examples/cpp))
- **Performance**: up to 1280 CPU cores and 256 GPUs, with 10x (CPU) and 100x (GPU) speedups over Dask on comparable programs. **[paper]** ([Bauer and Garland, SC19](https://research.nvidia.com/publication/2019-11_Legate-NumPy%3A-Accelerated)). Diffuse adds runtime task fusion plus JIT kernel fusion, 1.86x geometric-mean speedup on unmodified cuNumeric and Legate Sparse programs on up to 128 GPUs. **[paper]** ([Yadav et al., ASPLOS 2025](https://arxiv.org/abs/2406.18109))
- **Status**: archived, final release v26.06.01. **[verified]**
- **Verdict**: learn only. Diffuse is the most relevant published design for distributed multi-output fusion and should be read before tensorlib designs its graph. NVIDIA dropping a distributed NumPy product also suggests how expensive one is to maintain **[inference]**.

### DaCe

- **Covers**: Python/NumPy code (via `@dace`) is lowered to the SDFG data-centric IR, transformed (including fusion), and emitted as C++/CUDA/HIP code. Targets multi-core CPUs, NVIDIA and AMD GPUs, and FPGAs in a separate repo. **[verified]** ([README](https://github.com/spcl/dace))
- **Performance**: automatic MPI distribution reached 93.16% scaling efficiency on 512 nodes of Piz Daint (up to 23,328 cores). **[paper]** ([Ziogas et al., SC21](https://arxiv.org/abs/2107.00555))
- **C++ usability**: the input is Python; the output is C++ that can be called from C++, but DaCe is not a C++ library you write tensor code against. **[inference]**
- **Verdict**: learn from the SDFG and its transformations; possibly use as an offline codegen tool later. Not a dependency for the core.

### Heat

- **Covers**: distributed tensor framework on PyTorch (node-local compute) and mpi4py, with a `split` attribute that picks the partitioned axis. CPUs and CUDA/ROCm GPUs. **[verified]** ([README](https://github.com/helmholtz-analytics/heat))
- **Evaluation**: eager, no lazy fusion **[inference from design: each call is a PyTorch op]**.
- **Performance**: "up to two orders of magnitude" over similar frameworks. **[paper]** ([Götz et al., IEEE BigData 2020](https://arxiv.org/abs/2007.13552))
- **Verdict**: learn from its single-split-axis partition model, which is simple and covers many cases. Python only, so not a dependency.

### Arkouda

- **Covers**: Python client that sends NumPy/pandas-like operations to a Chapel server running on the cluster. Aimed at data science (sort, groupby, set ops), not stencils or solvers. **[verified]** ([repo](https://github.com/Bears-R-Us/arkouda))
- **Performance**: argsort of 256 TiB on 8,192 HPE Cray EX nodes in about 31 s (about 8,500 GiB/s). **[paper, vendor-reported]** ([Chapel 2.0 announcement](https://chapel-lang.org/blog/posts/announcing-chapel-2.0))
- **Verdict**: learn only. Different language runtime, different workload.

### JAX / XLA with sharding

- **Covers**: `jax.numpy` with `jit` compilation to XLA, which performs operator fusion including dedicated multi-output fusion passes (`xla/service/multi_output_fusion.cc`, `xla/backends/gpu/transforms/multi_output_fusion.cc`, `xla/service/cpu/cpu_multi_output_fusion.cc`). **[verified]** ([openxla/xla](https://github.com/openxla/xla)). Distributed arrays via sharding and `shard_map`; multi-controller mode launches one process per host through Slurm or similar, with `jax.distributed.initialize`. **[verified]** ([multi_process.md](https://github.com/jax-ml/jax/blob/main/docs/multi_process.md))
- **What it lacks for tensorlib**: Python-first; the C++ entry point is the PJRT C API and StableHLO, not a tensor library API. It does not communicate over MPI. No distributed sparse solvers. **[inference]**
- **Verdict**: for a Python user with dense array workloads on GPU clusters, JAX already does the job and tensorlib should not pretend otherwise. For tensorlib, learn from XLA's fusion heuristics (including multi-output fusion) and from the sharding annotation model.

### HPX

- **Covers**: C++ asynchronous many-task runtime with parallel algorithms and a distributed `partitioned_vector` container. **[verified]** ([TheHPXProject/hpx](https://github.com/TheHPXProject/hpx), `components/containers/partitioned_vector`). GPU use goes through HPX-Kokkos, whose repo was last pushed 2025-04-24. **[verified]** ([hpx-kokkos](https://github.com/STEllAR-GROUP/hpx-kokkos))
- **Performance**: Octo-Tiger (HPX + Kokkos) reached 47.59% parallel efficiency on full Perlmutter (6,880 A100) and 51.37% on 2,048 MI250X on Frontier, plus runs up to 6,144 nodes on Fugaku. **[paper]** ([Daiß et al., arXiv:2412.15518](https://arxiv.org/abs/2412.15518))
- **Verdict**: learn from it. It replaces MPI with its own runtime, which conflicts with tensorlib's MPI-first and PETSc-interop goals **[inference]**. Kokkos itself can use HPX as an execution space, so it remains reachable through Kokkos if needed.

### Kokkos Kernels

- **Covers**: local BLAS, a LAPACK subset, sparse kernels (SpMV, SpGEMM, triangular solves, ILU, Gauss-Seidel), batched dense and sparse solvers, graph kernels, and calls into vendor libraries when available. **[paper]** ([Rajamanickam et al., arXiv:2103.11991](https://arxiv.org/abs/2103.11991); [Liegeois et al., TPDS 2023](https://www.sandia.gov/research/publications/details/performance-portable-batched-sparse-linear-solvers-2023-05-01/))
- **Status**: released with Kokkos, 5.2.2 on 2026-09-10. **[verified]**
- **Verdict**: depend on it for every local linear algebra kernel. Writing tensorlib's own GEMM or SpMV would be reinventing the wheel.

### Kokkos Comm

- **Covers**: explicit communication on Kokkos Views with MPI and an experimental NCCL back end. README: "This is a work in progress and is not yet ready for general use." First release v0.1.0 on 2026-05-06. **[verified]** ([kokkos-comm](https://github.com/kokkos/kokkos-comm))
- **Verdict**: track it and contribute; use plain MPI (GPU-aware where available) behind tensorlib's own hook until Kokkos Comm stabilises **[inference]**.

### Kokkos-FFT

- **Covers**: `numpy.fft`-like API on Kokkos Views, dispatching to FFTW, cuFFT, hipFFT/rocFFT or oneMKL. Local only: "not using MPI, or running within a single MPI process". Now needs C++20. **[verified]** ([kokkos-fft](https://github.com/kokkos/kokkos-fft), [JOSS paper](https://doi.org/10.21105/joss.08391))
- **Verdict**: depend on it for local FFTs. Distributed FFTs (pencil decomposition) are not covered; they would need either a wrapper over vendor distributed FFTs or new code.

### Trilinos (Tpetra and the solver stack)

- **Covers**: distributed vectors, multivectors and sparse matrices (Tpetra) built on Kokkos, with Krylov solvers (Belos), preconditioners (Ifpack2), algebraic multigrid (MueLu) and direct solver interfaces (Amesos2). Epetra is being removed; the modern stack is Tpetra + Kokkos. **[paper]** ([Mayr et al., arXiv:2503.08126, ACM TOMS 2026](https://arxiv.org/abs/2503.08126))
- **Status**: release 17.2.1 on 2026-08-31. Licence is per package, mostly BSD-3. **[verified]** ([Trilinos](https://github.com/trilinos/Trilinos), [licence page](https://trilinos.github.io/license.html))
- **Verdict**: depend on or interoperate with it for distributed sparse linear algebra. Because Tpetra exposes its local data as Kokkos Views, zero-copy wrapping of a local part as a tensorlib tensor should be feasible **[inference]**.

### PETSc with its Kokkos backend

- **Covers**: the reference "PETSc-like" layer. Kokkos implementations exist for vectors, sequential and MPI AIJ matrices, structured grids (DMDA), Jacobi-type preconditioners, among others (`src/mat/impls/aij/mpi/kokkos/mpiaijkok.kokkos.cxx`, `src/dm/impls/da/kokkos/`, `src/ksp/pc/impls/bjacobi/bjkokkos/`). **[verified]** ([petsc mirror](https://github.com/petsc/petsc)). The design separates the user's GPU programming model from the one PETSc uses internally. **[paper]** ([Mills et al., Parallel Computing 2021](https://arxiv.org/abs/2011.00715))
- **Licence**: BSD-2. **[verified]**
- **Verdict**: depend on it or interoperate with it. The ticket's planned "PETSc-like linear algebra" should become "PETSc and Tpetra interop": wrap `Vec`/`Mat` local parts as tensors and pass tensor data into PETSc, rather than writing a new solver stack.

### Ginkgo

- **Covers**: sparse linear algebra (solvers, preconditioners, SpMV) with its own back ends for OpenMP, CUDA, HIP and SYCL, plus distributed MPI support. **[verified]** ([ginkgo](https://github.com/ginkgo-project/ginkgo))
- **Performance**: weak scaling to about 2 PFlop/s on 20,000 GPUs on Frontier with over 80% efficiency. **[paper, poster]** ([PASC23 poster](https://pasc23.pasc-conference.org/wp-content/uploads/2023/07/pos126-P25-Ginkgo-8212-A-High-Performance-Portable-Numerical-Linear-Algebra-Software.pdf))
- **Verdict**: optional dependency for sparse solvers. Not on Kokkos, so data exchange would go through raw device pointers. BSD-3 matches tensorlib.

### TiledArray

- **Covers**: distributed block-sparse tensor algebra on the MADNESS runtime, aimed at quantum chemistry contractions, with GPU support. **[verified]** ([tiledarray](https://github.com/ValeevGroup/tiledarray))
- **Licence**: GPL-3.0. **[verified]** A BSD-3 library cannot take a GPL dependency without making combined distributions GPL **[inference]**.
- **Verdict**: learn only.

### Cyclops Tensor Framework (CTF)

- **Covers**: distributed dense and sparse tensors over MPI with cyclic distribution and symmetry support, C++ and Python (`numpy.ndarray`-like) interfaces. **[verified]** ([ctf](https://github.com/cyclops-community/ctf))
- **Performance**: CCSD on 8,192 nodes of Blue Gene/Q, faster than NWChem on Cray XE6 for the cases tested. **[paper]** ([Solomonik et al., tech report EECS-2013-11](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2013/EECS-2013-11.html), published in JPDC 2014)
- **Status**: last release v1.5.5 in 2019, one commit in 2026 after a gap since 2023. **[verified]**
- **Verdict**: learn from its contraction mapping and cyclic distribution; do not depend on it.

### xtensor

- **Covers**: the closest existing C++ NumPy API (broadcasting, views, lazy expression templates), SIMD through xsimd, bindings to Python, Julia and R. No GPU, no distribution. Lazy evaluation fuses one expression into one assignment; there is no multi-output fusion. **[verified for features; inference for the fusion limit]** ([xtensor](https://github.com/xtensor-stack/xtensor))
- **Status**: 0.27.1, active. BSD-3.
- **Verdict**: tensorlib is described as being "in the spirit of xtensor". Reusing its API design, type traits and possibly its binding approach is reasonable. Its evaluation model (compile-time expression templates evaluated on assignment) does not map onto Kokkos kernels or multi-output fusion, so the core cannot simply be xtensor plus a Kokkos backend **[inference, to confirm in a prototype ticket]**.

### Eigen (Tensor module)

- **Covers**: the Tensor module supports N-d tensors, expression templates, and devices for a thread pool, CUDA/HIP and SYCL. It still lives in `unsupported/`. Eigen 5.0 was released on 2025-09-30. **[verified]** ([Eigen changelog](https://eigen.googlesource.com/mirror/+/refs/heads/RVV_Chip2/CHANGELOG.md))
- **Licence**: MPL-2.0.
- **Verdict**: learn only. Single node, and the module's support status is officially "unsupported".

### Blaze

- **Covers**: dense and sparse vectors and matrices with "smart expression templates", SIMD up to AVX-512, OpenMP, C++ threads and HPX parallelism. Not N-d. **[verified from project description]** ([live-clones/blaze](https://github.com/live-clones/blaze))
- **Status**: last release 3.8.2; the mirror shows no pushes since 2025-12. **[verified]**
- **Verdict**: learn only (its expression template design is well regarded for single-node performance).

### Fastor

- **Covers**: small fixed-size tensors with compile-time contraction and expression optimisation, SIMD. **[verified]** ([Fastor](https://github.com/romeric/Fastor))
- **Status**: last release V0.6.4 (2023-06), no commits in the last six months. MIT.
- **Verdict**: learn from it for the fixed-shape container kind (`xtensor_fixed`-like).

### Others worth knowing (not in the ticket list)

- **nvmath-python distributed APIs**: NVIDIA's recommended successor for multi-node array math. It wraps cuFFTMp and cuBLASMp, needs NVSHMEM, and is NVIDIA-only and Python-only. **[verified from docs index]** ([nvmath-python](https://github.com/NVIDIA/nvmath-python), [distributed APIs](https://docs.nvidia.com/cuda/nvmath-python/latest/distributed-apis/index.html))
- **mdspan / std::linalg**: `kokkos/mdspan` is the reference implementation of C++23 `std::mdspan`, active (last push 2026-10-28). **[verified]** ([kokkos/mdspan](https://github.com/kokkos/mdspan)). A strong candidate for tensorlib's non-owning view type.

## What tensorlib should reuse

| Need | Reuse | Mode |
|---|---|---|
| Parallel execution, memory spaces, SIMD | Kokkos core, `Kokkos::Experimental::simd` | Depend on |
| Local dense and sparse linear algebra | Kokkos Kernels | Depend on |
| Local FFT | Kokkos-FFT | Depend on |
| Distributed sparse LA, solvers, preconditioners | PETSc (Kokkos back end) and Trilinos/Tpetra; Ginkgo as an option | Interoperate, zero-copy where possible |
| Communication on device buffers | MPI directly now; Kokkos Comm once stable | Depend on later, contribute |
| Non-owning multidimensional view | `mdspan` | Depend on |
| Python bindings pattern | xtensor-python as prior art | Learn |
| Fusion design | XLA multi-output fusion, Diffuse, DaCe SDFG | Learn |
| Partition model | Heat (`split` axis), Tpetra `Map`, PETSc DMDA | Learn |

## Gaps nobody fills

1. **A C++-first, NumPy/Array-API-like tensor library on Kokkos.** Kokkos Views are containers, not a NumPy-like API (no broadcasting, no array expressions). xtensor has the API but no GPU. The C++ API of cuPyNumeric came closest and is now end of life. **[verified across sources]**
2. **Lazy evaluation with multi-output fusion, from C++, with Kokkos code generation.** XLA does multi-output fusion but lives behind Python and StableHLO. Diffuse did distributed task and kernel fusion but went down with Legate. DaCe does fusion from a Python front end. Expression-template libraries (xtensor, Eigen, Blaze, Fastor) fuse only one output per assignment. **[inference from the above]**
3. **Partitioned tensors over MPI with halo exchanges tied into the fusion graph.** PETSc DMDA handles structured-grid halos but not array-expression fusion; Heat partitions but evaluates eagerly; JAX shards but uses NCCL and no halo abstraction for stencils. **[inference]**
4. **Zero-copy bridge between a tensor API and the established distributed linear algebra (PETSc `Vec`/`Mat`, Tpetra `MultiVector`/`CrsMatrix`).** Each library exposes its local data as Kokkos Views, but there is no NumPy-like layer that wraps them. **[inference]**
5. **One C++ core with bindings for Python, Julia and Rust.** xtensor covers Python, Julia and R; PETSc covers Python and Julia; nothing covers Rust, and none of the distributed ones offer all three. **[inference]**
6. **Intel GPU support in the distributed array tools.** Of the distributed array libraries, none lists Intel GPUs as a supported target in the sources read here (Legate: NVIDIA only; Heat: CUDA/ROCm; DaCe: NVIDIA/AMD). Kokkos-based libraries and Ginkgo (SYCL) do cover them. **[verified for the listed READMEs]**

Distributed FFTs on Kokkos are a smaller gap as well: Kokkos-FFT is local only.

## Bottom line

- If a user wants distributed NumPy from Python on GPU clusters today, JAX (or Heat for MPI-centric CPU/GPU clusters) already does it. tensorlib should not claim to replace them.
- If a user wants distributed sparse solvers, PETSc and Trilinos already do it at exascale. tensorlib should interoperate, not rebuild.
- tensorlib's justified scope is the C++ layer between Kokkos and those libraries: a NumPy-like tensor API, lazy multi-output fusion compiled to Kokkos kernels, and a partitioned-tensor model over MPI that can hand its local parts to PETSc, Tpetra, Kokkos Kernels and Kokkos-FFT without copies.
