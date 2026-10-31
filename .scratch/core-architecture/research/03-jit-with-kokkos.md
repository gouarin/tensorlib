# Feasibility of JIT compilation with Kokkos

Ticket: [03-jit-with-kokkos](../issues/03-jit-with-kokkos.md)

## Question

Can a run-time graph be compiled into fused kernels on CPU, CUDA, HIP and SYCL in a library built on Kokkos? Covers NVRTC, hiprtc, SYCL online compilation, LLVM ORC, MLIR, kernel caches, first-execution latency, supercomputer constraints, and existing projects.

## Short answer

Yes, but Kokkos cannot be the JIT target. Kokkos is a compile-time C++ template layer. It has no run-time code generation, and its headers cannot go through the GPU run-time compilers (NVRTC and hiprtc compile device code only, without the standard library). A JIT path in tensorlib therefore has to emit code for each backend natively and use Kokkos only for what it provides at run time: memory (Views and raw pointers), execution space instances (streams and queues), and the CPU thread pool. The interop hooks for that exist and are documented.

Two designs are realistic:

1. One IR, lowered through LLVM (or MLIR) linked into the library. CPU through LLVM ORC, NVIDIA through the NVPTX backend and the driver PTX JIT, AMD through the AMDGPU backend, Intel through the SPIR-V backend and Level Zero or SYCL interop. No external compiler is needed on compute nodes. This is how Dr.Jit, AdaptiveCpp SSCP and Proteus work.
2. Source-string generation per backend. CUDA C++ for NVRTC, HIP for hiprtc, SYCL for the DPC++ `kernel_compiler` extension, C++ for a host compiler. Simpler to write and debug, but four generators, three of them tied to one vendor's toolkit version, and the SYCL path is experimental.

Generating Kokkos C++ and calling the full toolchain at run time (what PyKokkos and Kokkos.jl do) works, but costs seconds per kernel and needs the compiler, CMake and the exact Kokkos install on the compute node. It does not fit a library where fusion happens on every new expression shape.

First-execution latency is the main cost of any of these designs: tens of milliseconds for a small kernel through the driver, seconds when heavy C++ headers are parsed. So a two-level kernel cache (in memory, then on disk) is mandatory, and the disk cache must behave well on parallel file systems with thousands of ranks.

## 1. Why Kokkos itself cannot be JIT-compiled

- NVRTC "is meant for compiling only device CUDA C++ code. It does not accept host code or host compiler extensions in the input code" ([NVRTC docs](https://docs.nvidia.com/cuda/nvrtc/index.html), quoted through a search result; the page was not directly reachable from the sandbox). A `Kokkos::parallel_for` is host code: it builds a functor, picks a policy and launches.
- hiprtc "pre-defines HIP device API functions, HIP specific types and macros for device compilation, but doesn't include standard C/C++ headers by default. Users can only include header files provided to `hiprtcCreateProgram`" ([HIP docs, hip_rtc.rst](https://github.com/ROCm/HIP/blob/develop/docs/how-to/hip_rtc.rst)). Kokkos headers pull in a large part of the standard library.
- Inference: getting `Kokkos_Core.hpp` through NVRTC or hiprtc is not realistic. Small device-only helpers (math functions, an index computation) can be shared as plain headers passed to the RTC compiler, but not Kokkos itself.
- On SYCL, Kokkos does rely on JIT, but only the vendor's device JIT of SPIR-V: `Kokkos_ARCH_INTEL_GEN` "enables Just-In-Time compilation for Intel GPUs whereas all the other flags for Intel compilers request Ahead-Of-Time compilation" ([Kokkos configuration guide](https://github.com/kokkos/kokkos-core-wiki/blob/main/docs/source/get-started/configuration-guide.rst)). This is JIT of code that already existed at build time, not run-time code generation.

## 2. Kokkos interop hooks a JIT layer can use

- Execution spaces wrap and expose native streams and queues ([Kokkos_Cuda.hpp](https://github.com/kokkos/kokkos/blob/develop/core/src/Cuda/Kokkos_Cuda.hpp), [Kokkos_HIP.hpp](https://github.com/kokkos/kokkos/blob/develop/core/src/HIP/Kokkos_HIP.hpp), [Kokkos_SYCL.hpp](https://github.com/kokkos/kokkos/blob/develop/core/src/SYCL/Kokkos_SYCL.hpp)):
  - `explicit Cuda(cudaStream_t)` and `cudaStream_t cuda_stream() const`
  - `explicit HIP(hipStream_t)` and `hipStream_t hip_stream() const`
  - `explicit SYCL(const sycl::queue&)` and `sycl::queue& sycl_queue() const`
- Unmanaged Views wrap external pointers, and `View::data()` gives the raw pointer; the layout must be stated explicitly ([Kokkos Interoperability guide](https://github.com/kokkos/kokkos-core-wiki/blob/main/docs/source/ProgrammingGuide/Interoperability.md)).
- So a JIT kernel can take pointers from Kokkos Views and be launched with `cuLaunchKernel` / `hipModuleLaunchKernel` on the stream of a Kokkos execution space instance, or submitted to the `sycl::queue` of `Kokkos::SYCL`. Stream order then keeps the JIT kernel ordered with the surrounding Kokkos kernels.
- On CPU, a JIT-compiled loop body (a function pointer `f(begin, end, args)`) can be called from inside a `Kokkos::parallel_for` over chunks with the OpenMP or Threads backend. Kokkos keeps ownership of the thread pool; LLVM vectorizes the inner loop. (Design inference, not a documented Kokkos pattern.)

## 3. Run-time compilers per backend

### NVIDIA: NVRTC, nvJitLink, driver PTX JIT

- NVRTC is a library (`libnvrtc.so`, `libnvrtc-builtins.so`, `nvrtc64_*.dll`), "does not depend on any other libraries or headers from the CUDA toolkit, and can be run on a system without a GPU". It exists so that users do not have to install nvcc, and avoids "spawning separate processes, disk I/O" ([NVRTC docs](https://docs.nvidia.com/cuda/nvrtc/index.html), via search result).
- Output is PTX or CUBIN, loaded with `cuModuleLoadData(Ex)`. NVRTC has no linker; linking is done by nvJitLink or the driver's `cuLink*` (same source).
- CUDA 12.8 added precompiled headers to NVRTC (`--pch`, automatic PCH, `nvrtcGetPCHCreateStatus`) ([NVRTC 12.8 index](https://docs.nvidia.com/cuda/archive/12.8.2/nvrtc/genindex.html)). CuPy exposes it as `CUPY_NVRTC_USE_PCH` and says PCH can "drastically speed up compilation when many kernels are compiled" ([CuPy env vars](https://github.com/cupy/cupy/blob/main/docs/source/reference/environment.rst)).
- Bypass option: generate PTX text directly (or from LLVM NVPTX) and hand it to `cuModuleLoadData`, which runs the driver's PTX-to-SASS compiler. Dr.Jit does this: "users don't need to install the CUDA SDK, just having an NVIDIA graphics driver is enough" ([Dr.Jit-Core README](https://github.com/mitsuba-renderer/drjit-core)).
- Jitify (NVIDIA, BSD-3) wraps NVRTC, nvJitLink and the driver, handles template instantiation by name, JIT-safe standard headers, and in-memory plus on-disk caches ([Jitify README](https://github.com/NVIDIA/jitify)).

### AMD: hiprtc and comgr

- hiprtc compiles HIP source to a code object, can run without GPU drivers when `--gpu-architecture` is given, and "Developers can bundle this library with their application" ([HIP docs, hip_rtc.rst](https://github.com/ROCm/HIP/blob/develop/docs/how-to/hip_rtc.rst)).
- It relies on comgr, "a shared library that incorporates the LLVM/Clang compiler". Bitcode ingestion is only guaranteed when the bitcode comes from the same ROCm version (same source).
- `-fgpu-rdc` yields LLVM bitcode, linkable with `hiprtcLink*`. Template kernels are reachable through `hiprtcAddNameExpression` / `hiprtcGetLoweredName` (same source).
- Since ROCm 7.0 hiprtc is a separate library from the HIP runtime (same source).

### Intel and SYCL

- SYCL 2020 core has no compile-from-source API. DPC++ adds it through `sycl_ext_oneapi_kernel_compiler`: create a `kernel_bundle` from a source string in "free function kernel" syntax, build it, then enqueue with standard SYCL. Its status: "experimental ... Shipping software products should not rely on APIs defined in this specification" ([extension spec](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_kernel_compiler.asciidoc)).
- DPC++ notes on that extension: AOT targets (`-fsycl-targets=`) are rejected, `--auto-pch` and `--persistent-auto-pch=<dir>` cache the preamble, and the persistent kernel cache is enabled by `SYCL_CACHE_PERSISTENT=1` (same source).
- A companion extension takes a SPIR-V binary instead of source, supported on the Level Zero and OpenCL backends, also experimental ([kernel_compiler_spirv](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_kernel_compiler_spirv.asciidoc)). With LLVM's SPIR-V backend (a regular, non-experimental LLVM target, see [llvm/CMakeLists.txt](https://github.com/llvm/llvm-project/blob/main/llvm/CMakeLists.txt)) this gives a path that does not depend on the DPC++ front end at run time.
- AdaptiveCpp's generic SSCP flow embeds LLVM IR in the binary and lowers it at run time to PTX, amdgcn or SPIR-V; it "allows implementing JIT-based features such as runtime kernel fusion in a backend-independent way" ([AdaptiveCpp compilation doc](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/compilation.md)). Kokkos' SYCL backend targets DPC++, though, so this is a reference design rather than something tensorlib gets through Kokkos.

### CPU: LLVM ORC

- ORCv2 (since LLVM 7) is a modular JIT API: links COFF, ELF and MachO objects in process or out of process, eager or lazy, with concurrent compilation and concurrent JIT'd code, and removable code. `LLJIT` is the ready-made entry point ([ORCv2 design doc](https://github.com/llvm/llvm-project/blob/main/llvm/docs/ORCv2.md)).
- Platform caveat: "MachO and ELF tend to have better support than COFF" (same source). Windows/MSVC is the weakest CPU target.
- `llvm::ObjectCache` (`notifyObjectCompiled`, `getObject`) is the hook for a persistent object cache ([ObjectCache.h](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ExecutionEngine/ObjectCache.h)); the ORC doc notes that mapping external symbols as absolute symbols lets cached objects be reused across sessions.
- LLVM targets the host's SIMD units directly. Kokkos SIMD types are not available inside JIT code, but LLVM's vectorizers replace them. Dr.Jit's CPU backend uses "vectorized LLVM IR and parallel execution using a thread pool" (Dr.Jit-Core README).

### MLIR

- The `gpu` dialect has a full pipeline from `gpu.module` to `gpu.binary`, with NVVM, ROCDL (and SPIR-V) targets; `gpu-module-to-binary` serializes one object per target ([MLIR GPU dialect](https://github.com/llvm/llvm-project/blob/main/mlir/docs/Dialects/GPU.md)).
- Serialization still calls external tools unless built otherwise: the NVVM target runs `ptxas` and `fatbinary` as subprocesses unless MLIR is built with `MLIR_ENABLE_NVPTXCOMPILER` (in-process `nvPTXCompiler` library) ([NVVM Target.cpp](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Target/LLVM/NVVM/Target.cpp)). The ROCDL target writes temporary files and runs `ld.lld` from the ROCm toolkit ([ROCDL Target.cpp](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Target/LLVM/ROCDL/Target.cpp)). Both matter on compute nodes (subprocesses, writable temp dir, toolkit path).
- MLIR already has fusion and tiling passes (linalg, vector), but it is a large, fast-moving dependency with no stable API. Proteus uses it as an optional frontend (`PROTEUS_ENABLE_MLIR`).

## 4. First-execution latency

| Step | Measured cost | Source |
|---|---|---|
| Build the kernel IR/PTX from a traced graph | 2.4 us for a small kernel; 1-15 ms "in typical cases" | Dr.Jit-Core README; [Dr.Jit paper](https://arxiv.org/abs/2202.01284) Fig. 3 |
| Driver PTX to SASS, small kernel | 33.4 ms ("a factor of 1000-10000x slower than the preceding steps") | Dr.Jit-Core README |
| Parse heavy C++ headers (Eigen) before any code | 2.58 s baseline | [ClangJIT paper](https://arxiv.org/abs/1904.08555) Sec. 4 |
| Backend compile of large kernels | grows "super-linearly with kernel size", can be "longer than the kernel execution itself" | Dr.Jit paper |

Further signals: JAX only writes an entry to its persistent cache if compilation took more than 1 s by default ([JAX persistent cache](https://github.com/jax-ml/jax/blob/main/docs/persistent_compilation_cache.md)). Proteus reports that its JIT recovers its own compilation overhead in end-to-end runs, with speedups of up to 2.8x on AMD and 1.78x on NVIDIA versus AOT, and 1.23x over Jitify on average (CGO 2025 abstract, [conference page](https://2025.cgo.org/details/cgo-2025-papers/24/Proteus-Portable-Runtime-Optimization-of-GPU-Kernel-Execution-with-Just-In-Time-Comp)); the paper's full text was not read.

Consequences for tensorlib:

- Generated code must be lean: no big headers in the JIT unit. Generating IR (or PTX / minimal C) directly keeps the front-end cost near zero; source strings that include heavy headers pay seconds per kernel unless PCH is used.
- Fusion granularity has to be bounded, since backend time is super-linear in kernel size.
- Time-stepping codes repeat the same graphs, so the cache hit rate should be high after the first step. Ad hoc interactive use will feel the cold start.

## 5. Kernel caches

Every system surveyed uses an in-memory cache plus an on-disk cache:

| System | Disk cache default | Notable behaviour |
|---|---|---|
| CUDA driver (PTX JIT) | `~/.nv/ComputeCache`, `CUDA_CACHE_PATH`, `CUDA_CACHE_MAXSIZE` up to 4 GiB | [CUDA env vars](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html) (via search result) |
| comgr / hiprtc | `$XDG_CACHE_HOME/comgr` | Cache is disabled when the default location is on a remote file system such as NFS "to avoid concurrency issues on shared storage" ([comgr README](https://github.com/ROCm/llvm-project/blob/amd-staging/amd/comgr/README.md)) |
| DPC++ | `$XDG_CACHE_HOME/libsycl_cache`, off by default | LRU eviction, 8 GB default cap; programs built from source were not cached by the classic cache ([KernelProgramCache.md](https://github.com/intel/llvm/blob/sycl/sycl/doc/design/KernelProgramCache.md)) |
| AdaptiveCpp | `$HOME/.acpp` | `ACPP_RT_NO_JIT_CACHE_POPULATION=1` exists "in an MPI context, where it is sufficient that only one process among many populates the cache" ([env vars](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/env_variables.md)) |
| JAX | user-set directory | All ranks compile on a cold cache, only rank 0 writes; the cache must be on a shared file system or other ranks recompile next run; cache can be pre-populated on one node; the cache "is considered trusted" because it holds executable code (JAX doc above) |
| CuPy | `~/.cupy/kernel_cache` | in-memory mode `CUPY_CACHE_IN_MEMORY` to avoid disk I/O (CuPy doc) |
| Dr.Jit | `~/.drjit` | in-memory then on-disk (Dr.Jit-Core README) |

Design points that follow:

- Key = hash of the canonical graph (ops, dtypes, layouts, strides class, broadcast pattern), target architecture, code generator version, LLVM/toolkit version and options.
- Cache directory configurable, with a fall-back to memory only if it is missing or read-only.
- One writer per job (rank 0, or one per node with a node-local directory), many readers. Atomic rename on write.
- A "warm-up" mode: run the application once (or on one node) to fill the cache, then ship the cache read-only with the job. JAX documents exactly this.
- The cache holds executable code: trust and permissions must be documented.

## 6. Supercomputer constraints

- **No compiler on compute nodes.** NVRTC, hiprtc/comgr and an LLVM linked into the library are all in-process libraries, so they work. The approaches that spawn `nvcc`, `g++`, `cmake` (PyKokkos, Kokkos.jl) or tools such as `ptxas` and `ld.lld` (MLIR defaults) do not, or need the toolkit visible on the node.
- **Read-only or shared file systems.** Default cache locations are all under `$HOME`. comgr turns its cache off on NFS; AdaptiveCpp and JAX added MPI-specific knobs because of many ranks writing the same cache. Temporary files (MLIR ROCDL, NVRTC PCH) need a writable `TMPDIR`.
- **Version coupling.** hiprtc bitcode compatibility requires matching ROCm versions; NVRTC output is tied to the driver that loads it; MLIR and LLVM have no stable API. Bundling a pinned LLVM in tensorlib trades a large dependency for control.
- **Startup storms.** Thousands of ranks compiling the same kernels at start is wasted time; compile on one rank and broadcast the binary over MPI is a possible design (not found documented in the sources read here).
- (Not verified from a primary source here: which centres mount `$HOME` read-only on compute nodes. Treat it as a requirement to support, not a documented fact.)

## 7. Existing projects that combine a portability layer with a JIT

- **PyKokkos** (Kokkos project): translates Python kernels to Kokkos C++, then runs `cmake` and `g++` or `nvcc_wrapper` in a subprocess and caches the module ([cpp_setup.py](https://github.com/kokkos/pykokkos/blob/main/pykokkos/core/cpp_setup.py)). It also has a tracing layer that records operations with data dependencies and fuses them (`pykokkos/core/fusion/`, [trace.py](https://github.com/kokkos/pykokkos/blob/main/pykokkos/core/fusion/trace.py)). Closest to "Kokkos + JIT", and shows the cost: full toolchain at run time.
- **Kokkos.jl**: compiles a wrapper library with CMake at initialization, then compiles "most Kokkos functions (and views) ... separately on demand", caching the shared library ([README](https://github.com/Keluaa/Kokkos.jl)).
- **ClangJIT** (LLNL/ANL): run-time template instantiation in C++, demonstrated with RAJA (a Kokkos-like layer), CUDA support through serialized ASTs ([arXiv:1904.08555](https://arxiv.org/abs/1904.08555)). Research Clang fork.
- **Proteus** (LLNL): LLVM-based JIT library for host, CUDA and HIP, with annotation, C++ string, LLVM IR, MLIR and DSL frontends, and in-memory plus persistent caches ([README](https://github.com/Olympus-HPC/proteus)). Apache-2.0 with LLVM exception. The most direct building block if tensorlib wants an IR-to-GPU path without writing it.
- **Dr.Jit**: tracing JIT that fuses array operations into one kernel, emits PTX (driver JIT) or vectorized LLVM IR (CPU thread pool) or Metal ([README](https://github.com/mitsuba-renderer/drjit-core), [paper](https://arxiv.org/abs/2202.01284)). Closest to tensorlib's "run-time graph to fused kernel" model, but no AMD or Intel GPU.
- **ArrayFire**: builds an AST of element-wise operations and generates one kernel at evaluation time ([JIT doc](https://github.com/arrayfire/arrayfire/blob/master/docs/pages/jit.md)).
- **AdaptiveCpp SSCP**: single IR lowered at run time to all three GPU vendors, with persistent JIT cache and run-time specialization ([docs](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/compilation.md)).
- **CuPy / JAX / Triton**: NVRTC/hiprtc-based, XLA-based and MLIR-based JITs, all with disk caches; Triton calls `ptxas` ([Triton README](https://github.com/triton-lang/triton)).

None of them generates Kokkos code at the speed needed for per-expression fusion. The mature systems all go below the portability layer to an IR.

## 8. Recommendation for the map

- JIT is feasible on all four targets, alongside Kokkos rather than through it. Kokkos stays the owner of memory, streams and the CPU thread pool; the JIT owns the fused kernel bodies.
- Preferred shape: one internal kernel IR, lowered by an in-process LLVM (ORC for CPU; NVPTX to PTX to driver; AMDGPU to code object; SPIR-V to Level Zero / SYCL interop). This removes any need for a compiler on compute nodes and keeps a single code generator. Evaluate Proteus' LLVM IR frontend before writing this from scratch.
- Source-string backends (NVRTC, hiprtc) are a reasonable first prototype for one GPU vendor, since they are easier to debug.
- The SYCL source path is experimental in DPC++; target SPIR-V instead.
- Build the two-level cache, the single-writer policy and the warm-up/read-only mode into the design from the start.
- Keep the template path (expression templates compiled ahead of time) as the zero-latency fallback when no JIT is available or allowed. Windows (COFF in ORC) and macOS (CPU only) are the weakest JIT platforms.

## Source access note

From the sandbox only GitHub and arXiv were reachable. NVIDIA documentation (NVRTC, CUDA environment variables) was read through search-engine excerpts of the official pages, not fetched directly. Everything else was read from the primary file linked.
