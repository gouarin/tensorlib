# Test and benchmark infrastructure

Type: grilling
Status: resolved

## Question

Decide how the implementation is tested and measured, so that the spec can ask for it from the first implementation ticket:

- correctness oracles: the serial interpreter backend of [Backend interface](10-backend-interface.md), NumPy as a reference for Array API semantics, and how results are compared (bit-exact under `TL_FP=strict`, tolerances otherwise);
- generated tests: a random expression and window generator, starting from the one in [Prototype of compile times for lighter templates](11-compile-time-prototype.md), run against every backend, with broadcasting, views, segments and poison;
- multi-rank tests: rank counts, the `self` communicator, and checks that every rank builds the same plan ([Communication in the evaluation plan](28-communication-in-the-evaluation-plan.md));
- the CI matrix across compilers, operating systems and GPUs, the debug checks of [Error reporting and debug mode](24-error-reporting-and-debug-mode.md), and sanitizers;
- benchmarks: Google Benchmark with interleaved repetitions, the samurai and ponio reference workloads from the Notes, where they run, and how regressions are detected;
- compile-time measurements and the header whitelist check of [Minimum C++ standard and build system](14-cxx-standard-and-build.md).

## Answer

Decided with the user on 2026-10-31. No ADR: these are tooling choices that can change without breaking the code.

- **Oracles.**
  - The serial backend ([Backend interface](10-backend-interface.md)) judges everything about code generation: lowering, fusion, segments, chunking.
  - Array API semantics (type promotion, broadcasting, slicing, edge cases) are checked independently of our own tables: a small test-only Python shim over `<tl/c.h>` ([C ABI and binding surface](31-c-abi-and-binding-surface.md)) runs the official **array-api-tests** suite and comparisons with NumPy in CI. The shim is not a supported binding, and it exercises the C ABI end to end.
  - Hand-written expected values serve the documentation examples and numerical cases (analytic solutions for stencils, order of convergence).
- **Comparison.**
  - Bit-exact under `TL_FP=strict` for everything built only from correctly rounded IEEE operations (`+ - * /`, `sqrt`, explicit `fma`, conversions, comparisons, canonical reductions, deterministic scatter), across backends and thread counts.
  - Transcendental functions get a documented ulp bound per function and per target, checked against extended precision (MPFR or `long double`) on a sample.
  - In the default mode, the tolerance is a number of ulp proportional to the kernel's contractible operations, which the generator knows.
  - The reproducibility promises of [Code-generation strategy](12-code-generation-strategy.md) and [Communication in the evaluation plan](28-communication-in-the-evaluation-plan.md) (same bits for any thread count, and at a fixed rank count under strict) are bit-exact tests.
- **Generated tests.**
  - A tensorlib-specific generator, grown from the one in [Prototype of compile times for lighter templates](11-compile-time-prototype.md), produces random windows that mix:
    - element-wise operations and reductions;
    - broadcasts, slices and stencil shifts;
    - views and partial writes, in-place updates with aliasing;
    - tensors dead before evaluation;
    - segment spaces with interval lengths taken from the real samurai meshes of [Lowering of short segments](30-lowering-of-short-segments.md);
    - injected poison.
  - Each window runs on every backend and is compared with the serial backend under the rule above.
  - On failure, the generator shrinks the program (drops statements, simplifies shapes) to the smallest one that still fails and prints it as readable C++, ready to become a regression test.
  - Each PR runs a fixed budget with fixed seeds. A nightly campaign runs long with fresh seeds.
- **Multi-rank tests.**
  - A test implementation of `tl::communicator` runs every rank as a thread in one process, which [ADR 0010](../../../docs/adr/0010-persistent-communication-patterns-and-a-progress-thread.md) makes possible. It runs under ASan, UBSan and TSan without MPI, and injects faults: slow ranks, reordered completions, corrupted headers. Most multi-rank tests and the SPMD version of the generator use it on every PR.
  - Real MPI (MPICH and Open MPI, 1 to 4 ranks, oversubscribed) runs a subset on every PR. 8 and 16 ranks and GPU-aware MPI run nightly.
  - Cases include `tl_communicator_self`, uneven blocks (3 ranks), a 2×2 grid with corners, empty local parts, periodicity, wide ghosts and diagonal stencils, poison on one rank, and deliberately divergent SPMD programs that must raise `tl::spmd_error`.
  - Debug checks (plan hash, same bits on every rank) are on in every multi-rank test, and every multi-rank test has a timeout.
- **CI matrix.**
  - Every PR, on hosted runners:
    - Linux with GCC at the floor (10.4) and latest, Clang at the floor (14) and latest;
    - macOS with Apple Clang;
    - Windows with MSVC at the floor and latest.
  - Two builds: serial only (no LLVM, no Kokkos) and Kokkos OpenMP with LLVM.
  - ASan and UBSan on the serial and Kokkos OpenMP builds; TSan on the threaded communicator and the progress thread.
  - The whole suite once under `TL_DEBUG=all` on the real backend ([Error reporting and debug mode](24-error-reporting-and-debug-mode.md)).
  - CUDA (nvcc 12.2), HIP (ROCm 6.2) and SYCL (IntelLLVM) compile-only in containers.
  - GPU execution runs on self-hosted runners (an NVIDIA machine, AMD if possible), nightly and on PRs labelled `gpu`. Periodic campaigns in computing centres (GitLab CI with Jacamar, or batch jobs) cover multi-node MPI, AMD and Intel GPUs, and the calibration of GPU defaults. The machines and allocations are an infrastructure need for the spec to state; they are not secured yet.
- **Benchmarks.**
  - Google Benchmark with interleaved repetitions, medians and CV.
  - The reference workloads of the map's Notes rewritten with the tensorlib API on real data:
    - samurai: `advection_2d`, `burgers`, LBM Rayleigh-Taylor, `benchmark_fv_scheme`, `benchmark_update_fields`, `benchmark_detail_computation`;
    - ponio: `heat_bench` with `rk_44`, `rkc_202`, ROCK2/4 on 10^6 cells.
  - Micro-benchmarks: node recording, window evaluation, cold compile and cache reload, multi-output fusion, reductions, scatter, ghost exchange.
  - Every result is reported against a machine ceiling measured beside it (bandwidth by stream count, FMA peak, storage-span ceiling for segments), as in prototypes 27 and 30.
  - A dedicated physical benchmark machine (the GPU runner when it is idle) runs the suite nightly and archives the results (a data branch or a service such as Bencher). It alerts when a regression is significant (Mann-Whitney on interleaved repetitions) and above 5 %.
  - A PR labelled `bench` runs an A/B in one job: base and PR built and measured interleaved on the same machine.
  - Instruction counts (Cachegrind) on hosted runners guard the front end on every PR (node recording, about 5 ns per node).
  - Once samurai and ponio are ported, a nightly job builds their tensorlib branches against `main` and runs their own benchmarks, as the final acceptance test.
- **Tools and compile time.**
  - GoogleTest, as in Kokkos and samurai.
  - A CI script preprocesses `<tl/tl.hpp>` and each component header and fails on a standard header outside the whitelist of [Minimum C++ standard and build system](14-cxx-standard-and-build.md). It also checks the one-line facade rule of [ADR 0011](../../../docs/adr/0011-thin-templates-over-an-erased-core-and-layered-extension-points.md).
  - A reference translation unit generated with 64 statements has a budget on template instantiations and front-end time measured with Clang `-ftime-trace`, which are stable on shared runners. The threshold is set at the first implementation, plus 10 %.
- **Layout.**
  - `tests/` mirrors `src/`.
  - CTest labels (`unit`, `generated`, `mpi`, `gpu`, `bench`, `abi`, `array_api`) select what each CI job runs.
  - Every fixed bug adds a regression test, often printed by the generator.
- **From the first ticket.** The serial oracle, the comparison rule, GoogleTest and the header check exist before the first operation. The other pieces arrive with the feature they test:
  - the threaded communicator with the first distributed code;
  - the Python shim with the C ABI;
  - the benchmark machine with the first fused kernel.

Grilling log (2026-10-31):

- Q1, oracles: option B. Serial backend for code generation; a test-only Python shim over the C ABI runs array-api-tests and NumPy comparisons; hand-written values for examples and numerical cases.
- Q2, comparison: option A. Bit-exact under `TL_FP=strict` for correctly rounded IEEE operations; ulp bound per function and target for transcendentals; default mode tolerance proportional to contractible operations; reproducibility promises tested bit-exact.
- Q3, generated tests: option A. Tensorlib-specific window generator with shrinking to readable C++ and reproducible seeds; fixed budget per PR, long nightly campaign.
- Q4, multi-rank: option B. Threaded test communicator with fault injection and sanitizers for most tests; real MPICH and Open MPI at 1 to 4 ranks per PR; 8, 16 ranks and GPU-aware MPI nightly.
- Q5, CI matrix: hosted runners per PR as listed; GPU execution on self-hosted runners nightly and on the `gpu` label (option A), plus periodic campaigns in computing centres (option C).
- Q6, benchmarks: option A plus B. Dedicated physical benchmark machine, nightly archive and alert on significant regressions above 5 %, same-job A/B on the `bench` label; instruction counts per PR for the front end.
- Q7, tools: GoogleTest; header whitelist and facade rule checked by preprocessing; compile-time budget on instantiations and front-end time with `-ftime-trace`.
- The user confirmed the recap, including the layout and the order in which pieces arrive.

## Comments

2026-10-31: the host floor rose to GCC 11.1, Clang 15 and Apple Clang 15 ([Minimum C++ standard and build system](14-cxx-standard-and-build.md), amendment on the floor). The CI floor jobs follow.
