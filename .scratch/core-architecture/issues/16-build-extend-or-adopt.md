# Build, extend or adopt

Type: grilling
Status: resolved
Blocked by: 15

## Question

Given the existing libraries found in research, which parts of tensorlib should be built, which should extend an existing project, and which should be adopted as dependencies or left to interoperability (for example linear algebra through PETSc or Trilinos)? Does the answer change the destination of this map?

## Answer

Decided with the user on 2026-10-30, from [Existing HPC libraries that cover part of tensorlib](15-existing-hpc-libraries.md).

- Build: the C++ tensor layer only. That means a NumPy-like API, lazy multi-output fusion, and partitioned tensors over MPI. tensorlib is a new project with a new code base. It takes its API style, names and the three container kinds from xtensor, but not xtensor's evaluation model (expression templates evaluated on assignment) and not xsimd.
- Depend on:
  - `mdspan` (kokkos/mdspan, header only) in the core, as the non-owning view type;
  - Kokkos core and SIMD, Kokkos Kernels and Kokkos-FFT, through the Kokkos backend only, so the core still builds without Kokkos. The serial backend uses simple implementations or delegates to BLAS and FFTW.
- Communication: plain MPI (GPU aware when available) behind a small tensorlib communication interface. Kokkos Comm can replace it once stable, and contributing to it is welcome.
- Interoperate: zero-copy exchange of local parts with PETSc (`Vec`, `Mat`, DMDA) and Trilinos/Tpetra (`MultiVector`, `CrsMatrix`). Ginkgo is optional. tensorlib does not write its own PETSc-like layer.
- Learn from: XLA, Diffuse and DaCe for fusion; Heat, Tpetra `Map` and PETSc DMDA for the partition model. Legate and cuPyNumeric are end of life, so they are study material only.
- Primary user: the C++ HPC developer writing simulation codes (stencils, numerical schemes, coupling with PETSc or Trilinos). Bindings drive the same core from Python, Julia and Rust for steering, prototyping and post-processing. Competing with JAX or Heat as a distributed NumPy for Python is not a goal.
- Effect on the map: the destination now includes the PETSc/Tpetra interop hooks. The PETSc-like layer, distributed FFT and Python-first distributed NumPy are out of scope.
