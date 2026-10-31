# PETSc and Tpetra APIs for zero-copy interop

Facts gathered on 2026-10-30 for [Zero-copy interop with PETSc and Tpetra](../issues/17-petsc-tpetra-interop.md). Sources are PETSc 3.26.0 (release branch, read on the GitHub mirror of gitlab.com/petsc/petsc), Trilinos master (18.0-dev), and the main Tpetra constructors re-checked on `trilinos-release-16-2-2`. PETSc paths are relative to `https://gitlab.com/petsc/petsc/-/blob/release/`, Trilinos paths to `https://github.com/trilinos/Trilinos/blob/master/`.

## PETSc

**Creating a Vec on user memory.**
- Available: `VecCreateSeqWithArray`, `VecCreateMPIWithArray` and `VecCreateGhostWithArray`; `VecCreate{Seq,MPI}{CUDA,HIP}WithArray` (and `...WithArrays`, which take both a host and a device pointer); `VecCreate{Seq,MPI}KokkosWithArray`. The Kokkos ones take a raw device pointer, not a `Kokkos::View`.
- PETSc 3.26 adds `VecCreate{Seq,MPI}WithArrayAndMemType` for standard, CUDA and HIP vectors, and lets the Kokkos creators take NULL, to be filled later with `VecKokkosPlaceArray`.
- PETSc never frees an array given to a `*WithArray` creator.
- Sources: `include/petscvec.h`, `src/vec/vec/impls/mpi/pbvec.c`, `src/vec/vec/impls/mpi/kokkos/mpikok.kokkos.cxx`, `src/vec/vec/utils/veccreatememtype.c`, `doc/changes/326.md`.

**Borrowing a Vec's memory.**
- `VecGetArray{,Read,Write}AndMemType` (3.15, Write in 3.17) returns the pointer and its memory type. For device vectors the pointer is always on the device, and data is copied there first if needed.
- `VecPlaceArray`/`VecResetArray` and `Vec{CUDA,HIP,Kokkos}PlaceArray` place a user array without taking ownership. `VecReplaceArray` takes ownership and needs `PetscMalloc` memory.
- `VecLockReadPush`/`Pop` makes writes an error while held.
- Source: `src/vec/vec/interface/rvector.c`.

**Asynchrony.**
- Since 3.16, device work is ordered by `PetscDeviceContext`, with one stream per context. The default stream type maps to the CUDA/HIP legacy null stream, and PETSc's Kokkos execution space is built on it.
- There is no public API to hand PETSc a stream. It can only be read (`PetscDeviceContextGetStreamHandle`). The docs advise synchronising immediately before and after foreign use.
- Sources: `include/petscdevice.h`, `src/sys/objects/device/interface/dcontext.cxx`, `src/sys/objects/kokkos/kinit.kokkos.cxx`.

**DMDA.**
- Storage order is `[k][j][i*dof + c]`: dof fastest, then i, j, k.
- A global vector holds each rank's owned block, contiguous and without ghosts, in PETSc's parallel numbering. A local vector holds the owned block plus the ghost band (`gxm*gym*gzm*dof`).
- `DMDACreate3d` takes `m, n, p`, `lx, ly, lz`, the stencil width `s`, `DMDA_STENCIL_STAR|BOX` and `DM_BOUNDARY_PERIODIC`.
- There is no `DMDA...WithArray` creator. A global vector on user memory comes from `DMCreateGlobalVector` plus `VecPlaceArray`, or from `VecCreateMPIWithArray` plus `VecSetDM`.
- Sources: `src/dm/impls/da/da3.c`, `dagetarray.c`, `dalocal.c`, `dadist.c`.

**Mat.**
- `MatCreateDense` stores column-major data distributed by rows (with `MatDenseSetLDA`).
- `MatCreateMPIAIJWithArrays` copies its arrays. `MatCreateSeqAIJWithArrays` does not.
- `MatCreateSeqAIJKokkosWithKokkosViews` (3.23) wraps device views, but copies `i` and `j` to the host and does not fence.
- `MatCreateShell` exists.
- Sources: `src/mat/impls/dense/`, `src/mat/impls/aij/`.

**PetscSF.** It describes arbitrary root/leaf graphs, and `PetscSFBcastWithMemTypeBegin` works on raw user buffers. Source: `include/petscsf.h`.

**Communicators.** `PetscCommDuplicate` caches an inner duplicate on a user communicator and uses it for PETSc traffic. Source: `src/sys/objects/tagm.c`.

## Tpetra

**MultiVector and Vector.**
- `MultiVector(map, dual_view_type)` views the data without copying. Storage is `Kokkos::DualView<T**, LayoutLeft>`, column-major, with a column stride of at least the local row count.
- `MultiVector(map, t_dev)` also views, but calls `create_mirror_view` and so allocates a host mirror unless the memory is host-accessible.
- `Vector` has the DualView constructors and no `t_dev` one. The `ArrayView` constructors copy.
- Source: `packages/tpetra/core/src/Tpetra_MultiVector_decl.hpp`, `Tpetra_Vector_decl.hpp`.

**Map.**
- The contiguous constructor takes local sizes, and the non-contiguous one takes a list of global indices.
- Ghosting uses an overlapping Map with `Import`/`Export`.
- `offsetView` gives the owned part without copying only when the owned rows come first in local numbering.
- Source: `Tpetra_Map_decl.hpp`.

**CrsMatrix.** The constructors from a `KokkosSparse::CrsMatrix` local matrix view it and come out fill-complete. Source: `Tpetra_CrsMatrix_decl.hpp`.

**Kokkos.** Trilinos 17.2 bundles Kokkos and Kokkos Kernels 5.2.1, and can build against an installed Kokkos (`TPL_ENABLE_Kokkos=ON`). The application and Tpetra must share one Kokkos.

**Communicators.** `Teuchos::MpiComm<int>(MPI_Comm)` wraps the communicator without duplicating it. Source: `packages/teuchos/comm/src/Teuchos_DefaultMpiComm_decl.hpp`.
