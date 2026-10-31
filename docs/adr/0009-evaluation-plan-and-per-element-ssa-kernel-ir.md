# Evaluation plans over kernels written as per-element SSA programs

Evaluating a pending window produces two levels. The evaluation plan orders kernels and the nodes that cannot be fused: library operations, Kokkos interop kernels, PETSc and Tpetra operators, ghost exchanges and reductions across ranks. A kernel is written in the internal IR as a per-element SSA program over an iteration space, rectangular or segmented. It has typed operands read through affine maps or gathers, run-time arguments, SSA operations that each carry their source site, and three kinds of outputs: stores (masked or not), reductions and scatters. A kernel may hold several bodies, and each segment of its iteration space names the body it runs. The IR has no loops, tiles or shared memory: the lowering for each target decides them from the backend's capabilities.

We rejected an IR with explicit loop nests (TVM TIR, MLIR affine). It would let the core tile once for every target, but it is harder to hash canonically, it makes the serial interpreter and every lowering larger, and tiling depends on the target anyway. We rejected adopting MLIR (linalg, vector, gpu dialects): its passes are attractive, but it is a heavier dependency than LLVM with no stable API, and the dependency-free serial interpreter, our test oracle, could no longer read the IR.

## Consequences

- The IR is small enough to hash incrementally while recording, to interpret, and to print and parse as text. Tests of passes and lowerings are written as text files run through `tl-opt` and FileCheck.
- GPU lowerings must recognise stencils from the shifts in the maps to tile them in shared memory, since nothing in the IR asks for it.
- Each lowering handles one operation at a time, which is also where debug checks are inserted.
- Communication and library calls never enter a kernel, so a backend can compile kernels without knowing about MPI, PETSc or Kokkos Kernels.
