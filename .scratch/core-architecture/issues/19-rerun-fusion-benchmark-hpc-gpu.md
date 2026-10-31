# Rerun the fusion benchmark on an HPC node and a GPU

Type: task
Status: claimed
Blocked by: 09

## Question

[Prototype of multi-output fusion](09-multi-output-fusion-prototype.md) was measured only on an 18-core aarch64 VM, with no GPU. Rerun it where the numbers count, and record them in the prototype's `results.txt`. This is HITL: the user runs it on machines the agent cannot reach, and the agent supplies the build and the GPU kernel variant.

Use `run.sh` from the prototype: Google Benchmark with interleaved repetitions. Leave one or two cores free when the node is shared or virtualized.

Checklist:

- [ ] an x86 HPC node (and a Grace node if one is available), with the thread count swept from 1 to all cores and `OMP_PROC_BIND=spread`, to check whether one core really gains nothing from fusion, as the VM showed;
- [ ] one NVIDIA or AMD GPU (Kokkos CUDA or HIP): compare two loops, the hand-fused loop, and a straight-line kernel generated from the recorded program, in place of the block interpreter, which does not map to a GPU;
- [ ] launch and fence overhead per evaluation on each machine.
- [ ] from [Buffer reuse and allocation](21-buffer-reuse-and-allocation.md): Umpire `ResourceAwarePool` against `QuickPool` on small allocations (samurai-sized fields), and whether staggered start offsets help on the GPU.
- [ ] from [Code-generation strategy](12-code-generation-strategy.md): kernels generated through LLVM NVPTX (driver PTX JIT) or AMDGPU against the hand-fused CUDA or HIP loop, cold compile time per kernel, GPU defaults for `TL_MAX_KERNEL_OPS` and `TL_MAX_KERNEL_LIVE`, and whether turning a cached plan into a CUDA or HIP graph pays off.
- [ ] from [Kernel IR](22-kernel-ir.md): the 32-bit index variant against 64-bit indices on a stencil and a gather, and atomic indirect scatter-add against the deterministic sort path.
- [ ] from [Prototype of the LLVM JIT on CPU](27-llvm-jit-cpu-prototype.md): its `run.sh` and `ceilings.cpp` on x86 (AVX2, AVX-512), CPU defaults of the two ceilings there, and the unexplained 1.9x cost of two launches over one two-body launch at 8M elements.
- [ ] from [Communication in the evaluation plan](28-communication-in-the-evaluation-plan.md): calibrate the default of `TL_OVERLAP_MIN_CELLS` (interior cells above which a kernel is split into interior and boundary launches to overlap a ghost exchange) on x86 and on a GPU.
- [ ] from [Lowering of short segments](30-lowering-of-short-segments.md): `prototypes/30-short-segments/run.sh` on x86 and a GPU; whether bridging still pays there, and whether masked tails (AVX-512, SVE, GPU) beat LLVM's scalar remainder.

The result is the set of numbers that [Code-generation strategy](12-code-generation-strategy.md) needs on real hardware.

## Comments

2026-10-30: no GPU or HPC node is available yet. This task no longer blocks [Code-generation strategy](12-code-generation-strategy.md); it stays open as a later check of that decision and of the device defaults in [Buffer reuse and allocation](21-buffer-reuse-and-allocation.md).
