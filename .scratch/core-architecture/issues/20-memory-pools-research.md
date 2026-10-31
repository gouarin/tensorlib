# Memory pools for HPC tensors

Type: research
Status: resolved
Blocked by: 09

## Question

Which existing allocators and memory pools can tensorlib use for tensor buffers and kernel temporaries on CPU and GPU, instead of writing its own? Cover at least Umpire (LLNL), RAPIDS RMM, `cudaMallocAsync` and `hipMallocAsync` stream-ordered pools, SYCL USM, Kokkos memory spaces and `Kokkos::Experimental::MemoryPool`, and how PETSc and Trilinos (Tpetra) allocate. For each: platforms (CPU, CUDA, HIP, SYCL, macOS, Windows), stream-ordered semantics, thread safety, fragmentation behaviour, how it plugs into Kokkos (Kokkos owns memory per [Feasibility of JIT compilation with Kokkos](03-jit-with-kokkos.md)), licence and maintenance. The prototype showed fresh allocations double the time of a large fused kernel ([Prototype of multi-output fusion](09-multi-output-fusion-prototype.md)).

## Answer

Kokkos cannot pool tensor storage: every View free fences, its `cudaMallocAsync`/`hipMallocAsync` options are off by default and fenced anyway, and `Kokkos::MemoryPool` is a fixed-capacity in-kernel pool with 4 KB blocks by default.
Umpire (MIT) is the only pool that covers host, CUDA, HIP and SYCL on Linux, macOS and Windows. Its `ResourceAwarePool` handles several streams, Kokkos streams wrap into camp resources, and LLNL and Kokkos are building `UmpireSpace` on top of it.
RMM is strong but CUDA and Linux only (hipMM is archived). SYCL 2020 has no stream-ordered free. PETSc and Tpetra each keep internal schemes that others cannot reuse.
Recommendation: depend on Umpire (`QuickPool`, then `ResourceAwarePool`, both wrapped in `ThreadSafeAllocator`) as an optional dependency behind the backend allocator interface, and wrap buffers as unmanaged Kokkos Views. Offer CCCL/HIP native pools and RMM as optional adapters.
Gap left for ticket 21: deferring frees until the evaluation's events complete, planning temporaries inside a window, and pool limits that stay deterministic across ranks.

Findings: [research/20-memory-pools.md](../research/20-memory-pools.md)
