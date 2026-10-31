# Requirements from samurai and ponio

Type: research
Status: resolved

## Question

tensorlib must give optimal performance to the user's existing libraries [samurai](https://github.com/hpc-maths/samurai) (adaptive mesh refinement and multiresolution on interval-based meshes) and [ponio](https://github.com/hpc-maths/ponio) (time integration: Runge-Kutta, splitting, Lawson, ROCK...). How do they use array libraries today, and what does that require from tensorlib's core?

For each library, cover:

- which array library or libraries it uses (xtensor, Eigen, std containers, its own abstractions), through which layer, and whether that layer is already pluggable;
- the access patterns in its hot loops: interval-based views and slices on cell arrays, stencils across refinement levels, prediction and projection operators, masks, small fixed-size arrays (field components, Butcher tableaux), reductions, and linear combinations such as `u + dt * (a1*k1 + a2*k2 + ...)`;
- the element types and container kinds (fixed-size, fixed-rank, dynamic), the memory layout (SoA or AoS for vector fields), and where shapes are known at compile time;
- how it parallelises today (OpenMP, MPI and its own domain decomposition in samurai, GPU work if any), and how a distributed mesh relates to tensorlib's partitioned tensors (see `CONTEXT.md`);
- known pain points: compile times, run-time overheads of xtensor views, temporaries, anything recorded in issues, PRs, benchmarks or docs;
- the benchmarks or demos that best represent their performance, which could become tensorlib reference workloads.

Finish with a list of concrete requirements for tensorlib, each tied to an open or resolved ticket of this map, and flag anything that contradicts a decision already taken.

## Answer

samurai stores each field as one flat array per rank (AoS for vector fields) over all cells and ghosts of all levels. Its intervals give the offsets, there is no global index space, and every size changes at each adaptation. Its hot loops already left xtensor for raw loops over intervals (detail computation 7.7 s to 1.5 s, LBM stream 11.8 s to 0.36 s).
ponio is generic over the state. It needs `+`, scalar `*`, copies, O(1) `swap`, in-place `f(t, y, dy)` and fused combinations of up to 20 stages with run-time coefficients (`rkc_202`).
Requirements: segmented iteration spaces (per-interval offsets, affine inside) in the kernel IR, a kernel cache keyed on symbolic extents, cheap reallocation every step with staggered buffer offsets, scatter-add, fused reductions, `fixed` covering `xtensor_fixed`, and a fast path for per-element user lambdas.
Conflicts: samurai meshes do not map to a partitioned tensor with a regular global index space (samurai fields are local tensors unless ticket 18 adds index-list distributions). The DMDA-style ghost band does not fit interleaved ghosts. One affine map per leaf misses per-interval access, and ADR 0003 leaves most of samurai's physics (user lambdas) outside fusion.
Neither library runs on GPUs today, so tensorlib would be their GPU path.

Findings: [research/25-samurai-ponio-requirements.md](../research/25-samurai-ponio-requirements.md)
