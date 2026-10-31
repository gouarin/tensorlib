# tensorlib

A C++ tensor library in the spirit of NumPy and xtensor, built on Kokkos for parallel execution on CPUs and GPUs, and designed to run on distributed HPC machines.

## Language

### Scope

**Core**:
The part of the library that defines tensors, expressions, their evaluation and the execution backends. Distribution, language bindings and linear algebra build on it.
_Avoid_: kernel (overloaded with compute kernels), engine

**Backend**:
An implementation of the core's execution interface for a family of hardware. Kokkos is the main backend; the serial backend is the reference used for testing and debugging.
_Avoid_: device, runtime, execution engine

**Kernel**:
One fused unit of work that the core hands to a backend after splitting a pending window, written in the core's internal representation.
_Avoid_: functor, task, loop

**Library operation**:
A named operation that a backend provides from an existing library instead of a generated kernel, such as a matrix product, an FFT or a sort. It cannot be fused.
_Avoid_: primitive, builtin

**Capabilities**:
What a backend declares about its hardware, so that kernels can be specialised for it without changing their meaning.
_Avoid_: features, traits

**Queue**:
An ordered stream of work on one device. Work on one queue runs in order; work on different queues may overlap.
_Avoid_: stream (one backend's name for it), execution space instance

**Event**:
A marker recorded on a queue that later work, on any queue or on the host, can wait for.
_Avoid_: fence, future

### Containers

**Container kind**:
The category of a tensor container, set by how much of its shape is known at compile time. There are three: fixed-shape, fixed-rank and dynamic-rank.

**Fixed-shape tensor**:
A small tensor whose full shape is known at compile time, so its storage lives inline (like `std::array`, or `xtensor_fixed`). Used for local computations and as the element type of larger tensors; never partitioned.
_Avoid_: static tensor, small tensor

**Fixed-rank tensor**:
A tensor whose number of dimensions is known at compile time but whose extents are set at run time (like `xtensor`).
_Avoid_: static-dim tensor

**Dynamic-rank tensor**:
A tensor whose number of dimensions and extents are both set at run time (like `xarray` or a NumPy array).
_Avoid_: dynamic tensor, array

**View**:
A reference to part or all of a tensor's elements, through that owning tensor rather than its raw storage. Copying a view gives another view of the same tensor. Writing through a view writes into the owning tensor, and never into copies of it.
_Avoid_: slice (one way to make a view), alias, reference

**Box**:
A rectangular set of positions, one range per axis, that can be shifted by an offset to read a tensor's neighbours. On a partitioned tensor it is expressed in global indices.
_Avoid_: domain, region, window (the pending window)

**Segment space**:
A sequence of segment lengths shared by several views, so that operations between those views line up segment by segment. Each view built on it adds its own starting offsets and stride.
_Avoid_: interval set (samurai's notion), segment table, iteration space

**Bridging**:
Merging two segments of a kernel whose storage gap is small, so that the kernel also computes the cells of the gap. It is allowed only where the segment space declares its gaps dead: allocated and readable, and undefined after a write.
_Avoid_: merging, fusion (kernel fusion), padding

**Erased tensor**:
A tensor handle whose element type, rank, memory space and layout are known only at run time. It is the form tensors take at the language-binding and code-generation boundaries.
_Avoid_: dynamic tensor, untyped tensor, array

**Memory space**:
Where a tensor's elements live: host memory, pinned host memory, a given GPU's memory, or managed memory shared by host and GPU. It belongs to one backend, and is a run-time property of a tensor.
_Avoid_: device (one kind of memory space), location

**Memory resource**:
Where the storage of one memory space comes from: it hands out and takes back memory in the order of a queue, so memory freed after a kernel can go to later work on that queue without waiting. Every memory space has exactly one.
_Avoid_: allocator (one part of it), pool (one kind of resource), memory manager

**Capacity**:
The number of bytes reserved for a tensor's storage, at least what its elements need. A tensor can take a new shape in place while that shape fits its capacity.
_Avoid_: size (the number of elements), reserved size

**Broadcasting**:
Combining tensors of different shapes by stretching size-1 or missing dimensions to a common shape, without copying. A broadcast result can be read but never written.
_Avoid_: expansion, tiling, repeat

**Index map**:
How one operand of a kernel finds its element for a given position of the result, within one segment: an offset plus one stride per result dimension, zero along a broadcast dimension. Slices, transposes and shifts are index maps too.
_Avoid_: stride (one part of it), accessor, layout

**Segment**:
One contiguous run of a kernel's work, such as one mesh interval, with its own starting offset for every operand. A kernel can run over many segments at once; a plain tensor is a single segment.
_Avoid_: interval (samurai's mesh notion), chunk, block

**Traced function**:
A user function written with tensorlib operations that is run once on placeholder values to record what it computes, so that the recorded operations can be fused like any other expression.
_Avoid_: user kernel (that is interop code), functor

### Evaluation

**Expression**:
A deferred computation over tensors that is never stored. It is recomputed inside each consumer, and captures the values its inputs had when it was built.
_Avoid_: xfunction, lazy tensor, node

**Operand**:
Anything an operation accepts as an argument: a tensor, a view, an expression or a scalar. Every operation takes any mix of operands and treats them alike.
_Avoid_: argument, input

**Pending window**:
The set of deferred operations recorded in one context since the last evaluation.
_Avoid_: queue, batch, stream, graph

**Context**:
What a sequence of tensorlib calls runs in: it owns one pending window and the last error. It is used from one thread at a time and may move from thread to thread. Each host thread has one implicitly, and bindings pass one explicitly.
_Avoid_: session, stream, queue, thread

**Trigger**:
An event that makes a pending window evaluate, such as a host read, an explicit evaluation call, a non-fusible consumer or a resource limit.
_Avoid_: flush, sync point

**Evaluation**:
Turning a pending window into kernels and launching them, which writes every live tensor of the window.
_Avoid_: execution, materialization, flush

**Evaluation plan**:
What evaluating a pending window decides before launching anything: the kernels it is split into, where their temporaries live, and the communication between them (ghost exchanges, reductions across processes, redistributions). A window with the same structure as an earlier one reuses its plan.
_Avoid_: schedule, compiled graph, executable

**Exchange point**:
A place in an evaluation plan where communication starts. Every ghost exchange or reduction across processes placed at the same exchange point shares one message per neighbour, or one collective.
_Avoid_: sync point, barrier

**Communication pattern**:
A communication that an evaluation plan sets up once, with fixed partners and sizes, and starts again at every evaluation that reuses the plan: a ghost exchange, a combined reduction across processes or a redistribution.
_Avoid_: request (one start of a pattern), schedule, message

**Kernel body**:
One of the element-wise programs of a kernel. A kernel has one body, or several when different regions of its iteration space compute different things (interior and boundaries); each segment says which body it runs.
_Avoid_: branch, region (in prose, a part of the iteration space), variant

**Combiner**:
The associative operation, with its identity, that a reduction or a scan uses to merge values: a built-in one such as sum or max, or one traced from a user lambda.
_Avoid_: reducer, monoid, accumulator

**External function**:
A pure scalar function written outside tensorlib and registered under a name, which a kernel can call without breaking fusion.
_Avoid_: user kernel (that is interop code), intrinsic, callback

**Kernel bundle**:
A set of kernels compiled ahead of a run for given targets, shipped with a job and loaded read-only, so the run needs no compilation for the kernels it holds.
_Avoid_: kernel cache (filled during runs), AOT library, precompiled kernels

**Borrow**:
Lending a tensor's storage to another library (PETSc, Tpetra) for a delimited scope. The tensor is up to date when the scope starts, cannot be used by tensorlib meanwhile, and is known to have changed when it ends.
_Avoid_: export, alias, share

**Wrap**:
Seeing another library's storage as a tensor for a delimited scope, the mirror of a borrow. The other library gets its storage back, up to date, when the scope ends.
_Avoid_: adopt, import, adapt

### Errors and diagnostics

**Site**:
The place in the user's source (file, line, function) of the statement that produced a node of a pending window. A binding gives the line of the script that called it.
_Avoid_: location, origin (the rank where an error started), call site

**Check**:
An optional verification or diagnostic that can be turned on for a whole run or for one scope, such as evaluating statement by statement, bounds of index views, NaN produced from finite values, or counting copy-on-write detaches.
_Avoid_: debug level, assertion (compiled into host headers), sanitizer

**Poisoned tensor**:
A tensor that a failed kernel or evaluation should have written. It holds the error, passes it to every result computed from it, and throws it when read.
_Avoid_: invalid tensor, NaN tensor, failed tensor

### Distribution

**Distribution**:
The description of how a tensor's global index space is split across processes. Every tensor has one; the default is local (one process holds everything).
_Avoid_: sharding, decomposition, layout

**Partitioned tensor**:
A tensor whose distribution splits its elements across several processes, each one owning a part of the global index space. It is the same kind of object as any other tensor, not a separate type.
_Avoid_: distributed array, global tensor

**Replicated tensor**:
A tensor whose distribution gives every process of a communicator a full, identical copy. It can be combined with any partitioned tensor on that communicator.
_Avoid_: global copy, broadcast tensor

**Local tensor**:
A tensor whose distribution is local: private to one process, with no promise that other processes hold the same values.
_Avoid_: serial tensor, non-distributed tensor

**Redistribution**:
An explicit operation that moves a tensor's data from one distribution to another.
_Avoid_: reshuffle, repartition

**Local part**:
The portion of a partitioned tensor owned by one process.
_Avoid_: local view, chunk, block

**Process grid**:
The arrangement of a communicator's processes in a Cartesian grid, one dimension per tensor axis, along which a block distribution splits the tensor.
_Avoid_: topology, mesh (samurai's and PETSc's meaning)

**Ownership range**:
The range of global indices along one axis that a process owns.
_Avoid_: bounds, extent (the size of an axis)

**Ghost region**:
Copies of neighbouring processes' elements, kept around a local part so that stencils can read them. Its width, shape and periodicity are part of the distribution, and the library refreshes it when it is stale.
_Avoid_: halo (accepted in prose, not in the API), overlap, border
