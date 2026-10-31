# Queue-ordered memory resources, with Umpire required by the Kokkos backend

Tensor storage comes from one memory resource per memory space. A resource is a small type-erased interface, `allocate(bytes, alignment, queue)` and `deallocate(ptr, bytes, alignment, queue)`, ordered by queue like the CCCL `cuda::mr` resources. When a buffer dies, the core makes the queue of its last access wait on its other pending events, then deallocates on that queue, so memory is reused in stream order and the host never blocks. The default resource is an Umpire `ResourceAwarePool`, and the Kokkos backend requires Umpire. The serial backend uses aligned `operator new`.

Kokkos users will expect tensors to own `Kokkos::View`s. We rejected that because every View free fences the device, and Kokkos has no pool for multi-gigabyte buffers allocated from the host. We also rejected keeping Umpire optional with a plain allocator as fallback. Without a pool, fresh outputs cost 2 to 6.6 times more in the multi-output fusion prototype, and every free synchronizes, which breaks asynchronous evaluation. Users would get a slow build without being told why. Writing our own fallback pool would mean maintaining a second allocator next to Umpire. Finally, we rejected using `umpire::Allocator` as the public extension point, because RMM and CCCL pools could not plug in and the public API would depend on Umpire.

## Consequences

- Umpire is a hard dependency of the Kokkos backend, in CMake and in the Spack and conda-forge recipes.
- An application can hand tensorlib its own pool (`tl::umpire_resource`, or CCCL and RMM adapters in optional headers), so two caches never compete for one GPU. The resource is set before the first allocation in its space and cannot change afterwards.
- A user resource that is not queue-ordered declares it. The core then keeps its frees in a list that it polls at each allocation and evaluation.
- Adaptors such as `tl::staggered(resource)` wrap any resource, including a user one.
