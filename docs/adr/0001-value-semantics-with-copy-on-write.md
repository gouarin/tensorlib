# Value semantics with copy-on-write for tensors

Copying a tensor (`auto c = a;`) behaves like copying a `std::vector`: later writes to one never show up in the other. Internally the storage is reference counted and shared until the first write, which copies it. Views refer to their owning tensor, not to the raw buffer, so a write through a view detaches the owner from its copies first. We chose this over the reference semantics of `Kokkos::View`, NumPy and MLX because the primary users are C++ developers, for whom hidden aliasing on `auto c = a;` is a classic bug. We also rejected eager value copies, which cost too much on GPUs and partitioned data and fight a lazy graph that holds references to its inputs. Shared, reference-counted storage is also what DLPack export and the lazy graph need.

## Consequences

- Every write path, including generated kernels and `local_part()` writes, must check that the storage is unique before writing and copy it if not.
- Code ported from Kokkos or NumPy that relies on `b = a` aliasing must use an explicit view instead.
