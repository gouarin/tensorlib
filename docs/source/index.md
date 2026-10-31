# tensorlib

tensorlib is a C++20 library of tensors close to NumPy and the Array API standard.
It records operations, fuses them into kernels at run time across several outputs, and runs them on Kokkos, from CPU threads to GPUs, with tensors partitioned over MPI.

The pages below document the interface that is implemented.
The design of the whole library is in the [architecture spec](https://github.com/gouarin/tensorlib/blob/main/.scratch/core-architecture/spec.md).

```{toctree}
:maxdepth: 2

api/index
```

To report a problem in this documentation or suggest a change, open an issue on [GitHub](https://github.com/gouarin/tensorlib/issues).
