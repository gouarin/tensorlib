# Binding surfaces for a lazy tensor library

Research for ticket `issues/06-binding-surfaces.md`. Read on 2026-10-30.

Question: how can a lazy, graph-based C++ tensor library expose itself to Python, Julia and Rust? What constraints does each option put on the core data model?

Sources were read from the projects' own repositories (headers, spec sources, docs sources). `data-apis.org` was not reachable from the sandbox, so the Array API standard was read from its source repository on GitHub, which is what the site is built from.

## Short answer

Every library studied uses the same two-layer shape:

1. An opaque handle to a lazy array (or lazy store), with explicit "evaluate" and "get data" calls. This is the binding surface proper. It is either a C ABI (`mlx_array`, `af_array`) or a C++ API bound with a C++-aware generator (nanobind for MLX, Cython for Legate).
2. DLPack at the edge, for zero-copy exchange with other array libraries. DLPack only describes a materialized, strided, single-device buffer. So every export forces evaluation, and a partitioned tensor cannot be exported directly: Legate exports DLPack from a `PhysicalStore` (gathered to one node), never from a `LogicalStore`.

For tensorlib this suggests: a small, versioned C ABI over an opaque dynamic-rank handle is the one surface that Python, Julia and Rust can all reach. C++-specific generators (nanobind, CxxWrap, cxx) cannot bind templated fixed-shape or fixed-rank containers without listing instantiations, so the bound type has to be the dynamic-rank tensor. Python can use nanobind over the C++ API directly; Julia (`ccall`) and Rust (`bindgen` on the C header) get the C ABI.

## 1. DLPack

DLPack is a C header with plain structs. Current version is 1.3 (`DLPACK_MAJOR_VERSION 1`, `DLPACK_MINOR_VERSION 3`). Source: [dlpack.h](https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h).

What `DLTensor` carries:

- `void* data`, a `DLDevice` (device type plus id), `int32_t ndim`, a `DLDataType` (code, bits, lanes), `int64_t* shape`, `int64_t* strides`, `uint64_t byte_offset`.
- Strides are in elements, not bytes. Since DLPack 1.2, `strides` "can not be NULL if ndim != 0"; NULL meaning "contiguous" is no longer allowed.
- The header calls it a "Plain C Tensor object, does not manage memory".

Ownership: `DLManagedTensorVersioned` wraps a `DLTensor` with a version, a `manager_ctx`, a `deleter` and `flags`. The consumer calls the producer's `deleter` when done. Flags include `DLPACK_FLAG_BITMASK_READ_ONLY` and `DLPACK_FLAG_BITMASK_IS_COPIED` ("the tensor is considered solely owned throughout its lifetime by the consumer").

Device types cover what tensorlib targets: `kDLCPU`, `kDLCUDA`, `kDLCUDAManaged`, `kDLROCM`, `kDLOneAPI`, `kDLMetal`, and others.

DLPack 1.3 adds a C-level exchange table, `DLPackExchangeAPI`, published on the Python type as `__dlpack_c_exchange_api__` (a capsule named `"dlpack_exchange_api"`). It has function pointers for allocation, export to a managed tensor, a non-owning fast export into a stack `DLTensor`, and `current_work_stream`. These functions "must not throw C++ exceptions" and report errors through a `SetError` callback.

What this means for a lazy library:

- DLPack has no notion of an unevaluated array. Export implies evaluation and a concrete buffer.
- The buffer must be strided. A buffer that is an expression, a broadcast with zero stride, or a fused temporary that does not exist yet cannot be exported without materializing it.
- One buffer on one device. See section 2 for the distributed case.
- The core must allow a buffer's lifetime to be extended by an external owner (the `deleter` and `manager_ctx`). In practice: reference-counted storage that an exported capsule can hold.
- Streams: the producer must be able to make the export safe on a consumer-given stream (section 2). The backend interface therefore needs a way to say "data is ready on stream S", or to synchronize.

## 2. Array API standard

The standard picks DLPack as its interchange protocol. Source: [data_interchange.rst](https://github.com/data-apis/array-api/blob/main/spec/draft/design_topics/data_interchange.rst). It lists what DLPack does not support:

> Use of DLPack requires that the data can be represented by a strided, in-memory layout on a single device. [...] Use cases that are not supported by DLPack include: Distributed arrays, i.e., the data residing on multiple nodes or devices, Sparse arrays [...]

In those cases the library "may raise a `BufferError` in the `__dlpack__` or `__dlpack_device__` method", or omit them entirely.

`__dlpack__(stream, max_version, dl_device, copy)` ([array_object.py](https://github.com/data-apis/array-api/blob/main/src/array_api_stubs/_draft/array_object.py)):

- `stream` is provided by the consumer "to instruct the producer to ensure that operations can safely be performed on the array (e.g., by inserting a dependency between streams via 'wait for event')". `-1` means no synchronization. On CPU only `None` is accepted.
- `copy=False` must never copy and raises `BufferError` if a copy is needed; `copy=True` always copies and sets `IS_COPIED`.
- `max_version` lets the consumer ask for `DLManagedTensorVersioned`.

Lazy evaluation is addressed directly in [lazy_eager.rst](https://github.com/data-apis/array-api/blob/main/spec/draft/design_topics/lazy_eager.rst):

- `__bool__`, `__int__`, `__float__`, `__complex__` and `__index__` must return Python scalars. A "100% lazy" library must raise; a "mostly lazy" one may trigger evaluation. Both are valid.
- Data-dependent output shapes (boolean indexing, `unique`, `nonzero`) may be omitted by graph-based libraries ([data_dependent_output_shapes.rst](https://github.com/data-apis/array-api/blob/main/spec/draft/design_topics/data_dependent_output_shapes.rst)).
- `array.shape` returns `Tuple[Optional[int], ...]`: "An array dimension must be `None` if and only if a dimension is unknown." `size` is `Optional[int]` too.

What this means for tensorlib:

- The graph's shape type should allow "unknown extent" if data-dependent operations are ever supported lazily. Otherwise those operations must evaluate eagerly or be left out.
- Python dunder methods that return scalars need a synchronous evaluate path through the binding.
- A partitioned tensor can follow the standard but cannot export through DLPack. It needs either a gather step (Legate's approach) or a separate protocol.

## 3. Python: nanobind

nanobind binds C++ directly (templates, classes, smart pointers). Relevant facts from its docs:

- `nb::ndarray<...>` exchanges data zero-copy through the buffer protocol or DLPack, with NumPy, PyTorch, TensorFlow, JAX, CuPy and MLX ([ndarray.rst](https://github.com/wjakob/nanobind/blob/master/docs/ndarray.rst)).
- On return, a framework tag picks the Python type: `nb::numpy`, `nb::pytorch`, `nb::jax`, `nb::cupy`, `nb::mlx`, `nb::array_api` (an object with buffer protocol plus `__dlpack__` and `__dlpack_device__`), or a raw `dltensor` capsule.
- Ownership: "wrapping an existing memory region without copying requires that this memory region remains valid throughout the lifetime of the created array". The `owner` argument takes a Python object (often an `nb::capsule` with a deleter) whose lifetime pins the storage.
- nanobind can target Python's stable ABI from Python 3.12 (or 3.10 in split mode), so one wheel works across later Python versions. It supports free-threaded Python 3.13+ ([why.rst](https://github.com/wjakob/nanobind/blob/master/docs/why.rst)).
- `std::shared_ptr`, `std::unique_ptr` and intrusive reference counting are supported ownership models ([ownership.rst](https://github.com/wjakob/nanobind/blob/master/docs/ownership.rst)).

MLX uses nanobind for its Python bindings, which shows the pattern for a lazy array (see section 6).

Constraints: nanobind is C++-only, so templates must be instantiated in the binding code. A Python `Tensor` type maps naturally to the dynamic-rank container with a runtime dtype. Storage must be reference counted so a capsule can keep it alive. Evaluation must be callable with the GIL released (MLX does `nb::gil_scoped_release` before `eval()`).

## 4. Julia: ccall and CxxWrap.jl

`ccall` calls a C function by symbol. It needs a C ABI and nothing else.

CxxWrap.jl wraps C++ through the `libcxxwrap-julia` C++ library ([README](https://github.com/JuliaInterop/CxxWrap.jl/blob/main/README.md)):

- "Template classes map to parametric types, for the instantiations listed in the wrapper." Each instantiation is declared with `add_type<Parametric<TypeVar<1>>>(...)` and `.apply<...>`.
- Objects returned by value or as `shared_ptr` get a Julia finalizer and are owned by Julia. `std::shared_ptr`, `std::unique_ptr` and `std::weak_ptr` are supported.
- Exceptions: a free function added directly "will be called directly using `ccall` and any exception will abort the Julia program", unless wrapped with `calling_policy::std_function`. Member functions and lambdas catch exceptions.
- `ArrayRef<T>` gives C++ a view of a Julia array of primitive types.
- The default binaries "are cross-compiled using GCC, and thus incompatible with Visual Studio C++ (MSVC)". Releases often need a matching `libcxxwrap-julia` release and a JLL rebuild; version 0.12 of the C++ part was "binary incompatible with previous versions". Binaries ship through BinaryBuilder JLL packages ([libcxxwrap-julia README](https://github.com/JuliaInterop/libcxxwrap-julia/blob/main/README.md)).

DLPack.jl gives zero-copy exchange of CPU and CUDA arrays between Julia and Python libraries, and reverses dimensions for row-major tensors ([DLPack.jl README](https://github.com/pabloferz/DLPack.jl/blob/main/README.md)).

Constraints: CxxWrap ties tensorlib's Julia package to a C++ ABI and a compiler family, and only exposes listed instantiations. MSVC is a tensorlib platform, which conflicts with CxxWrap's GCC-built binaries. A C ABI with `ccall` avoids both problems. Julia is column-major by default, so the core's layout or strides must express column-major without a copy (DLPack strides already do).

## 5. Rust: cxx and bindgen

cxx ([README](https://github.com/dtolnay/cxx/blob/master/README.md)):

- A safe bridge declared on both sides in a `#[cxx::bridge]` module. Opaque C++ types are passed behind `&`, `UniquePtr<T>` (`std::unique_ptr`) or `SharedPtr<T>` (`std::shared_ptr`). "Zero or negligible overhead".
- Generic types: only a fixed set of std types (`UniquePtr`, `SharedPtr`, `CxxVector`, ...). Template instantiations must be visible to cxx. Arbitrary C++ templates cannot be bound generically.
- `Result<T>` maps to throw/catch, "allowed as return type only".

bindgen ([C++ chapter of the book](https://github.com/rust-lang/rust-bindgen/blob/main/book/src/cpp.md)):

- Does not support "Template functions, methods of template classes and structs. We don't know which monomorphizations exist", inline functions, template specialization, SFINAE, automatic constructor/destructor calls.
- Exceptions escaping through a bindgen interface are undefined behaviour.
- It handles plain C headers well. That is the usual way to bind a C ABI.

For DLPack on the Rust side, `dlpark` implements the legacy and versioned ABIs and DLPack 1.3 C Exchange in pure Rust ([dlpark README](https://github.com/SunDoge/dlpark/blob/main/README.md)).

Constraints: header-only template C++ is close to unbindable from Rust. Rust needs either a C ABI (bindgen, then a safe wrapper crate) or a cxx bridge with a concrete, non-template C++ facade. Exceptions must not cross the boundary, so the facade must catch and return error codes.

## 6. How MLX exposes lazy arrays

- Lazy by design: "When you perform operations in MLX, no computation actually happens. Instead a compute graph is recorded. The actual computation only happens if an `eval()` is performed." ([lazy_evaluation.rst](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/lazy_evaluation.rst))
- Python bindings use nanobind over the C++ `mx::array`. Every data exit evaluates first:
  - the buffer protocol: `getbuffer` calls `a.eval()` with the GIL released; the comment says "Exceptions can not propagate through the buffer protocol" ([buffer.h](https://github.com/ml-explore/mlx/blob/main/python/src/buffer.h));
  - `to_scalar` (used for `item()` and friends) calls `a.eval()` ([convert.cpp](https://github.com/ml-explore/mlx/blob/main/python/src/convert.cpp));
  - `__dlpack__` calls `mlx_to_dlpack`, which builds an `nb::ndarray` after `arr.eval()`. `stream` and `max_version` are accepted but ignored. `__dlpack_device__` returns Metal `(8, 0)` or CPU `(1, 0)` ([array.cpp](https://github.com/ml-explore/mlx/blob/main/python/src/array.cpp)).
  - Import rejects DLPack arrays with negative strides.
- MLX C is a C API "as a bridge to bind other languages to MLX"; MLX Swift is built on it ([mlx-c README](https://github.com/ml-explore/mlx-c/blob/main/README.md)). The array is an opaque `typedef struct mlx_array_ { void* ctx; } mlx_array;` with `mlx_array_free`, `mlx_array_eval`, and typed accessors like `mlx_array_data_float32` that return NULL if the array is not evaluated ([array.h](https://github.com/ml-explore/mlx-c/blob/main/mlx/c/array.h)). Functions return `int`; errors go through a global handler set with `mlx_set_error_handler` ([error.h](https://github.com/ml-explore/mlx-c/blob/main/mlx/c/error.h)).

## 7. How ArrayFire exposes lazy arrays

- ArrayFire records JIT-able element-wise operations into an AST per variable and compiles one kernel when the value is needed, either by an explicit `eval` or when a non-JIT function needs it ([jit.md](https://github.com/arrayfire/arrayfire/blob/master/docs/details/jit.md)).
- The public surface is a C API over an opaque handle, `typedef void * af_array;`, with `af_err` return codes ([defines.h](https://github.com/arrayfire/arrayfire/blob/master/include/af/defines.h)). Lifetime is manual reference counting: `af_retain_array`, `af_release_array`. Evaluation: `af_eval`, `af_eval_multiple` (several outputs in one pass), `af_set_manual_eval_flag` ([array.h](https://github.com/arrayfire/arrayfire/blob/master/include/af/array.h)).
- Raw data access: `af_get_device_ptr` returns the device pointer and locks the buffer so the memory manager does not free it until `af_unlock_array`. `af_get_raw_ptr` notes "This pointer may be shared with other arrays". `af_get_strides`, `af_get_offset`, `af_is_linear`, `af_is_owner` expose layout ([device.h](https://github.com/arrayfire/arrayfire/blob/master/include/af/device.h), [internal.h](https://github.com/arrayfire/arrayfire/blob/master/include/af/internal.h)).
- Rank is capped: `#define AF_MAX_DIMS 4`.
- Python (arrayfire-py), Rust (arrayfire-rust) and Julia (ArrayFire.jl) all wrap the C library. arrayfire-py has an experimental, partial Array API implementation ([arrayfire-py README](https://github.com/arrayfire/arrayfire-py/blob/master/README.md)).

The C ABI with a handle is what lets one C++ library serve three languages, and what lets ArrayFire pick its backend (CUDA, OpenCL, oneAPI, CPU) at run time.

## 8. How Legate exposes lazy, distributed arrays

Legate reached end of life; the final release is `v26.06.01` ([README](https://github.com/nv-legate/legate/blob/main/README.md)). Its design is still the closest match to tensorlib's partitioned tensors. Code below is from commit `880b2756`.

- The core is C++. Python bindings are written in Cython over the C++ classes (`src/python/legate/core/_lib/*.pyx`). There is no general C ABI.
- `LogicalStore` is the lazy, partitioned object. It does not implement DLPack.
- `LogicalStore.get_physical_store(target)` "blocks the client's control flow and fetches the data for the whole array to the current node", into system memory or the framebuffer of the first GPU ([logical_store.pyx](https://github.com/nv-legate/legate/blob/main/src/python/legate/core/_lib/data/logical_store.pyx)). `__array_interface__` and `__cuda_array_interface__` on a logical store just call it.
- `PhysicalStore` implements `__dlpack__` and `__dlpack_device__` ([physical_store.pyx](https://github.com/nv-legate/legate/blob/main/src/python/legate/core/_lib/data/physical_store.pyx)). Import goes through `from_dlpack`, which turns a DLPack tensor into a `LogicalStore` attached to external memory.
- Between Legate libraries (cuPyNumeric and others), the exchange is a separate protocol, `__legate_data_interface__`: a versioned dict (`version: 2`) whose `data` maps field names to `LogicalStore` objects ([data_interface.py](https://github.com/nv-legate/legate/blob/main/src/python/legate/core/data_interface.py)). It passes the lazy, partitioned handle itself, not a buffer.

## Constraints on tensorlib's core data model

| Option | Needs from the core |
|---|---|
| DLPack export | Evaluated, strided, single-device buffer. Reference-counted storage an external deleter can hold. Strides in elements; negative and zero strides are expressible but some consumers reject them (MLX rejects negative ones on import). Stream-aware "ready" signal or a sync. |
| Array API | Runtime dtype and device on the array. Synchronous eval for scalar dunders. Optional unknown extents if data-dependent ops stay lazy. |
| nanobind | C++ types with concrete instantiations. Shared or intrusive ownership. Eval callable without the GIL. |
| CxxWrap | Listed template instantiations. Exceptions caught at the boundary. GCC-built binaries only, which conflicts with MSVC support. |
| `ccall` / bindgen | A C ABI: opaque handles, `extern "C"` functions, error codes, no exceptions or templates in signatures. |
| cxx | A concrete C++ facade (no generic templates), `unique_ptr`/`shared_ptr` handles, `Result`-style errors. |
| Partitioned tensors | Not exportable via DLPack. Needs a gather to one node (Legate `get_physical_store`) for DLPack, and a separate handle-passing protocol (Legate `__legate_data_interface__`) for zero-copy exchange between distributed libraries. Per-local-part DLPack export is possible because each local part is a strided single-device buffer. |

Concretely, the libraries above point to these requirements for tensorlib:

1. The tensor exposed to bindings is the dynamic-rank container with runtime dtype and device. Fixed-shape and fixed-rank containers stay C++-only, or are converted to dynamic rank at the boundary. ArrayFire's 4-dimension cap is a limit to avoid.
2. The handle is reference counted, and storage is separately reference counted, so an exported DLPack capsule can outlive the handle.
3. The handle can be in an unevaluated state. The ABI has explicit `eval` and `eval_many` (ArrayFire's `af_eval_multiple` matches the `c = a + b; d = a + 2*b` multi-output workload). Data accessors either evaluate or fail when the handle is unevaluated (MLX C returns NULL).
4. Errors cross the boundary as codes plus a message, never as C++ exceptions (DLPack 1.3, MLX C, ArrayFire, CxxWrap and bindgen all impose this). Lazy evaluation means an error from a fused kernel shows up at `eval`, not at the call that built the node.
5. The backend interface exposes a stream or queue and a way to make data ready on a consumer-given stream, to honour `__dlpack__(stream=...)`.
6. A partitioned tensor needs two export paths: gather to one node then DLPack, and per-local-part DLPack (with its global offset) for distributed consumers. A tensorlib-specific handle protocol, like Legate's, could pass the lazy partitioned tensor between tensorlib-based libraries without materializing it.

## Open points for later tickets

- Whether the C ABI is the only surface (Python goes through it too) or Python uses nanobind on the C++ API directly, as MLX does while also shipping MLX C. The second option gives better Python ergonomics at the cost of two surfaces to keep in sync.
- ABI versioning of the C layer: DLPack's major/minor scheme (major = ABI break) and the `prev_api` chain of `DLPackExchangeAPI` are a model.
- Whether data-dependent shapes are lazy (unknown extents in the graph) or force evaluation.
