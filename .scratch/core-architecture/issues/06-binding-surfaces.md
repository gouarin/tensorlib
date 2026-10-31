# Binding surfaces for a lazy tensor library

Type: research
Status: resolved

## Question

How can a lazy, graph-based C++ tensor library expose itself to Python, Julia and Rust? Cover a stable C ABI, DLPack, the Array API standard, nanobind, CxxWrap or ccall for Julia, cxx or bindgen for Rust, and how MLX, ArrayFire and Legate expose lazy arrays. What constraints does each option put on the core data model?

## Answer

MLX, ArrayFire and Legate all use the same two layers: an opaque, reference-counted handle to a lazy array with explicit eval calls (a C ABI in MLX C and ArrayFire, nanobind or Cython over C++ elsewhere), plus DLPack at the edge for zero-copy exchange.
DLPack and the Array API only cover an evaluated, strided buffer on one device. Export always forces evaluation, and a partitioned tensor must be gathered (Legate `get_physical_store`) or exported one local part at a time. Legate passes lazy distributed stores between libraries through its own `__legate_data_interface__`.
nanobind, CxxWrap, cxx and bindgen cannot bind arbitrary templates, and exceptions must not cross the boundary. So the bound type is the dynamic-rank tensor with runtime dtype and device, behind a small versioned C ABI with error codes (reached by Julia `ccall` and Rust bindgen). Python can use nanobind directly. CxxWrap's GCC-only binaries conflict with MSVC support.
Core needs: separately refcounted storage that a DLPack deleter can hold, an unevaluated handle state with `eval`/`eval_many`, stream-aware readiness for `__dlpack__(stream=...)`, and optionally unknown extents for data-dependent shapes.

Findings: [research/06-binding-surfaces.md](../research/06-binding-surfaces.md)
