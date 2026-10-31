# Views are references distinct from tensors, and `+=` through an index view accumulates

Slicing returns a `tl::view<T, Rank>`, a separate type with reference semantics: copying a view gives another view of the same owner, and assigning to a view writes elements. A `tl::tensor` stays a value ([ADR 0001](0001-value-semantics-with-copy-on-write.md)), converts implicitly to a view of itself, and `tl::view<T>` is the type of output parameters, so a function such as ponio's `f(t, y, tl::view<double> dy)` accepts a whole tensor or a slice. A view holds a shared reference to its owner's state, so it never dangles while a pending window still holds it.

We rejected a single tensor type whose slices stay bound to their owner. Either copying such a tensor would keep the binding, so `auto c = x;` would stop being a value copy whenever `x` came from a slice, or it would drop it, so passing a slice by value would lose writes in silence. We also rejected non-owning views in the style of `std::span`, because in a lazy library a view often outlives the scope of its owner.

Writing `u(idx) += e` through an integer index view accumulates duplicate indices (a scatter-add, with atomics or colouring chosen by the lowering). NumPy keeps only the last write for duplicates in `a[idx] += v`, a frequent source of bugs. We chose accumulation because flux schemes, samurai's among them, add the contributions of two interfaces to one cell and should be able to write exactly that.

## Consequences

- Code ported from NumPy that relies on `a[idx] += v` ignoring duplicates changes meaning. The Python bindings must decide which rule `a[idx] += v` follows.
- Functions that only read take `const tl::tensor<T>&` or a `tl::tensor_like` argument; functions that write take `tl::view<T>`.
- `tl::reshape` returns a value, never a view, so whether a write reaches the source never depends on run-time strides.
