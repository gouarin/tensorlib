# Element types

An element type is what one element of a tensor holds: a scalar, a `tl::fixed` of scalars, or your own type stored exactly like one.
`tl::element_traits` says how a type is stored, as a scalar repeated over a component shape, and the `tl::element` concept checks that the type matches its traits.
Kernels see an element type through that description: a `tl::fixed<double, 3>` is three lanes of `double`.

To store your own type in tensors, specialise `tl::element_traits` for it:

```{literalinclude} ../../../examples/element_traits.cpp
:language: cpp
```

## scalar

```{doxygenconcept} tl::scalar
```

## element

```{doxygenconcept} tl::element
```

## element_traits

```{doxygenstruct} tl::element_traits
```

```{doxygenstruct} tl::element_traits< T >
:members:
```

```{doxygenstruct} tl::element_traits< fixed< T, N... > >
:members:
```
