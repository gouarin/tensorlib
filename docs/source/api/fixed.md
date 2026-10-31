# Fixed-shape tensors

`tl::fixed<T, N...>` is a small tensor whose shape is known at compile time, such as a 3-vector or a 2×2 matrix.
It stores its elements inline, in C order, and every operation on it is evaluated at once and can run in constant expressions.
It serves in local computations and as the element type of larger tensors.

```{literalinclude} ../../../examples/fixed.cpp
:language: cpp
```

Values are written as a flat list in C order, `tl::fixed<double, 2, 2>{1., 2., 3., 4.}`, or with nested braces through `tl::fixed_from`.
As many indices as the rank give an element; fewer give a `tl::fixed_view` of a block, and writing through it writes into the `fixed`.

Arithmetic works element by element between two `fixed` of the same shape and element type, or with a scalar, which takes the element type of the `fixed`: `tl::fixed<float, 2>{} * 0.5` is a `tl::fixed<float, 2>`.
Operands of different shapes or element types do not compile.
`==` and `!=` compare whole values; the Array API comparisons (`tl::equal`, `tl::less`, ...) compare element by element and return a `tl::fixed<bool, N...>`.

## fixed

```{doxygenstruct} tl::fixed
:members:
```

## fixed_view

```{doxygenclass} tl::fixed_view
:members:
```

## fixed_from

```{doxygenfunction} tl::fixed_from(const T (&values)[N0][N1])
```

```{doxygenfunction} tl::fixed_from(const T (&values)[N0][N1][N2])
```

```{doxygenfunction} tl::fixed_from(const T (&values)[N0][N1][N2][N3])
```

## Comparisons

```{doxygenfunction} tl::operator==(const A &a, const B &b)
```

```{doxygenfunction} tl::equal
```

```{doxygenfunction} tl::not_equal
```

```{doxygenfunction} tl::less
```

```{doxygenfunction} tl::less_equal
```

```{doxygenfunction} tl::greater
```

```{doxygenfunction} tl::greater_equal
```

## Arithmetic

```{doxygenfunction} tl::operator+(const A &a, const B &b)
```

```{doxygenfunction} tl::operator-(const A &a, const B &b)
```

```{doxygenfunction} tl::operator*(const A &a, const B &b)
```

```{doxygenfunction} tl::operator/(const A &a, const B &b)
```

```{doxygenfunction} tl::operator-(const A &a)
```

```{doxygenfunction} tl::operator+=
```

```{doxygenfunction} tl::operator-=
```

```{doxygenfunction} tl::operator*=
```

```{doxygenfunction} tl::operator/=
```
