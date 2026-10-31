// Small tensors whose shape is known at compile time: a vector, a matrix built from
// nested braces, arithmetic, and a view of one row.
#include <tl/tl.hpp>

int main()
{
    using vec2 = tl::fixed<double, 2>;

    constexpr vec2 u{1., 2.};
    tl::fixed<double, 2, 2> m = tl::fixed_from({
        {1., 0.},
        {0., 1.}
    });

    m(1) += 2. * u;        // m(1) is a view of row 1: this writes into m
    const vec2 row = m(1); // a copy of the row

    return row == vec2{2., 5.} && m(1, 1) == 5. ? 0 : 1;
}
