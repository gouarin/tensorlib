#include <tl/tl.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace
{
    using vec3 = tl::fixed<double, 3>;
    using mat2 = tl::fixed<double, 2, 2>;

    // Storage and indexing

    static_assert(std::is_aggregate_v<vec3> && std::is_trivially_copyable_v<vec3>);
    static_assert(sizeof(mat2) == 4 * sizeof(double));
    static_assert(mat2::shape() == std::array<std::size_t, 2>{2, 2});
    static_assert(vec3{}(0) == 0. && vec3{}(2) == 0.);

    constexpr mat2 m{1., 2., 3., 4.};
    static_assert(m(0, 1) == 2. && m(1, 0) == 3.); // C order
// MSVC before 19.50 cannot read nested braces bound to an array reference in constant
// evaluation (C2131); fixed_from works there at run time, tested below.
#if !defined(_MSC_VER) || _MSC_VER >= 1950
    static_assert(tl::fixed_from({
                      {1., 2.},
                      {3., 4.}
    })
                  == m);
    static_assert(tl::fixed_from({
                      {{1, 2}, {3, 4}},
                      {{5, 6}, {7, 8}}
    })
                  == tl::fixed<int, 2, 2, 2>{1, 2, 3, 4, 5, 6, 7, 8});
#endif

    // Arithmetic

    constexpr vec3 a{1., 2., 3.};
    constexpr vec3 b{4., 5., 6.};
    static_assert(a + b == vec3{5., 7., 9.});
    static_assert(b - a == vec3{3., 3., 3.});
    static_assert(a * b == vec3{4., 10., 18.});
    static_assert(b / a == vec3{4., 2.5, 2.});
    static_assert(-a == vec3{-1., -2., -3.});

    // A scalar is broadcast over the components, on either side.
    static_assert(a * 2. == vec3{2., 4., 6.});
    static_assert(2. * a == vec3{2., 4., 6.});
    static_assert(1. - a == vec3{0., -1., -2.});
    static_assert(a / 2 == vec3{0.5, 1., 1.5});

    // ... and adopts the element type, as Python scalars do in the Array API.
    static_assert(std::is_same_v<decltype(tl::fixed<float, 2>{} * 0.5), tl::fixed<float, 2>>);
    static_assert(tl::fixed<std::int8_t, 2>{100, 20} + 27 == tl::fixed<std::int8_t, 2>{127, 47});

    constexpr vec3 compound()
    {
        vec3 v{1., 2., 3.};
        v += a;
        v *= 2.;
        v -= vec3{1., 1., 1.};
        v /= 2.;
        return v;
    }

    static_assert(compound() == vec3{1.5, 3.5, 5.5});

    // Operands must have the same shape and element type, and booleans do no arithmetic.
    template <class A, class B>
    concept addable = requires(A x, B y) { x + y; };
    static_assert(!addable<vec3, tl::fixed<double, 2>>);
    static_assert(!addable<vec3, tl::fixed<float, 3>>);
    static_assert(!addable<tl::fixed<bool, 2>, tl::fixed<bool, 2>>);

    // Comparisons compare whole values.

    static_assert(a == vec3{1., 2., 3.});
    static_assert(a != b);
    template <class A>
    concept ordered = requires(A x) { x < x; };
    static_assert(!ordered<vec3>);

    // Element-wise comparisons are the Array API functions.
    using mask3 = tl::fixed<bool, 3>;
    static_assert(tl::equal(a, vec3{1., 0., 3.}) == mask3{true, false, true});
    static_assert(tl::not_equal(a, 2.) == mask3{true, false, true});
    static_assert(tl::less(a, b) == mask3{true, true, true});
    static_assert(tl::less_equal(2., a) == mask3{false, true, true});
    static_assert(tl::greater(a, 2.) == mask3{false, false, true});
    static_assert(tl::greater_equal(m(1), tl::fixed<double, 2>{3., 5.}) == tl::fixed<bool, 2>{true, false});
    static_assert(tl::equal(mask3{true, false, true}, true) == mask3{true, false, true});

    // Views

    static_assert(std::is_same_v<decltype(std::declval<mat2&>()(1)), tl::fixed_view<double, 2>>);
    static_assert(std::is_same_v<decltype(std::declval<const mat2&>()(1)), tl::fixed_view<const double, 2>>);
    static_assert(m(1) == tl::fixed<double, 2>{3., 4.});
    static_assert(m(1)(0) == 3.);

    constexpr mat2 write_row()
    {
        mat2 r{};
        r(1) = tl::fixed<double, 2>{3., 4.}; // writes into r
        r(0) = 7.;                           // a scalar fills the block
        r(0) += r(1);
        return r;
    }

    static_assert(write_row() == mat2{10., 11., 3., 4.});

    constexpr tl::fixed<double, 2> copy_row()
    {
        mat2 r{1., 2., 3., 4.};
        const tl::fixed<double, 2> row = r(1); // a value: later writes to r do not reach it
        r(1)                           = 0.;
        return row;
    }

    static_assert(copy_row() == tl::fixed<double, 2>{3., 4.});

    constexpr mat2 views_share_elements()
    {
        mat2 r{};
        auto row  = r(1);
        auto same = row; // copying a view gives another view of the same elements
        same(0)   = 5.;
        return r;
    }

    static_assert(views_share_elements()(1, 0) == 5.);

    // A view of a const fixed is read-only, and a temporary cannot be viewed.
    static_assert(!std::is_assignable_v<tl::fixed_view<const double, 2>, tl::fixed<double, 2>>);
    template <class F>
    concept viewable_temporary = requires(F x) { std::move(x)(0); };
    static_assert(!viewable_temporary<mat2>);
    static_assert(std::is_same_v<decltype(mat2{}(0, 0)), double>);
}

TEST(fixed, writes_through_views_at_run_time)
{
    mat2 r{};
    for (std::size_t i = 0; i < 2; ++i)
    {
        r(i) = static_cast<double>(i + 1);
    }
    EXPECT_EQ(r, (mat2{1., 1., 2., 2.}));
}

TEST(fixed, builds_from_nested_braces_at_run_time)
{
    EXPECT_EQ(tl::fixed_from({
                  {1., 2.},
                  {3., 4.}
    }),
              m);
    EXPECT_EQ(tl::fixed_from({
                  {{1, 2}, {3, 4}},
                  {{5, 6}, {7, 8}}
    }),
              (tl::fixed<int, 2, 2, 2>{1, 2, 3, 4, 5, 6, 7, 8}));
}

TEST(fixed, checks_indices_with_assertions)
{
#ifdef TL_ENABLE_ASSERTIONS
    mat2 r{};
    const std::size_t out = 2;
    EXPECT_DEATH(static_cast<void>(r(out, 0)), "index 2 is out of bounds for an extent of 2");
#else
    GTEST_SKIP() << "built without TL_ENABLE_ASSERTIONS";
#endif
}
