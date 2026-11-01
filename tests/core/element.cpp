#include <tl/tl.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace
{
    struct dual
    {
        double value;
        double derivative;
    };

    struct mislabelled
    {
        double value;
        double derivative;
    };

    struct untraced
    {
        double value;
    };
}

template <>
struct tl::element_traits<dual>
{
    using scalar = double;
    static constexpr std::array<std::size_t, 1> extents{2};
};

template <>
struct tl::element_traits<mislabelled>
{
    using scalar = double;
    static constexpr std::array<std::size_t, 1> extents{3}; // claims three components
};

namespace
{
    static_assert(tl::element<double> && tl::element<std::uint8_t> && tl::element<bool>);
    static_assert(tl::element<tl::fixed<float, 3>> && tl::element<tl::fixed<double, 2, 3, 4, 5>>);
    static_assert(tl::element<tl::fixed<double, 2, 2, 2, 2, 2, 2, 2, 2>>);
    static_assert(tl::element_traits<tl::fixed<float, 2, 3>>::extents == std::array<std::size_t, 2>{2, 3});

    // A user type is an element once its traits describe its layout exactly.
    static_assert(tl::element<dual>);
    static_assert(!tl::element<mislabelled>);
    static_assert(!tl::element<untraced>);
    static_assert(!tl::element<long double>);
}
