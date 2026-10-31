// A user element type: a dual number stored as two doubles. Its traits say how it is laid
// out, so tensors can store it and kernels see it as two lanes of double.
#include <tl/tl.hpp>

#include <array>
#include <cstddef>

struct dual
{
    double value;
    double derivative;
};

template <>
struct tl::element_traits<dual>
{
    using scalar = double;
    static constexpr std::array<std::size_t, 1> extents{2};
};

static_assert(tl::element<dual>);

int main()
{
    return 0;
}
