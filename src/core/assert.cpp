#include <tl/detail/assert.hpp>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace tl::detail
{
    void index_out_of_bounds(std::size_t index, std::size_t extent) noexcept
    {
        std::fprintf(stderr, "tensorlib: index %zu is out of bounds for an extent of %zu\n", index, extent);
        std::abort();
    }
}
