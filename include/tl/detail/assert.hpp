#pragma once

#include <cstddef>

namespace tl::detail
{
    // Reports an out-of-bounds index and aborts. Compiled once, in the library.
    [[noreturn]] void index_out_of_bounds(std::size_t index, std::size_t extent) noexcept;

    // Host-side bounds check, on only with TL_ENABLE_ASSERTIONS. In constant evaluation an
    // out-of-bounds index is a compile error either way.
    constexpr void check_index(std::size_t index, std::size_t extent) noexcept
    {
#ifdef TL_ENABLE_ASSERTIONS
        if (index >= extent)
        {
            index_out_of_bounds(index, extent);
        }
#else
        static_cast<void>(index);
        static_cast<void>(extent);
#endif
    }
}
