#include <tl/tl.hpp>

#include <gtest/gtest.h>

TEST(version, headers_match_the_linked_library)
{
    EXPECT_EQ(tl::library_version(), tl::header_version);
}

TEST(version, is_comparable_at_compile_time)
{
    constexpr tl::version copy = tl::header_version;
    static_assert(copy == tl::header_version);
    static_assert(tl::version{.major = 0, .minor = 1, .patch = 0} != tl::version{.major = 0, .minor = 2, .patch = 0});
}
