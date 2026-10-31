// Checks that the toolchain provides what the public headers rely on at the floor
// (GCC 11.1, Clang 16, Apple Clang 16, MSVC 19.30). Clang 15 and Apple Clang 15 fail the line check below:
// it evaluates a defaulted std::source_location::current() at the callee.
#include <source_location>
#include <string_view>

#include <gtest/gtest.h>

namespace
{
    // The way tl::site captures its caller: a defaulted std::source_location argument.
    std::source_location caller(std::source_location site = std::source_location::current())
    {
        return site;
    }
}

TEST(toolchain, source_location_captures_the_caller)
{
    const std::source_location here = caller();
    EXPECT_EQ(here.line(), __LINE__ - 1);
    EXPECT_NE(std::string_view(here.file_name()).find("toolchain.cpp"), std::string_view::npos);
}
